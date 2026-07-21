"""
哨声六类型分类器 — ResNet18
基于 dolphin-acoustics-vip 的 ML 分类方案
支持：
  - ResNet18 六类型分类（Flat/Down/Rise/Convex/U-shaped/Sine）
  - 从归一化轮廓图像预测类型
  - Fine-tune + LoRA 微调
"""
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class WhistleClassifier:
    """哨声六类型分类器 — ResNet18"""

    TYPE_MAP = {0: 'Flat', 1: 'Down', 2: 'Rise', 3: 'Convex', 4: 'U-shaped', 5: 'Sine'}
    TYPE_MAP_REV = {v: k for k, v in TYPE_MAP.items()}

    def __init__(self, model_path: str = None):
        self.model = None
        self.input_size = (128, 128)
        self.backend = 'numpy'
        self._try_load_model(model_path)

    def _try_load_model(self, model_path):
        if model_path and Path(model_path).exists():
            try:
                import torch
                self.model = torch.load(model_path, map_location='cpu')
                self.backend = 'pytorch'
                print('[Classifier] PyTorch model loaded.')
                return
            except ImportError:
                pass
            except Exception as e:
                print(f'[Classifier] Load failed: {e}')

        # 尝试创建新模型
        try:
            self._build_model()
        except ImportError:
            print('[Classifier] PyTorch not available. Using rule-based classification.')

    def _build_model(self):
        """构建 ResNet18 模型"""
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18

        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 6)
        self.backend = 'pytorch'
        print('[Classifier] ResNet18 model built.')

    def contour_to_image(self, freqs: np.ndarray, times: np.ndarray = None) -> np.ndarray:
        """
        将频率轮廓转为归一化图像 (128, 128)
        用于 ML 模型输入
        """
        from scipy.ndimage import zoom

        if len(freqs) < 5:
            return np.zeros((128, 128), dtype=np.float32)

        # 创建画布
        h, w = 128, 128
        img = np.zeros((h, w), dtype=np.float32)

        # 归一化频率到 [0, 1]
        f_min, f_max = np.min(freqs), np.max(freqs)
        if f_max - f_min < 100:
            f_norm = np.full_like(freqs, 0.5)
        else:
            f_norm = (freqs - f_min) / (f_max - f_min)

        # 映射到图像坐标
        x = np.linspace(0, w - 1, len(freqs))
        y = (1 - f_norm) * (h - 1)  # 翻转 y 轴（上=高频）

        # 画线
        for i in range(len(x) - 1):
            x0, y0 = int(x[i]), int(y[i])
            x1, y1 = int(x[i + 1]), int(y[i + 1])
            # Bresenham-like
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            steps = max(dx, dy)
            if steps == 0:
                steps = 1
            for s in range(steps + 1):
                px = int(x0 + (x1 - x0) * s / steps)
                py = int(y0 + (y1 - y0) * s / steps)
                if 0 <= px < w and 0 <= py < h:
                    img[py, px] = 1.0

        # 轻微膨胀（让线条更明显）
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=0.8)
        img = np.clip(img * 2.0, 0, 1)

        return img.astype(np.float32)

    def predict(self, freqs: np.ndarray) -> tuple:
        """
        预测哨声类型
        Returns:
            (type_name, confidence, probabilities_dict)
        """
        if self.backend == 'pytorch' and self.model is not None:
            return self._predict_torch(freqs)
        return self._predict_rules(freqs)

    def _predict_torch(self, freqs: np.ndarray) -> tuple:
        import torch
        img = self.contour_to_image(freqs)
        x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        pred_idx = int(np.argmax(probs))
        return self.TYPE_MAP[pred_idx], float(probs[pred_idx]), {
            self.TYPE_MAP[i]: float(p) for i, p in enumerate(probs)
        }

    def _predict_rules(self, freqs: np.ndarray) -> tuple:
        """回退到规则分类"""
        from whistle_classifier import classify_whistle
        dummy_times = np.arange(len(freqs)) * 0.001
        whistle_type, info = classify_whistle(dummy_times, freqs)
        conf = 0.7  # 规则分类默认置信度
        return whistle_type, conf, {whistle_type: conf}

    def fine_tune(self, contours: list, labels: list, epochs: int = 15):
        """
        微调分类模型
        Args:
            contours: 频率轮廓列表 (每个是 1D array)
            labels: 类型标签列表 ['Flat', 'Down', ...]
        """
        if self.backend != 'pytorch':
            print('[Classifier] Fine-tune requires PyTorch.')
            return

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        images = np.array([self.contour_to_image(c) for c in contours])
        label_ids = np.array([self.TYPE_MAP_REV.get(l, 0) for l in labels])

        X = torch.from_numpy(images).float().unsqueeze(1)
        y = torch.from_numpy(label_ids).long()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        print(f'[Classifier] Fine-tuning on {len(contours)} samples...')
        self.model.train()
        for epoch in range(epochs):
            total_loss, correct = 0, 0
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (out.argmax(1) == by).sum().item()
            acc = correct / len(dataset)
            print(f'  Epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f} acc={acc:.3f}')

    def save(self, path: str):
        if self.backend == 'pytorch':
            import torch
            torch.save(self.model, path)
            print(f'[Classifier] Model saved to {path}')


_classifier = None


def get_classifier(model_path: str = None) -> WhistleClassifier:
    global _classifier
    if _classifier is None:
        _classifier = WhistleClassifier(model_path)
    return _classifier
