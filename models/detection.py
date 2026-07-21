"""
哨声检测深度学习模型
基于 zebrain-lab/Dolphins 的 VGG 架构 + DYOC 的 YOLO 方案
支持：
  - VGG16 声谱图 patch 二分类（哨声/非哨声）
  - 从预训练权重加载（兼容 Dolphins 项目的 model_vgg.h5）
  - 持续 Fine-tune
"""
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class WhistleDetector:
    """哨声检测器 — VGG16 声谱图 patch 分类"""

    def __init__(self, model_path: str = None):
        self.model = None
        self.input_size = (224, 224)
        self.threshold = 0.5
        self._try_load_model(model_path)

    def _try_load_model(self, model_path):
        """尝试加载预训练模型"""
        if model_path and Path(model_path).exists():
            try:
                # 尝试 TensorFlow/Keras
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                self.backend = 'tensorflow'
                return
            except ImportError:
                pass
            except Exception as e:
                print(f'[Detector] TensorFlow load failed: {e}')

            try:
                # 尝试 PyTorch
                import torch
                self.model = torch.load(model_path, map_location='cpu')
                self.backend = 'pytorch'
                return
            except ImportError:
                pass
            except Exception as e:
                print(f'[Detector] PyTorch load failed: {e}')

        # 无预训练模型，使用简化版（纯 NumPy 占位，后续可替换为真实模型）
        self.backend = 'numpy'
        print('[Detector] No pre-trained model found. Using rule-based detection fallback.')

    def predict_patch(self, spec_patch: np.ndarray) -> float:
        """
        预测单个声谱图 patch 是否包含哨声
        Args:
            spec_patch: 2D array (224, 224) dB 谱图
        Returns:
            置信度 (0-1)
        """
        if self.backend == 'tensorflow':
            import tensorflow as tf
            x = np.expand_dims(spec_patch, axis=(0, -1))
            pred = self.model.predict(x, verbose=0)[0][0]
            return float(pred)

        elif self.backend == 'pytorch':
            import torch
            x = torch.from_numpy(spec_patch).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                pred = self.model(x)
            return float(torch.sigmoid(pred).item())

        else:
            # 简化规则：检测 patch 中是否有明显能量峰值
            return self._rule_based_confidence(spec_patch)

    def _rule_based_confidence(self, patch: np.ndarray) -> float:
        """基于规则的简单置信度评估"""
        if patch.size == 0:
            return 0.0
        p95 = np.percentile(patch, 95)
        p50 = np.percentile(patch, 50)
        dynamic_range = p95 - p50
        # 动态范围越大 → 越可能有哨声
        conf = min(1.0, max(0.0, dynamic_range / 40.0))
        return conf

    def detect_regions(self, S_db: np.ndarray, f: np.ndarray, t: np.ndarray,
                       stride: int = 112) -> list:
        """
        滑动窗口扫描全尺寸声谱图，返回检测到的哨声区域
        Args:
            S_db: 声谱图 (freq_bins, time_bins) dB
        Returns:
            [(start_time, end_time, low_freq, high_freq, confidence), ...]
        """
        regions = []
        h, w = S_db.shape
        th, tw = self.input_size

        for i in range(0, h - th + 1, stride):
            for j in range(0, w - tw + 1, stride):
                patch = S_db[i:i+th, j:j+tw]
                conf = self.predict_patch(patch)
                if conf > self.threshold:
                    # 将 patch 坐标映射回时间-频率
                    t_start = t[j]
                    t_end = t[min(j + tw, w - 1)]
                    f_low = f[i]
                    f_high = f[min(i + th, h - 1)]
                    regions.append((t_start, t_end, f_low, f_high, conf))

        # NMS (Non-Maximum Suppression) 简版
        regions = self._simple_nms(regions)
        return regions

    def _simple_nms(self, regions: list, iou_threshold: float = 0.3) -> list:
        """简单的 IoU 非极大值抑制"""
        if not regions:
            return []
        regions = sorted(regions, key=lambda r: r[4], reverse=True)
        kept = []
        for r in regions:
            overlap = False
            for k in kept:
                if self._iou(r, k) > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                kept.append(r)
        return kept

    def _iou(self, a, b):
        """计算两个边界框的 IoU"""
        t_min = max(a[0], b[0])
        t_max = min(a[1], b[1])
        f_min = max(a[2], b[2])
        f_max = min(a[3], b[3])
        if t_min >= t_max or f_min >= f_max:
            return 0.0
        intersection = (t_max - t_min) * (f_max - f_min)
        area_a = (a[1] - a[0]) * (a[3] - a[2])
        area_b = (b[1] - b[0]) * (b[3] - b[2])
        return intersection / (area_a + area_b - intersection + 1e-12)

    def fine_tune(self, patches: list, labels: list, epochs: int = 10):
        """
        使用用户修正数据微调模型
        Args:
            patches: 声谱图 patch 列表
            labels: 0/1 标签列表
        """
        if self.backend == 'numpy':
            print('[Detector] Fine-tune not available without PyTorch/TensorFlow.')
            return

        print(f'[Detector] Fine-tuning on {len(patches)} samples for {epochs} epochs...')
        # 具体实现取决于后端
        if self.backend == 'pytorch':
            self._finetune_pytorch(patches, labels, epochs)
        elif self.backend == 'tensorflow':
            self._finetune_tensorflow(patches, labels, epochs)

    def _finetune_pytorch(self, patches, labels, epochs):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        X = torch.from_numpy(np.array(patches)).float().unsqueeze(1)
        y = torch.from_numpy(np.array(labels)).float()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model(batch_x).squeeze()
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'  Epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}')

    def _finetune_tensorflow(self, patches, labels, epochs):
        import tensorflow as tf
        X = np.array(patches)[..., np.newaxis]
        y = np.array(labels)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                           loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=epochs, batch_size=16, verbose=1)

    def save(self, path: str):
        """保存模型权重"""
        if self.backend == 'pytorch':
            import torch
            torch.save(self.model, path)
        elif self.backend == 'tensorflow':
            self.model.save(path)
        print(f'[Detector] Model saved to {path}')


# 全局单例
_detector = None


def get_detector(model_path: str = None) -> WhistleDetector:
    global _detector
    if _detector is None:
        _detector = WhistleDetector(model_path)
    return _detector
