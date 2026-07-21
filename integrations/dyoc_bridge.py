"""
DYOC (Draw Your Own Contours) 集成桥接
=======================================
对接 Lehnhoff et al. (2025) 的 YOLOv8 + ResNet18 半自动轮廓提取
论文: Scientific Reports (2025)
"""
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def dyoc_detect_contours(wav_path: str) -> list:
    """
    使用 YOLOv8 在声谱图上检测哨声边界框
    Returns:
        [(start_time, end_time, low_freq, high_freq, confidence), ...]
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print('[DYOC] ultralytics not installed. pip install ultralytics')
        return _dyoc_fallback(wav_path)

    # 尝试加载预训练 YOLO 模型（用户需自行下载或训练）
    model_path = Path(__file__).parent.parent / 'models' / 'weights' / 'yolo_whistle.pt'
    if not model_path.exists():
        print('[DYOC] No YOLO model found at', model_path)
        return _dyoc_fallback(wav_path)

    try:
        model = YOLO(str(model_path))
        results = model(wav_path)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))
        return detections
    except Exception as e:
        print(f'[DYOC] Inference error: {e}')
        return _dyoc_fallback(wav_path)


def _dyoc_fallback(wav_path: str) -> list:
    """DYOC 不可用时的回退方案"""
    from run_pipeline import generate_spectrogram, detect_whistle_regions
    S_db, f, t, _ = generate_spectrogram(wav_path, n_fft=4096, overlap=0.75)
    regions = detect_whistle_regions(S_db, f, t, threshold_db=-115)
    return [(r[0], r[1], r[2], r[3], 0.7) for r in regions]


def dyoc_is_available() -> bool:
    try:
        import ultralytics
        return True
    except ImportError:
        return False
