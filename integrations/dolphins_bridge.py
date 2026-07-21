"""
zebrain-lab/Dolphins 集成桥接
=============================
对接 GitHub 开源项目 zebrain-lab/Dolphins 的检测和提取功能
项目地址: https://github.com/zebrain-lab/Dolphins
"""
import sys, os
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DOLPHINS_PATH = None


def find_dolphins():
    """自动查找 Dolphins 项目路径"""
    global DOLPHINS_PATH
    if DOLPHINS_PATH:
        return DOLPHINS_PATH

    candidates = [
        Path('D:/Dolphins'),
        Path(os.path.expanduser('~/Dolphins')),
        Path('D:/dolphin-whistle-analyzer/Dolphins'),
    ]
    for c in candidates:
        if (c / 'AutomaticExtraction').exists():
            DOLPHINS_PATH = c
            return DOLPHINS_PATH
    return None


def dolphins_detect(wav_path: str, model_path: str = None) -> list:
    """
    调用 Dolphins 的 VGG 检测模型
    Returns:
        [(start_time, end_time, confidence), ...]
    """
    dolphins = find_dolphins()
    if not dolphins:
        print('[Dolphins] Not found. Install: git clone https://github.com/zebrain-lab/Dolphins')
        return []

    sys.path.insert(0, str(dolphins / 'AutomaticExtraction'))
    try:
        from src.detection.predict import predict_whistles
        from src.detection.model import load_detection_model

        if not model_path:
            model_path = str(dolphins / 'AutomaticExtraction' / 'models' / 'model_vgg.h5')

        model = load_detection_model(model_path)
        results = predict_whistles(wav_path, model)
        return [(r['initial_point'], r['finish_point'], r['confidence'])
                for r in results]
    except ImportError as e:
        print(f'[Dolphins] Import error: {e}. Need TensorFlow 2.x.')
        return []
    except Exception as e:
        print(f'[Dolphins] Detection error: {e}')
        return []


def dolphins_extract(wav_path: str, predictions_csv: str, output_csv: str = None) -> tuple:
    """
    调用 Dolphins 的轮廓提取
    Returns:
        (times, freqs) arrays
    """
    dolphins = find_dolphins()
    if not dolphins:
        return np.array([]), np.array([])

    sys.path.insert(0, str(dolphins / 'AutomaticExtraction'))
    try:
        from src.extraction.contour import extract_whistle_contours_from_file
        times, freqs = extract_whistle_contours_from_file(
            wav_path, predictions_csv, output_csv,
            min_freq=500, max_freq=96000,
        )
        return times, freqs
    except Exception as e:
        print(f'[Dolphins] Extraction error: {e}')
        return np.array([]), np.array([])


def dolphins_is_available() -> bool:
    return find_dolphins() is not None
