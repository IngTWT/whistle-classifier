"""
哨声六类型分类器
基于 Wang et al. (2013) JASA 的定义：
Flat, Down, Rise, U-shaped, Convex, Sine
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, List


def smooth_contour(freq: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    """平滑频率轮廓，减少噪声干扰"""
    if len(freq) < window:
        return freq
    try:
        return savgol_filter(freq, min(window, len(freq) - (1 - len(freq) % 2)), polyorder)
    except Exception:
        return freq


def detect_inflection_points(freq: np.ndarray, min_span_hz: float = 500.0) -> Tuple[int, List[int]]:
    """
    检测拐点：斜率符号变化且两侧频率跨度均 > min_span_hz

    Returns:
        (拐点数量, 拐点索引列表)
    """
    if len(freq) < 3:
        return 0, []

    # 计算一阶差分
    diff = np.diff(freq)

    # 找斜率符号变化的位置
    inflection_indices = []
    for i in range(1, len(diff)):
        if diff[i-1] * diff[i] < 0:  # 符号变化
            inflection_indices.append(i)

    # 过滤：两侧频率跨度需 > min_span_hz
    valid_inflections = []
    for idx in inflection_indices:
        left_span = abs(freq[idx] - freq[0])
        right_span = abs(freq[-1] - freq[idx])

        # 更精确地检查：找到两侧最近的极值点
        left_max_span = max(abs(freq[:idx+1] - freq[idx]))
        right_max_span = max(abs(freq[idx:] - freq[idx]))

        if left_max_span >= min_span_hz and right_max_span >= min_span_hz:
            valid_inflections.append(idx)

    return len(valid_inflections), valid_inflections


def segment_by_trend(freq: np.ndarray, min_span_hz: float = 1000.0) -> List[dict]:
    """
    将轮廓按趋势（上升/下降/平直）分段
    返回每段的信息列表
    """
    if len(freq) < 3:
        return [{'trend': 'flat', 'start': 0, 'end': len(freq)-1, 'span': 0}]

    segments = []
    i = 0
    while i < len(freq) - 1:
        start = i
        diff = freq[i+1] - freq[i]

        if abs(diff) <= 2:  # 微小波动视为平直
            current_trend = 'flat'
        elif diff > 0:
            current_trend = 'rise'
        else:
            current_trend = 'down'

        # 扩展此段直到趋势改变
        j = i + 1
        while j < len(freq) - 1:
            d = freq[j+1] - freq[j]
            if current_trend == 'flat':
                if d > 2:
                    current_trend = 'rise'
                elif d < -2:
                    current_trend = 'down'
                else:
                    j += 1
            elif current_trend == 'rise':
                if d < -2:
                    break
                else:
                    j += 1
            else:  # down
                if d > 2:
                    break
                else:
                    j += 1

        span = abs(freq[j] - freq[start])
        segments.append({
            'trend': current_trend,
            'start': start,
            'end': j,
            'span': span
        })
        i = j

    # 合并相邻同趋势段
    merged = []
    for seg in segments:
        if merged and merged[-1]['trend'] == seg['trend']:
            merged[-1]['end'] = seg['end']
            merged[-1]['span'] = abs(freq[seg['end']] - freq[merged[-1]['start']])
        else:
            merged.append(seg)

    return merged


def classify_whistle(times: np.ndarray, freqs: np.ndarray,
                     min_span_hz: float = 1000.0) -> Tuple[str, dict]:
    """
    对单个哨声轮廓进行六类型分类

    Parameters:
        times: 时间序列 (秒)
        freqs: 频率序列 (Hz)
        min_span_hz: 判断升降的最小频率跨度 (Hz)，论文默认 1kHz

    Returns:
        (类型名称, 诊断信息字典)
    """
    if len(freqs) < 5:
        return 'Unknown', {'reason': '轮廓太短'}

    # 平滑
    freq_smooth = smooth_contour(freqs)

    # 总频率变化
    total_change = abs(freq_smooth[-1] - freq_smooth[0])

    # 检测拐点
    n_inflections, inf_idx = detect_inflection_points(freq_smooth, min_span_hz=500.0)

    # 分段分析
    segments = segment_by_trend(freq_smooth, min_span_hz=min_span_hz)

    # 统计有意义的升降段（跨度 > min_span_hz）
    significant_segments = [s for s in segments if s['span'] >= min_span_hz]
    n_rise = sum(1 for s in significant_segments if s['trend'] == 'rise')
    n_down = sum(1 for s in significant_segments if s['trend'] == 'down')
    n_flat = sum(1 for s in significant_segments if s['trend'] == 'flat')

    # 计算平直段占总时长的比例
    flat_duration = sum(
        times[s['end']] - times[s['start']]
        for s in segments if s['trend'] == 'flat'
    )
    total_duration = times[-1] - times[0]
    flat_ratio = flat_duration / total_duration if total_duration > 0 else 0

    # 诊断信息
    info = {
        'total_change_hz': total_change,
        'n_inflections': n_inflections,
        'n_rise': n_rise,
        'n_down': n_down,
        'n_flat': n_flat,
        'flat_ratio': flat_ratio,
        'significant_segments': [(s['trend'], s['span']) for s in significant_segments],
    }

    # ── 分类逻辑（严格按论文定义）──

    # Flat: 90%以上时长频率变化 < 1kHz
    if flat_ratio >= 0.9 or (total_change <= min_span_hz and n_inflections <= 1):
        return 'Flat', info

    # 无拐点：Up / Down
    if n_inflections == 0:
        if freq_smooth[-1] < freq_smooth[0] - min_span_hz:
            return 'Down', info
        elif freq_smooth[-1] > freq_smooth[0] + min_span_hz:
            return 'Rise', info
        else:
            return 'Flat', info

    # 仅看有效拐点（含小摆动但宏观趋势明确的）
    # 用一个更稳健的方法：对轮廓做更重的平滑后重新数拐点
    from scipy.signal import savgol_filter as sgf
    w = min(15, len(freq_smooth) - (1 - len(freq_smooth) % 2))
    if w >= 5:
        freq_heavily_smoothed = sgf(freq_smooth, w, 2)
        robust_noip, _ = detect_inflection_points(freq_heavily_smoothed, min_span_hz=500)
    else:
        robust_noip = n_inflections

    # 一个拐点
    if robust_noip == 1 or (n_inflections == 1 and robust_noip <= 1):
        first_trend = significant_segments[0]['trend'] if significant_segments else 'flat'
        if first_trend == 'rise':
            return 'Convex', info
        elif first_trend == 'down':
            return 'U-shaped', info
        else:
            # 按整体方向判断
            if freq_smooth[-1] < freq_smooth[0] - min_span_hz:
                return 'Down', info
            elif freq_smooth[-1] > freq_smooth[0] + min_span_hz:
                return 'Rise', info
            else:
                return 'Flat', info

    # 两个及以上拐点（经重度平滑仍为2+）= Sine
    if robust_noip >= 2:
        return 'Sine', info

    # 兜底：按整体方向
    if total_change > min_span_hz:
        if freq_smooth[-1] < freq_smooth[0]:
            return 'Down', info
        else:
            return 'Rise', info
    return 'Flat', info


def classify_all_whistles(contour_list: List[np.ndarray]) -> List[dict]:
    """
    批量分类所有哨声

    Parameters:
        contour_list: 每个元素为 (N, 2) 的 numpy array [time, frequency]

    Returns:
        每个哨声的分类结果列表
    """
    results = []
    for i, contour in enumerate(contour_list):
        if len(contour) < 5:
            results.append({
                'id': i,
                'type': 'Unknown',
                'info': {'reason': 'contour too short'},
                'contour': contour
            })
            continue

        times = contour[:, 0]
        freqs = contour[:, 1]
        whistle_type, info = classify_whistle(times, freqs)

        results.append({
            'id': i,
            'type': whistle_type,
            'info': info,
            'contour': contour
        })

    return results


# 类型统计
TYPE_NAMES = ['Flat', 'Down', 'Rise', 'Convex', 'U-shaped', 'Sine']
TYPE_NAMES_CN = {
    'Flat': '平直型',
    'Down': '下行型',
    'Rise': '上升型',
    'Convex': '凸型',
    'U-shaped': 'U型',
    'Sine': '正弦型',
    'Unknown': '未知',
}
