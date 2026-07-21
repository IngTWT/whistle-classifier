"""
哨声 17 参数计算模块
基于 Wang et al. (2013) Table II 定义
"""

import numpy as np
from scipy.signal import spectrogram, find_peaks
from typing import Tuple, Dict, Optional
import warnings


def calculate_all_parameters(
    times: np.ndarray,
    freqs: np.ndarray,
    audio_path: Optional[str] = None,
    whistle_start_s: Optional[float] = None,
    whistle_end_s: Optional[float] = None,
    sample_rate: int = 384000,
) -> Dict:
    """
    计算单个哨声的全部 17 个声学参数

    Parameters:
        times: 时间序列 (秒)
        freqs: 基频轮廓频率序列 (Hz)
        audio_path: 音频文件路径（用于计算谐波参数）
        whistle_start_s: 哨声在音频中的起始时间
        whistle_end_s: 哨声在音频中的结束时间
        sample_rate: 采样率（用于谐波检测）

    Returns:
        包含 17 个参数的字典
    """
    params = {}

    if len(freqs) < 5:
        return {'error': 'contour too short'}

    # ─── 平滑处理 ───
    from whistle_classifier import smooth_contour
    freq_s = smooth_contour(freqs, window=min(7, len(freqs) - (1 - len(freqs) % 2)))

    duration_ms = (times[-1] - times[0]) * 1000  # 转为 ms

    # ─── 频率参数 (9个) ───
    bf = freq_s[0]                               # Beginning Frequency (Hz)
    ef = freq_s[-1]                              # Ending Frequency (Hz)
    min_f = np.min(freq_s)                       # Minimum Frequency (Hz)
    max_f = np.max(freq_s)                       # Maximum Frequency (Hz)
    delta_f = max_f - min_f                      # Delta Frequency (Hz)
    mean_f = 0.25 * (bf + ef + min_f + max_f)    # Mean Frequency (Hz)

    # 四分位频率（插值）
    q25 = freq_at_ratio(freq_s, 0.25)
    q50 = freq_at_ratio(freq_s, 0.50)
    q75 = freq_at_ratio(freq_s, 0.75)

    params['BF'] = round(bf, 2)
    params['Ft0.25'] = round(q25, 2)
    params['Ft0.5'] = round(q50, 2)
    params['Ft0.75'] = round(q75, 2)
    params['EF'] = round(ef, 2)
    params['MinF'] = round(min_f, 2)
    params['MaxF'] = round(max_f, 2)
    params['DeltaF'] = round(delta_f, 2)
    params['MeF'] = round(mean_f, 2)

    # ─── 时间与定性参数 (6个) ───
    params['Dur'] = round(duration_ms, 2)  # Duration (ms)

    # 起始扫向 (BS): 前 10% 段的趋势
    bs_val = sweep_direction(freq_s, position='start', fraction=0.1, min_span_hz=1000)
    params['BS'] = bs_val  # -1=Down, 0=Flat, 1=Rise

    # 终止扫向 (ES): 后 10% 段的趋势
    es_val = sweep_direction(freq_s, position='end', fraction=0.1, min_span_hz=1000)
    params['ES'] = es_val

    # 拐点数量 (NoIP)
    from whistle_classifier import detect_inflection_points
    noip, _ = detect_inflection_points(freq_s, min_span_hz=500.0)
    params['NoIP'] = noip

    # 间隙数量 (NoG): 检测时间序列中的中断
    nog = count_gaps(times, freq_s, gap_threshold_ms=200)
    params['NoG'] = nog

    # 台阶数量 (NoS): 频率突变 >500Hz
    nos = count_stairs(freq_s, threshold_hz=500)
    params['NoS'] = nos

    # ─── 谐波参数 (2个) ───
    noh, mfh = 0, 0
    if audio_path and whistle_start_s is not None and whistle_end_s is not None:
        noh, mfh = detect_harmonics(
            audio_path, whistle_start_s, whistle_end_s,
            fundamental_contour=(times, freq_s),
            sample_rate=sample_rate
        )
    params['NoH'] = noh
    params['MFH'] = round(mfh, 2)

    return params


def freq_at_ratio(freqs: np.ndarray, ratio: float) -> float:
    """获取持续时间某比例处的频率（线性插值）"""
    idx = (len(freqs) - 1) * ratio
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    if lo == hi:
        return freqs[lo]
    frac = idx - lo
    return freqs[lo] * (1 - frac) + freqs[hi] * frac


def sweep_direction(freqs: np.ndarray, position: str = 'start',
                    fraction: float = 0.1, min_span_hz: float = 1000) -> int:
    """
    判断起始/终止扫向
    BS/ES: -1 = Down, 0 = Flat, 1 = Rise
    """
    n = len(freqs)
    seg_len = max(3, int(n * fraction))

    if position == 'start':
        seg = freqs[:seg_len]
    else:
        seg = freqs[-seg_len:]

    span = seg[-1] - seg[0]

    if abs(span) < min_span_hz:
        return 0   # Flat
    elif span > 0:
        return 1   # Rise
    else:
        return -1  # Down


def count_gaps(times: np.ndarray, freqs: np.ndarray,
               gap_threshold_ms: float = 200, freq_jump_hz: float = 3000) -> int:
    """
    检测哨声中的间隙 (NoG)
    间隙定义：时间间隔 > gap_threshold_ms 且频率跳跃 > freq_jump_hz
    """
    if len(times) < 2:
        return 0

    time_diffs = np.diff(times) * 1000  # 转为 ms
    freq_diffs = np.abs(np.diff(freqs))

    n_gaps = 0
    for dt, df in zip(time_diffs, freq_diffs):
        if dt > gap_threshold_ms and df > freq_jump_hz:
            n_gaps += 1

    return n_gaps


def count_stairs(freqs: np.ndarray, threshold_hz: float = 500) -> int:
    """
    检测台阶数量 (NoS)
    台阶定义：连续段中频率突变 > threshold_hz（近似垂直）
    """
    if len(freqs) < 3:
        return 0

    diffs = np.abs(np.diff(freqs))
    n_stairs = 0
    i = 0

    while i < len(diffs):
        if diffs[i] > threshold_hz:
            n_stairs += 1
            # 跳过紧邻的大跳跃（同一台阶）
            while i < len(diffs) and diffs[i] > threshold_hz:
                i += 1
        i += 1

    return n_stairs


def detect_harmonics(
    audio_path: str,
    start_s: float,
    end_s: float,
    fundamental_contour: Tuple[np.ndarray, np.ndarray],
    sample_rate: int = 384000,
    n_fft: int = 4096,
    max_harmonics: int = 20,
) -> Tuple[int, float]:
    """
    检测谐波数量和最高谐波频率 (NoH, MFH)

    原理：在声谱图中，沿基频轮廓的整数倍频率处检测能量峰值
    """
    try:
        from scipy.io import wavfile

        # 读取哨声区间的音频
        sr, y_full = wavfile.read(audio_path)
        offset_samples = int(start_s * sr)
        duration_samples = int((end_s - start_s) * sr)
        if duration_samples <= 0:
            return 0, 0
        y = y_full[offset_samples:offset_samples + duration_samples]
        if y.ndim > 1:
            y = y.mean(axis=1)
        if y.dtype != np.float32 and y.dtype != np.float64:
            y = y.astype(np.float32) / np.iinfo(y.dtype).max
        if len(y) < n_fft:
            return 0, 0

        # 生成声谱图
        f, t_spec, Sxx = spectrogram(
            y, fs=sr, nperseg=n_fft, noverlap=n_fft * 3 // 4,
            window='hann', mode='magnitude'
        )

        # 转为 dB
        S_db = 20 * np.log10(np.clip(Sxx, 1e-14, None))

        # 获取基频轮廓在声谱图时间网格上的值
        times_fund, freqs_fund = fundamental_contour
        times_fund_rel = times_fund - times_fund[0]  # 转为相对时间

        fundamental_on_grid = np.zeros(len(t_spec))
        for i, t_val in enumerate(t_spec):
            if t_val < times_fund_rel[0] or t_val > times_fund_rel[-1]:
                fundamental_on_grid[i] = np.nan
            else:
                fundamental_on_grid[i] = np.interp(t_val, times_fund_rel, freqs_fund)

        # 对每个时间帧，检查谐波频率处是否有显著能量
        harmonic_counts = []
        max_freqs = []

        noise_floor = np.percentile(S_db, 20)  # 噪声基线

        for i in range(len(t_spec)):
            f0 = fundamental_on_grid[i]
            if np.isnan(f0) or f0 < 100:
                continue

            n_detected = 0
            max_harmonic_freq = 0
            signal_threshold = noise_floor + 6  # 比噪声高 6dB

            for k in range(2, max_harmonics + 1):  # 从第2谐波开始
                harmonic_freq = f0 * k

                # 找最近的频率 bin
                if harmonic_freq > f[-1]:
                    break

                freq_idx = np.argmin(np.abs(f - harmonic_freq))
                energy = S_db[freq_idx, i]

                # 检查局部是否有峰值
                search_range = max(1, int(len(f) * 0.01))
                lo = max(0, freq_idx - search_range)
                hi = min(len(f) - 1, freq_idx + search_range + 1)
                local_max = np.max(S_db[lo:hi+1, i])
                local_max_idx = lo + np.argmax(S_db[lo:hi+1, i])

                if local_max > signal_threshold:
                    n_detected += 1
                    max_harmonic_freq = f[local_max_idx]

            harmonic_counts.append(n_detected)
            max_freqs.append(max_harmonic_freq)

        if harmonic_counts:
            noh = int(np.median(harmonic_counts))
            mfh = np.max(max_freqs) if max_freqs else 0
        else:
            noh, mfh = 0, 0

        return noh, mfh

    except Exception as e:
        warnings.warn(f"Harmonic detection failed: {e}")
        return 0, 0


def format_parameter_table(params: Dict, whistle_type: str = '') -> str:
    """将参数字典格式化为可读表格"""
    from whistle_classifier import TYPE_NAMES_CN

    type_cn = TYPE_NAMES_CN.get(whistle_type, whistle_type)

    lines = [
        f"类型: {whistle_type} ({type_cn})",
        "─" * 40,
        "【频率参数】",
        f"  BF     (起始频率):     {params.get('BF', 'N/A'):>10} Hz",
        f"  Ft0.25 (1/4处频率):    {params.get('Ft0.25', 'N/A'):>10} Hz",
        f"  Ft0.5  (1/2处频率):    {params.get('Ft0.5', 'N/A'):>10} Hz",
        f"  Ft0.75 (3/4处频率):    {params.get('Ft0.75', 'N/A'):>10} Hz",
        f"  EF     (终止频率):     {params.get('EF', 'N/A'):>10} Hz",
        f"  MinF   (最低频率):     {params.get('MinF', 'N/A'):>10} Hz",
        f"  MaxF   (最高频率):     {params.get('MaxF', 'N/A'):>10} Hz",
        f"  ΔF     (频率变化量):   {params.get('DeltaF', 'N/A'):>10} Hz",
        f"  MeF    (平均频率):     {params.get('MeF', 'N/A'):>10} Hz",
        "",
        "【时间与定性参数】",
        f"  Dur    (持续时间):     {params.get('Dur', 'N/A'):>10} ms",
        f"  BS     (起始扫向):     {_sweep_name(params.get('BS', 0)):>10}",
        f"  ES     (终止扫向):     {_sweep_name(params.get('ES', 0)):>10}",
        f"  NoIP   (拐点数量):     {params.get('NoIP', 'N/A'):>10}",
        f"  NoG    (间隙数量):     {params.get('NoG', 'N/A'):>10}",
        f"  NoS    (台阶数量):     {params.get('NoS', 'N/A'):>10}",
        "",
        "【谐波参数】",
        f"  NoH    (谐波数量):     {params.get('NoH', 'N/A'):>10}",
        f"  MFH    (谐波最高频率): {params.get('MFH', 'N/A'):>10} Hz",
    ]
    return '\n'.join(lines)


def _sweep_name(val: int) -> str:
    """扫向值转名称"""
    return {-1: 'Down', 0: 'Flat', 1: 'Rise'}.get(val, str(val))
