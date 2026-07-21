"""
主管线脚本：从 WAV 到标注声谱图的完整流程
============================================
输入: WAV 文件
输出: 标注声谱图 + 参数报告 + 类型统计表

自包含实现：
  1. 声谱图生成
  2. 哨声区域检测（能量阈值 + 形态学处理）
  3. 频率轮廓提取（峰值追踪）
  4. 哨声分割
  5. 六类型分类
  6. 17 参数计算
  7. 标注可视化
"""

import numpy as np
import pandas as pd
from scipy.signal import spectrogram, savgol_filter, find_peaks
from scipy.ndimage import label, binary_opening, binary_closing, gaussian_filter
from scipy.interpolate import interp1d
from scipy.io import wavfile
import os
import sys
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whistle_classifier import classify_whistle, TYPE_NAMES_CN, TYPE_NAMES
from parameter_calculator import calculate_all_parameters, format_parameter_table
from visualizer import (
    generate_annotated_spectrogram,
    generate_detailed_report,
    generate_type_summary_table,
)


@dataclass
class WhistleResult:
    """单个哨声的分析结果"""
    id: int
    start_time: float        # 绝对起始时间 (秒)
    end_time: float          # 绝对结束时间 (秒)
    whistle_type: str        # 六类型之一
    contour: np.ndarray      # (N, 2) [time, frequency]
    params: dict             # 17 个参数
    confidence: float        # 检测置信度


# ═══════════════════════════════════════════════════════════
#  第一步：声谱图生成
# ═══════════════════════════════════════════════════════════

def generate_spectrogram(
    wav_path: str,
    n_fft: int = 4096,
    hop_length: int = None,
    overlap: float = 0.75,
    min_freq: float = 500,
    max_freq: float = 96000,
    sample_rate: int = None,
    window: str = 'hann',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    生成声谱图

    Returns:
        S_db: (freq_bins, time_bins) dB 谱图
        f: 频率轴 (Hz)
        t: 时间轴 (秒)
        sr: 实际采样率
    """
    # 读取音频（使用 scipy，无需 librosa）
    sr_raw, y = wavfile.read(wav_path)
    if sample_rate and sample_rate != sr_raw:
        # 重采样
        from scipy.signal import resample
        ratio = sample_rate / sr_raw
        y = resample(y, int(len(y) * ratio))
        sr = sample_rate
    else:
        sr = sr_raw
    # 转为 mono
    if y.ndim > 1:
        y = y.mean(axis=1)
    # 归一化
    if y.dtype != np.float32 and y.dtype != np.float64:
        y = y.astype(np.float32) / np.iinfo(y.dtype).max

    if hop_length is None:
        hop_length = int(n_fft * (1 - overlap))

    f, t_spec, Sxx = spectrogram(
        y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length,
        window=window, mode='magnitude'
    )

    # 转为 dB
    S_db = 20 * np.log10(np.clip(Sxx, 1e-14, None))

    # 频率裁剪
    freq_mask = (f >= min_freq) & (f <= max_freq)
    f = f[freq_mask]
    S_db = S_db[freq_mask, :]

    return S_db, f, t_spec, sr


# ═══════════════════════════════════════════════════════════
#  第二步：哨声区域检测
# ═══════════════════════════════════════════════════════════

def detect_whistle_regions(
    S_db: np.ndarray,
    f: np.ndarray,
    t: np.ndarray,
    threshold_db: float = -85,
    min_duration_ms: float = 30,
    max_duration_ms: float = 3000,
    min_freq_hz: float = 1000,
    max_freq_hz: float = 35000,
    min_area_pixels: int = 200,
    dilation_size: int = 3,
    freq_smooth_sigma: float = 1.0,
) -> List[Tuple[float, float, float, float]]:
    """
    在声谱图中检测哨声区域

    算法：
    1. 能量阈值 binarization
    2. 形态学开闭操作去噪
    3. 连通域分析
    4. 按时间和频率范围过滤
    5. 合并重叠/相邻区域

    Returns:
        [(start_time, end_time, low_freq, high_freq), ...]
    """
    # 裁剪到关注频率范围
    freq_mask = (f >= min_freq_hz) & (f <= max_freq_hz)
    S_roi = S_db[freq_mask, :]
    f_roi = f[freq_mask]

    # 对谱图做轻度高斯平滑
    S_smooth = gaussian_filter(S_roi, sigma=(freq_smooth_sigma, 0.5))

    # 二值化
    binary = S_smooth > threshold_db

    # 形态学处理：去除孤立噪点，连接断裂区域
    from scipy.ndimage import binary_dilation, binary_erosion
    struct = np.ones((3, dilation_size))
    binary = binary_opening(binary, structure=struct)
    binary = binary_closing(binary, structure=struct)

    # 连通域标记
    labeled, n_features = label(binary)

    regions = []
    for i in range(1, n_features + 1):
        region_mask = labeled == i
        rows, cols = np.where(region_mask)

        if len(rows) < min_area_pixels:
            continue

        time_start = t[cols.min()]
        time_end = t[cols.max()]
        freq_low = f_roi[rows.min()]
        freq_high = f_roi[rows.max()]

        duration_ms = (time_end - time_start) * 1000

        # 持续时间过滤
        if duration_ms < min_duration_ms or duration_ms > max_duration_ms:
            continue

        regions.append((time_start, time_end, freq_low, freq_high))

    # 合并重叠/非常接近的区域
    regions.sort(key=lambda x: x[0])
    merged = []
    for reg in regions:
        if merged and reg[0] - merged[-1][1] < 0.1:  # 100ms 内合并
            prev = merged.pop()
            merged.append((
                min(prev[0], reg[0]),
                max(prev[1], reg[1]),
                min(prev[2], reg[2]),
                max(prev[3], reg[3]),
            ))
        else:
            merged.append(reg)

    return merged


# ═══════════════════════════════════════════════════════════
#  第三步：频率轮廓提取
# ═══════════════════════════════════════════════════════════

def extract_contour_from_region(
    S_db: np.ndarray,
    f: np.ndarray,
    t: np.ndarray,
    start_time: float,
    end_time: float,
    low_freq: float,
    high_freq: float,
    smoothing_window: int = 7,
    min_energy_db: float = -100,
) -> np.ndarray:
    """
    从检测到的区域中提取频率轮廓

    使用峰值追踪：在每个时间帧，在预期频率范围内找能量峰值

    Returns:
        (N, 2) array [time, frequency]
    """
    # 找到时间范围对应的列
    t_start_idx = np.argmin(np.abs(t - start_time))
    t_end_idx = np.argmin(np.abs(t - end_time))
    t_end_idx = min(t_end_idx + 1, len(t))

    # 频率范围对应的行
    f_start_idx = np.argmin(np.abs(f - low_freq))
    f_end_idx = np.argmin(np.abs(f - high_freq))
    f_end_idx = min(f_end_idx + 1, len(f))

    contour_times = []
    contour_freqs = []

    for col in range(t_start_idx, t_end_idx):
        col_data = S_db[f_start_idx:f_end_idx, col]

        # 找峰值
        peaks, properties = find_peaks(col_data, height=min_energy_db, distance=5)

        if len(peaks) == 0:
            continue

        # 选择最强的峰值
        best_peak_idx = peaks[np.argmax(properties['peak_heights'])]
        peak_freq = f[f_start_idx + best_peak_idx]
        peak_time = t[col]

        contour_times.append(peak_time)
        contour_freqs.append(peak_freq)

    if len(contour_times) < 3:
        return np.array([]).reshape(0, 2)

    contour_times = np.array(contour_times)
    contour_freqs = np.array(contour_freqs)

    # 异常值过滤（去除偏离中位数太远的点）
    if len(contour_freqs) > 10:
        median_f = np.median(contour_freqs)
        mad = np.median(np.abs(contour_freqs - median_f))
        threshold = 5 * mad
        valid = np.abs(contour_freqs - median_f) < threshold
        contour_times = contour_times[valid]
        contour_freqs = contour_freqs[valid]

    # 平滑
    if len(contour_freqs) >= smoothing_window:
        w = min(smoothing_window, len(contour_freqs) - (1 - len(contour_freqs) % 2))
        try:
            contour_freqs = savgol_filter(contour_freqs, w, 2)
        except Exception:
            pass

    return np.column_stack((contour_times, contour_freqs))


# ═══════════════════════════════════════════════════════════
#  第四步：哨声分割
# ═══════════════════════════════════════════════════════════

def split_into_individual_whistles(
    contour: np.ndarray,
    gap_threshold_s: float = 0.2,     # 论文：200ms
    freq_jump_threshold_hz: float = 3000,
    min_points: int = 10,
) -> List[np.ndarray]:
    """
    将连续的轮廓数据分割为单个哨声

    论文规则：
    - gap < 200ms 且频率差 < 3kHz 视为同一哨声
    """
    if len(contour) < min_points:
        return []

    times = contour[:, 0]
    freqs = contour[:, 1]

    # 找分割点
    split_indices = [0]

    for i in range(1, len(times)):
        time_gap = times[i] - times[i-1]
        freq_jump = abs(freqs[i] - freqs[i-1])

        # 论文条件：gap > 200ms 或 freq jump > 3kHz → 新哨声
        if time_gap > gap_threshold_s or freq_jump > freq_jump_threshold_hz:
            split_indices.append(i)

    split_indices.append(len(times))

    # 生成分割后的哨声
    whistles = []
    for j in range(len(split_indices) - 1):
        start = split_indices[j]
        end = split_indices[j + 1]
        segment = contour[start:end]

        if len(segment) >= min_points:
            whistles.append(segment)

    return whistles


# ═══════════════════════════════════════════════════════════
#  主管线
# ═══════════════════════════════════════════════════════════

def analyze_wav_file(
    wav_path: str,
    output_dir: str = './outputs',
    n_fft: int = 4096,
    overlap: float = 0.75,
    min_freq: float = 500,
    max_freq: float = 96000,
    threshold_db: float = -85,
    save_contours_csv: bool = True,
    save_json: bool = True,
    verbose: bool = True,
) -> List[WhistleResult]:
    """
    分析单个 WAV 文件：检测 → 提取 → 分类 → 参数计算

    Parameters:
        wav_path: WAV 文件路径
        output_dir: 输出目录
        n_fft: FFT 点数
        overlap: 窗口重叠率
        min_freq: 最小频率 (Hz)
        max_freq: 最大频率 (Hz)
        threshold_db: 检测阈值 (dB)
        save_contours_csv: 是否保存轮廓 CSV
        save_json: 是否保存 JSON 结果
        verbose: 是否打印进度

    Returns:
        WhistleResult 列表
    """
    wav_name = os.path.splitext(os.path.basename(wav_path))[0]
    file_output_dir = os.path.join(output_dir, wav_name)
    os.makedirs(file_output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  分析文件: {wav_name}.wav")
        print(f"{'='*60}")

    # ── Step 1: 生成声谱图 ──
    if verbose:
        print("  [1/5] 生成声谱图...")
    S_db, f, t, sr = generate_spectrogram(
        wav_path, n_fft=n_fft, overlap=overlap,
        min_freq=min_freq, max_freq=max_freq
    )
    if verbose:
        print(f"        谱图尺寸: {S_db.shape}, 频率范围: {f[0]:.0f}-{f[-1]:.0f} Hz")

    # ── Step 2: 检测哨声区域 ──
    if verbose:
        print("  [2/5] 检测哨声区域...")
    regions = detect_whistle_regions(
        S_db, f, t,
        threshold_db=threshold_db,
        min_duration_ms=30,
        max_duration_ms=3000,
        min_freq_hz=min_freq,
        max_freq_hz=min(max_freq, 35000),
    )
    if verbose:
        print(f"        检测到 {len(regions)} 个候选区域")

    if not regions:
        if verbose:
            print("  [!] 未检测到任何哨声区域，尝试降低阈值...")
        regions = detect_whistle_regions(
            S_db, f, t,
            threshold_db=threshold_db - 10,
            min_duration_ms=20,
            max_duration_ms=3500,
            min_freq_hz=min_freq,
            max_freq_hz=min(max_freq, 35000),
            min_area_pixels=100,
        )
        if verbose:
            print(f"        降低阈值后检测到 {len(regions)} 个候选区域")

    # ── Step 3: 提取轮廓 + 分割 ──
    if verbose:
        print("  [3/5] 提取频率轮廓并分割哨声...")
    all_contours = []
    for reg in regions:
        reg_start, reg_end, reg_low, reg_high = reg
        # 自适应能量阈值：取区域谱图能量的 80 分位数
        t_start_idx = np.argmin(np.abs(t - reg_start))
        t_end_idx = min(np.argmin(np.abs(t - reg_end)) + 1, len(t))
        f_start_idx = np.argmin(np.abs(f - reg_low))
        f_end_idx = min(np.argmin(np.abs(f - reg_high)) + 1, len(f))
        region_energy = S_db[f_start_idx:f_end_idx, t_start_idx:t_end_idx]
        adaptive_min_energy = np.percentile(region_energy, 80) if region_energy.size > 0 else -100

        contour = extract_contour_from_region(
            S_db, f, t, reg_start, reg_end, reg_low, reg_high,
            min_energy_db=adaptive_min_energy,
        )
        if len(contour) >= 3:
            whistles = split_into_individual_whistles(
                contour,
                gap_threshold_s=0.2,  # 论文 200ms
                freq_jump_threshold_hz=3000,
            )
            all_contours.extend(whistles)

    if verbose:
        print(f"        提取到 {len(all_contours)} 个哨声轮廓")

    if not all_contours:
        if verbose:
            print("  [!] 未提取到有效哨声轮廓")
        return []

    # ── Step 4: 分类 + 参数计算 ──
    if verbose:
        print("  [4/5] 分类哨声类型并计算 17 参数...")
    results = []

    for i, contour in enumerate(all_contours):
        times_c = contour[:, 0]
        freqs_c = contour[:, 1]

        # 单位转换：原始频率是 Hz
        whistle_type, info = classify_whistle(times_c, freqs_c)

        # 计算 17 参数
        params = calculate_all_parameters(
            times_c, freqs_c,
            audio_path=wav_path,
            whistle_start_s=times_c[0],
            whistle_end_s=times_c[-1],
            sample_rate=sr,
        )

        result = WhistleResult(
            id=i + 1,
            start_time=float(times_c[0]),
            end_time=float(times_c[-1]),
            whistle_type=whistle_type,
            contour=contour,
            params=params,
            confidence=1.0,
        )
        results.append(result)

        if verbose and len(results) <= 10:  # 只打印前 10 个
            cn = TYPE_NAMES_CN.get(whistle_type, '未知')
            dur = params.get('Dur', 'N/A')
            print(f"        #{i+1}: {whistle_type} ({cn})  Dur={dur:.1f}ms  "
                  f"NoIP={params.get('NoIP', '?')}")

    if verbose and len(results) > 10:
        print(f"        ... 共 {len(results)} 个哨声（省略中间输出）")

    # ── Step 5: 生成输出 ──
    if verbose:
        print("  [5/5] 生成输出文件...")

    # 5a. 保存每个哨声的轮廓 CSV
    if save_contours_csv:
        contours_dir = os.path.join(file_output_dir, 'contours')
        os.makedirs(contours_dir, exist_ok=True)
        for res in results:
            csv_path = os.path.join(contours_dir, f'whistle_{res.id:03d}.csv')
            pd.DataFrame({
                'time': res.contour[:, 0],
                'frequency': res.contour[:, 1],
            }).to_csv(csv_path, index=False)

    # 5b. 保存 JSON 结果
    if save_json:
        json_results = []
        for res in results:
            r = asdict(res)
            r['contour'] = res.contour.tolist()
            json_results.append(r)

        json_path = os.path.join(file_output_dir, f'{wav_name}_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)

    # 5c. 生成标注声谱图
    vis_results = [
        {
            'id': r.id,
            'type': r.whistle_type,
            'info': {},
            'contour': r.contour,
            'params': r.params,
            'start_time': r.start_time,
            'end_time': r.end_time,
        }
        for r in results
    ]

    img_path = os.path.join(file_output_dir, f'{wav_name}_annotated.png')
    generate_annotated_spectrogram(
        wav_path, vis_results, img_path,
        freq_range=(min_freq, min(max_freq, 50000)),
    )
    if verbose:
        print(f"        标注声谱图: {img_path}")

    # 5d. 生成详细报告
    report_path = os.path.join(file_output_dir, f'{wav_name}_report.txt')
    generate_detailed_report(vis_results, report_path, wav_name)
    if verbose:
        print(f"        详细报告: {report_path}")

    # 5e. 生成类型统计表（对标论文 Table II）
    table_path = os.path.join(file_output_dir, f'{wav_name}_table2.csv')
    generate_type_summary_table(vis_results, table_path)
    if verbose:
        print(f"        统计表: {table_path}")

    # ── 打印汇总 ──
    if verbose:
        print(f"\n{'─'*50}")
        type_counts = {}
        for r in results:
            t = r.whistle_type
            type_counts[t] = type_counts.get(t, 0) + 1

        print(f"  检测到 {len(results)} 个哨声:")
        for t in TYPE_NAMES:
            c = type_counts.get(t, 0)
            pct = c / len(results) * 100 if results else 0
            cn = TYPE_NAMES_CN.get(t, '?')
            print(f"    {t:10s} ({cn}): {c:3d}  ({pct:5.1f}%)")
        print(f"{'─'*50}")

    return results


def batch_analyze(
    wav_dir: str,
    output_dir: str = './outputs',
    **kwargs,
) -> dict:
    """批量分析目录中的全部 WAV 文件"""
    wav_files = sorted([
        f for f in os.listdir(wav_dir)
        if f.lower().endswith('.wav')
    ])

    if not wav_files:
        print(f"目录 {wav_dir} 中没有找到 WAV 文件")
        return {}

    print(f"找到 {len(wav_files)} 个 WAV 文件")
    all_results = {}

    for wav_file in wav_files:
        wav_path = os.path.join(wav_dir, wav_file)
        try:
            results = analyze_wav_file(wav_path, output_dir=output_dir, **kwargs)
            all_results[wav_file] = results
        except Exception as e:
            print(f"  [!!] 分析 {wav_file} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 生成汇总
    print(f"\n{'='*60}")
    print(f"  批量分析完成！共处理 {len(all_results)}/{len(wav_files)} 个文件")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}")

    return all_results


# ═══════════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='海豚哨声自动检测与分类工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析单个文件
  python run_pipeline.py -i D:/audio/recording.wav

  # 批量分析
  python run_pipeline.py -i D:/audio/ -o D:/results/

  # 调整检测灵敏度（阈值越高越严格）
  python run_pipeline.py -i D:/audio/file.wav --threshold -75
        """
    )
    parser.add_argument('-i', '--input', required=True,
                        help='输入 WAV 文件路径或目录')
    parser.add_argument('-o', '--output', default='./outputs',
                        help='输出目录 (默认: ./outputs)')
    parser.add_argument('--n_fft', type=int, default=4096,
                        help='FFT 点数 (默认: 4096)')
    parser.add_argument('--overlap', type=float, default=0.75,
                        help='窗口重叠率 (默认: 0.75)')
    parser.add_argument('--min-freq', type=float, default=500,
                        help='最小频率 Hz (默认: 500)')
    parser.add_argument('--max-freq', type=float, default=96000,
                        help='最大频率 Hz (默认: 96000)')
    parser.add_argument('--threshold', type=float, default=-85,
                        help='检测阈值 dB (默认: -85, 越大越严格)')
    parser.add_argument('--no-plots', action='store_true',
                        help='不生成标注图片')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='详细输出')

    args = parser.parse_args()

    if os.path.isfile(args.input):
        results = analyze_wav_file(
            args.input,
            output_dir=args.output,
            n_fft=args.n_fft,
            overlap=args.overlap,
            min_freq=args.min_freq,
            max_freq=args.max_freq,
            threshold_db=args.threshold,
            verbose=args.verbose,
        )
        print(f"\n完成！共检测 {len(results)} 个哨声")
    elif os.path.isdir(args.input):
        batch_analyze(
            args.input,
            output_dir=args.output,
            n_fft=args.n_fft,
            overlap=args.overlap,
            min_freq=args.min_freq,
            max_freq=args.max_freq,
            threshold_db=args.threshold,
            verbose=args.verbose,
        )
    else:
        print(f"错误: 输入路径不存在: {args.input}")
        sys.exit(1)
