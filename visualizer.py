"""
可视化标注模块
在声谱图上框选哨声并标注类型 + 参数
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.font_manager import FontProperties
from scipy.signal import spectrogram
from scipy.io import wavfile
import os
from typing import List, Dict, Optional, Tuple
import warnings

# 尝试使用中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# 六种类型颜色映射
TYPE_COLORS = {
    'Flat':     '#2196F3',  # 蓝
    'Down':     '#FF5722',  # 橙
    'Rise':     '#4CAF50',  # 绿
    'Convex':   '#9C27B0',  # 紫
    'U-shaped': '#FF9800',  # 琥珀
    'Sine':     '#E91E63',  # 粉
    'Unknown':  '#757575',  # 灰
}

TYPE_NAMES_CN = {
    'Flat': '平直型',
    'Down': '下行型',
    'Rise': '上升型',
    'Convex': '凸型',
    'U-shaped': 'U型',
    'Sine': '正弦型',
    'Unknown': '未知',
}


def generate_annotated_spectrogram(
    wav_path: str,
    results: List[dict],
    output_path: str,
    figsize: Tuple[int, int] = (20, 10),
    freq_range: Tuple[float, float] = (0, 50000),
    time_range: Optional[Tuple[float, float]] = None,
    n_fft: int = 4096,
    dpi: int = 150,
) -> str:
    """
    生成带标注的声谱图

    Parameters:
        wav_path: WAV 文件路径
        results: 分类和参数结果列表，每个元素:
            {
                'id': int,
                'type': str,
                'info': dict,
                'contour': np.ndarray (N,2),
                'params': dict,
                'start_time': float,  # 哨声在音频中的绝对起始
                'end_time': float,    # 哨声在音频中的绝对结束
            }
        output_path: 输出图片路径
        figsize: 图片尺寸
        freq_range: 显示频率范围 (Hz)
        time_range: 显示时间范围 (秒)，None 则显示全部
        n_fft: FFT 点数
        dpi: 分辨率
    """
    # ── 读取音频并生成声谱图 ──
    try:
        sr_raw, y = wavfile.read(wav_path)
        sr = sr_raw
    except Exception:
        sr = 384000
        sr_raw, y = wavfile.read(wav_path)
        sr = sr_raw
    # 转为 mono
    if y.ndim > 1:
        y = y.mean(axis=1)
    # 归一化
    if y.dtype != np.float32 and y.dtype != np.float64:
        y = y.astype(np.float32) / np.iinfo(y.dtype).max

    if time_range:
        t_start, t_end = time_range
        start_sample = int(t_start * sr)
        end_sample = int(t_end * sr)
        y = y[start_sample:end_sample]
        time_offset = t_start
    else:
        time_offset = 0

    # 生成声谱图
    f, t_spec, Sxx = spectrogram(
        y, fs=sr, nperseg=n_fft, noverlap=n_fft * 3 // 4,
        window='hann', mode='magnitude'
    )
    S_db = 20 * np.log10(np.clip(Sxx, 1e-14, None))

    # 添加时间偏移
    t_spec = t_spec + time_offset

    # ── 频率裁剪 ──
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_display = f[freq_mask]
    S_display = S_db[freq_mask, :]

    # ── 绘图 ──
    fig, (ax_spec, ax_legend) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={'height_ratios': [8, 1]}
    )

    # 声谱图
    im = ax_spec.pcolormesh(t_spec, f_display / 1000, S_display,
                            shading='auto', cmap='magma', vmin=-120, vmax=-20)
    ax_spec.set_ylabel('Frequency (kHz)', fontsize=12)
    ax_spec.set_xlabel('Time (s)', fontsize=12)

    if time_range:
        ax_spec.set_xlim(time_range)
    else:
        ax_spec.set_xlim(t_spec[0], t_spec[-1])

    ax_spec.set_ylim(freq_range[0] / 1000, freq_range[1] / 1000)

    # ── 绘制每个检测到的哨声 ──
    for i, res in enumerate(results):
        whistle_type = res['type']
        contour = res['contour']
        params = res.get('params', {})
        color = TYPE_COLORS.get(whistle_type, '#757575')

        times_c = contour[:, 0]
        freqs_c = contour[:, 1]

        # 框选矩形
        start_t = times_c[0]
        end_t = times_c[-1]
        min_f = np.min(freqs_c) / 1000
        max_f = np.max(freqs_c) / 1000

        # 添加边距
        margin_t = max(0.02, (end_t - start_t) * 0.08)
        margin_f = max(0.5, (max_f - min_f) * 0.15)

        rect = Rectangle(
            (start_t - margin_t, min_f - margin_f),
            (end_t - start_t) + 2 * margin_t,
            (max_f - min_f) + 2 * margin_f,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.08,
            linestyle='-'
        )
        ax_spec.add_patch(rect)

        # 类型标签
        label_text = f"#{i+1} {whistle_type}"
        label_y = min(max_f + margin_f + 1, freq_range[1] / 1000 - 0.5)
        ax_spec.annotate(
            label_text,
            xy=(start_t, label_y),
            fontsize=9, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, alpha=0.85),
            rotation=0,
        )

        # 覆盖轮廓线
        ax_spec.plot(times_c, freqs_c / 1000, color=color, linewidth=1.5, alpha=0.7)

    # ── 汇总统计 ──
    type_counts = {}
    for res in results:
        t = res['type']
        type_counts[t] = type_counts.get(t, 0) + 1

    stats_text = f"Total: {len(results)} whistles  |  "
    stats_text += "  ".join([f"{t}: {c}" for t, c in sorted(type_counts.items())])

    file_name = os.path.basename(wav_path)
    ax_spec.set_title(f'{file_name} — Whistle Detection & Classification\n{stats_text}',
                      fontsize=13, fontweight='bold')

    # ── 底部图例 ──
    ax_legend.axis('off')
    legend_items = []
    from matplotlib.patches import Patch
    for t in ['Flat', 'Down', 'Rise', 'Convex', 'U-shaped', 'Sine']:
        if t in type_counts:
            legend_items.append(
                Patch(color=TYPE_COLORS[t], alpha=0.7, label=f'{t} ({TYPE_NAMES_CN[t]})')
            )
    if legend_items:
        ax_legend.legend(
            handles=legend_items, loc='center', ncol=len(legend_items),
            fontsize=10, frameon=False
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return output_path


def generate_detailed_report(
    results: List[dict],
    output_path: str,
    wav_name: str = '',
) -> str:
    """
    生成详细的文本报告，包含所有哨声的完整参数表
    """
    from parameter_calculator import format_parameter_table

    lines = [
        "=" * 70,
        f"  海豚哨声自动分析报告",
        f"  文件: {wav_name}",
        f"  检测哨声数: {len(results)}",
        "=" * 70,
        "",
    ]

    # 类型统计
    type_counts = {}
    for res in results:
        t = res['type']
        type_counts[t] = type_counts.get(t, 0) + 1

    lines.append("【类型分布】")
    total = len(results)
    for t in ['Flat', 'Down', 'Rise', 'Convex', 'U-shaped', 'Sine', 'Unknown']:
        count = type_counts.get(t, 0)
        pct = count / total * 100 if total > 0 else 0
        cn = TYPE_NAMES_CN.get(t, '未知')
        bar = '█' * int(pct / 2)
        lines.append(f"  {t:10s} ({cn}): {count:4d}  {pct:5.1f}%  {bar}")
    lines.append("")

    # 每个哨声的详细参数
    for i, res in enumerate(results):
        lines.append("=" * 70)
        lines.append(f"  哨声 #{i+1}")
        lines.append("=" * 70)
        lines.append(f"  时间范围: {res['contour'][0,0]:.3f}s — {res['contour'][-1,0]:.3f}s")

        if 'params' in res:
            lines.append(format_parameter_table(res['params'], res['type']))
        else:
            lines.append(f"  类型: {res['type']}")
        lines.append("")

    # 汇总统计（与论文 Table II 格式对比）
    lines.append("=" * 70)
    lines.append("  汇总统计 (Mean ± SD)")
    lines.append("=" * 70)

    param_keys = ['Dur', 'BF', 'Ft0.25', 'Ft0.5', 'Ft0.75', 'EF',
                  'MinF', 'MaxF', 'DeltaF', 'MeF',
                  'BS', 'ES', 'NoIP', 'NoG', 'NoS', 'NoH', 'MFH']

    for key in param_keys:
        values = []
        for res in results:
            if 'params' in res and key in res['params']:
                v = res['params'][key]
                if isinstance(v, (int, float)):
                    values.append(v)
        if values:
            mean_v = np.mean(values)
            std_v = np.std(values)
            lines.append(f"  {key:8s}: {mean_v:10.2f} ± {std_v:8.2f}")

    report = '\n'.join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return output_path


def generate_type_summary_table(
    results: List[dict],
    output_path: str,
) -> str:
    """
    生成按类型分组的参数统计表 CSV（对标论文 Table II）
    """
    import csv

    param_keys = ['Dur', 'BF', 'Ft0.25', 'Ft0.5', 'Ft0.75', 'EF',
                  'MinF', 'MaxF', 'DeltaF', 'MeF',
                  'BS', 'ES', 'NoIP', 'NoG', 'NoS', 'NoH', 'MFH']

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        # 表头
        header = ['Tonal Style', 'N'] + param_keys
        writer.writerow(header)
        writer.writerow([''] * len(header))  # 空行

        # 按类型分组
        for whistle_type in ['Flat', 'Down', 'Rise', 'U-shaped', 'Convex', 'Sine']:
            group = [r for r in results if r['type'] == whistle_type]
            n = len(group)

            if n == 0:
                continue

            row = [whistle_type, n]
            for key in param_keys:
                values = []
                for r in group:
                    if 'params' in r and key in r['params']:
                        v = r['params'][key]
                        if isinstance(v, (int, float)):
                            values.append(v)
                if values:
                    mean_v = np.mean(values)
                    std_v = np.std(values)
                    row.append(f"{mean_v:.2f} ± {std_v:.2f}")
                else:
                    row.append('N/A')

            writer.writerow(row)

    return output_path
