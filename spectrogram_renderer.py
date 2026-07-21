"""
Raven Pro 风格声谱图渲染器
高质量、高对比度、科学可视化风格
"""
import numpy as np
from scipy.signal import spectrogram
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import os

# ── Raven Pro 风格自定义 colormap ──
# 低能量 → 高能量: 深蓝 → 蓝 → 青 → 绿 → 黄 → 橙 → 红
RAVEN_COLORS = [
    (0.000, '#0a0a2e'),  # 0%: deep navy (background)
    (0.125, '#0d1b4a'),  # dark blue
    (0.250, '#0f2d78'),  # blue
    (0.375, '#1a5ca8'),  # medium blue
    (0.500, '#1e90c4'),  # light blue → transition
    (0.625, '#1ea87a'),  # teal-green
    (0.750, '#4caf50'),  # green
    (0.830, '#c0ca33'),  # yellow-green
    (0.900, '#ffc107'),  # amber
    (0.950, '#ff7043'),  # orange
    (0.980, '#f44336'),  # red
    (1.000, '#ffffff'),  # white for peaks
]

RAVEN_CMAP = LinearSegmentedColormap.from_list(
    'raven_pro', [c[1] for c in RAVEN_COLORS]
)


def render_spectrogram(
    wav_path: str,
    output_path: str,
    min_freq: float = 1000,
    max_freq: float = 50000,
    n_fft: int = 4096,
    overlap: float = 0.75,
    figsize: tuple = (30, 11),
    dpi: int = 150,
) -> dict:
    """
    渲染 Raven Pro 风格的高质量声谱图

    Returns:
        dict with rendering metadata: img_w, img_h, f_min, f_max, t_min, t_max
    """
    # ── 读取音频 ──
    sr, y = wavfile.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.dtype != np.float32 and y.dtype != np.float64:
        y = y.astype(np.float32) / np.iinfo(y.dtype).max

    # ── 计算声谱图 ──
    hop_length = int(n_fft * (1 - overlap))
    f, t_spec, Sxx = spectrogram(
        y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length,
        window='hann', mode='magnitude'
    )
    S_db = 20 * np.log10(np.clip(Sxx, 1e-14, None))

    # ── 频率裁剪 ──
    freq_mask = (f >= min_freq) & (f <= max_freq)
    f_disp = f[freq_mask]
    S_disp = S_db[freq_mask, :]

    # ── 自适应对比度 ──
    # 取 5-95 百分位作为动态范围（去掉离群噪点）
    vmin = np.percentile(S_disp, 5)
    vmax = np.percentile(S_disp, 97)
    # 确保至少有 30dB 的动态范围
    if vmax - vmin < 30:
        mid = (vmax + vmin) / 2
        vmin = mid - 20
        vmax = mid + 20
    # 确保 vmin > -150
    vmin = max(vmin, -150)

    # ── 创建图形 ──
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0a0a2e')

    # 声谱图
    im = ax.pcolormesh(
        t_spec, f_disp / 1000, S_disp,
        shading='auto', cmap=RAVEN_CMAP,
        vmin=vmin, vmax=vmax,
        rasterized=True,
    )

    # ── 样式（Raven Pro 风格） ──
    ax.set_facecolor('#0a0a2e')

    # 坐标轴颜色
    ax_color = '#c0d0e0'
    ax.tick_params(colors=ax_color, labelsize=9, width=0.8)
    ax.spines['bottom'].set_color('#2a3a5c')
    ax.spines['left'].set_color('#2a3a5c')
    ax.spines['top'].set_color('#1a2a4c')
    ax.spines['right'].set_color('#1a2a4c')
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)

    # 网格线（淡色，仅水平）
    ax.grid(True, axis='y', color='#1a3050', linewidth=0.4, alpha=0.6)
    ax.grid(True, axis='x', color='#1a3050', linewidth=0.3, alpha=0.4)

    # 标签
    ax.set_xlabel('Time (s)', fontsize=12, color=ax_color, fontweight='medium')
    ax.set_ylabel('Frequency (kHz)', fontsize=12, color=ax_color, fontweight='medium')

    # 标题
    fname = os.path.basename(wav_path)
    ax.set_title(fname, fontsize=13, color='#e0e8f0', fontweight='bold', pad=8)

    # 范围
    ax.set_xlim(t_spec[0], t_spec[-1])
    ax.set_ylim(min_freq / 1000, max_freq / 1000)

    # colorbar 侧边栏（Raven Pro 风格）
    cbar = fig.colorbar(im, ax=ax, pad=0.015, shrink=0.92)
    cbar.set_label('Power Density (dB FS/Hz)', color=ax_color, fontsize=10)
    cbar.ax.yaxis.set_tick_params(colors=ax_color, labelsize=8)
    cbar.outline.set_edgecolor('#2a3a5c')
    cbar.outline.set_linewidth(1)

    # 添加时间标尺和频率标尺的可读性
    ax.tick_params(axis='both', which='major', length=5, width=0.8)
    ax.tick_params(axis='both', which='minor', length=3, width=0.4)

    plt.tight_layout(pad=1.0)

    # 保存
    plt.savefig(
        output_path, dpi=dpi,
        bbox_inches='tight',
        facecolor=fig.get_facecolor(),
        edgecolor='none',
    )
    plt.close(fig)

    return {
        'img_w': int(figsize[0] * dpi),
        'img_h': int(figsize[1] * dpi),
        'f_min': float(f_disp[0]),
        'f_max': float(f_disp[-1]),
        't_min': float(t_spec[0]),
        't_max': float(t_spec[-1]),
        'vmin': float(vmin),
        'vmax': float(vmax),
    }


def render_annotated_spectrogram(
    wav_path: str,
    output_path: str,
    whistle_results: list,
    min_freq: float = 1000,
    max_freq: float = 50000,
    n_fft: int = 4096,
    overlap: float = 0.75,
    figsize: tuple = (30, 11),
    dpi: int = 150,
) -> dict:
    """渲染带框选标注的高质量声谱图"""
    from whistle_classifier import TYPE_NAMES_CN

    sr, y = wavfile.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.dtype != np.float32 and y.dtype != np.float64:
        y = y.astype(np.float32) / np.iinfo(y.dtype).max

    hop_length = int(n_fft * (1 - overlap))
    f, t_spec, Sxx = spectrogram(
        y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length,
        window='hann', mode='magnitude'
    )
    S_db = 20 * np.log10(np.clip(Sxx, 1e-14, None))

    freq_mask = (f >= min_freq) & (f <= max_freq)
    f_disp = f[freq_mask]
    S_disp = S_db[freq_mask, :]

    vmin = np.percentile(S_disp, 5)
    vmax = np.percentile(S_disp, 97)
    if vmax - vmin < 30:
        mid = (vmax + vmin) / 2
        vmin = mid - 20
        vmax = mid + 20
    vmin = max(vmin, -150)

    TYPE_COLORS = {
        'Flat': '#4fc3f7', 'Down': '#ff7043', 'Rise': '#66bb6a',
        'Convex': '#ce93d8', 'U-shaped': '#ffa726', 'Sine': '#ef5350',
        'Unknown': '#90a4ae',
    }

    fig, ax = plt.subplots(figsize=figsize, facecolor='#0a0a2e')

    im = ax.pcolormesh(
        t_spec, f_disp / 1000, S_disp,
        shading='auto', cmap=RAVEN_CMAP,
        vmin=vmin, vmax=vmax, rasterized=True,
    )
    ax.set_facecolor('#0a0a2e')

    # 绘制框选
    from matplotlib.patches import Rectangle
    for i, r in enumerate(whistle_results):
        contour = np.array(r['contour'])
        times_c = contour[:, 0]
        freqs_c = contour[:, 1]

        start_t = times_c[0]
        end_t = times_c[-1]
        min_f_w = np.min(freqs_c) / 1000
        max_f_w = np.max(freqs_c) / 1000

        color = TYPE_COLORS.get(r['type'], '#90a4ae')
        margin_t = max(0.015, (end_t - start_t) * 0.06)
        margin_f = max(0.3, (max_f_w - min_f_w) * 0.12)

        rect = Rectangle(
            (start_t - margin_t, min_f_w - margin_f),
            (end_t - start_t) + 2 * margin_t,
            (max_f_w - min_f_w) + 2 * margin_f,
            linewidth=2.2, edgecolor=color, facecolor=color, alpha=0.10,
            linestyle='-', zorder=10,
        )
        ax.add_patch(rect)

        # 轮廓线
        ax.plot(times_c, freqs_c / 1000, color=color, linewidth=2.0, alpha=0.85, zorder=11)

        # 标签
        label_y = max_f_w + margin_f + 0.8
        if label_y > max_freq / 1000 - 0.8:
            label_y = min_f_w - margin_f - 1.0
        ax.annotate(
            f"#{r['id']} {r['type']}",
            xy=(start_t, min(label_y, max_freq / 1000 - 0.5)),
            fontsize=11, fontweight='bold', color='#ffffff',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=color,
                      edgecolor='none', alpha=0.92),
            zorder=20,
        )

    # 样式
    ax_color = '#c0d0e0'
    ax.tick_params(colors=ax_color, labelsize=9, width=0.8)
    for spine in ax.spines.values():
        spine.set_color('#2a3a5c')
    ax.grid(True, axis='y', color='#1a3050', linewidth=0.4, alpha=0.6)
    ax.grid(True, axis='x', color='#1a3050', linewidth=0.3, alpha=0.4)
    ax.set_xlabel('Time (s)', fontsize=12, color=ax_color)
    ax.set_ylabel('Frequency (kHz)', fontsize=12, color=ax_color)

    # 标题 + 统计
    type_counts = {}
    for r in whistle_results:
        t = r['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    stats = '  |  '.join([f'{t}: {c}' for t, c in sorted(type_counts.items())])
    title = f"{os.path.basename(wav_path)}   —   Total: {len(whistle_results)} whistles   |   {stats}"
    ax.set_title(title, fontsize=13, color='#e0e8f0', fontweight='bold', pad=8)

    ax.set_xlim(t_spec[0], t_spec[-1])
    ax.set_ylim(min_freq / 1000, max_freq / 1000)

    cbar = fig.colorbar(im, ax=ax, pad=0.015, shrink=0.92)
    cbar.set_label('Power Density (dB FS/Hz)', color=ax_color, fontsize=10)
    cbar.ax.yaxis.set_tick_params(colors=ax_color, labelsize=8)
    cbar.outline.set_edgecolor('#2a3a5c')

    plt.tight_layout(pad=1.0)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

    return {'img_w': int(figsize[0] * dpi), 'img_h': int(figsize[1] * dpi)}


if __name__ == '__main__':
    import sys
    wav = sys.argv[1] if len(sys.argv) > 1 else None
    if wav:
        out = Path(wav).stem + '_raven_style.png'
        meta = render_spectrogram(wav, out)
        print(f"谱图已生成: {out}")
        print(f"动态范围: {meta['vmin']:.1f} — {meta['vmax']:.1f} dB")
