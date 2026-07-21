"""
海豚哨声分析 Web 服务
Flask 后端 — 上传 WAV → 分析 → 返回结果
"""
import os
import sys
import json
import uuid
import shutil
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import numpy as np

# 项目根目录
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / 'uploads'
OUTPUT_DIR = BASE_DIR / 'outputs'
STATIC_DIR = BASE_DIR / 'static'

for d in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR]:
    d.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

from run_pipeline import (
    generate_spectrogram,
    detect_whistle_regions,
    extract_contour_from_region,
    split_into_individual_whistles,
    WhistleResult,
)
from whistle_classifier import classify_whistle, TYPE_NAMES, TYPE_NAMES_CN
from parameter_calculator import calculate_all_parameters
from spectrogram_renderer import render_spectrogram, render_annotated_spectrogram, RAVEN_CMAP
from visualizer import TYPE_COLORS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.io import wavfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

# ══════════════════════════════════════════════
#  核心分析函数
# ══════════════════════════════════════════════

def analyze_wav(wav_path: str, threshold_db: float = -115,
                min_freq: float = 1000, max_freq: float = 50000) -> dict:
    """分析 WAV 文件，返回完整结果 + 纯谱图路径"""
    wav_name = Path(wav_path).stem
    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_DIR / f'{wav_name}_{job_id}'
    job_dir.mkdir(exist_ok=True)

    # Step 1: 声谱图
    S_db, f, t, sr = generate_spectrogram(
        wav_path, n_fft=4096, overlap=0.75,
        min_freq=min_freq, max_freq=max_freq
    )

    # Step 2: 检测
    regions = detect_whistle_regions(
        S_db, f, t, threshold_db=threshold_db,
        min_freq_hz=min_freq, max_freq_hz=min(max_freq, 35000),
    )
    if not regions:
        regions = detect_whistle_regions(
            S_db, f, t, threshold_db=threshold_db - 10,
            min_freq_hz=min_freq, max_freq_hz=min(max_freq, 35000),
            min_area_pixels=100,
        )

    # Step 3: 提取轮廓
    all_contours = []
    for reg in regions:
        reg_start, reg_end, reg_low, reg_high = reg
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
                contour, gap_threshold_s=0.2, freq_jump_threshold_hz=3000,
            )
            all_contours.extend(whistles)

    # Step 4: 分类 + 参数
    results = []
    for i, contour in enumerate(all_contours):
        times_c = contour[:, 0]
        freqs_c = contour[:, 1]
        whistle_type, info = classify_whistle(times_c, freqs_c)
        params = calculate_all_parameters(
            times_c, freqs_c, audio_path=wav_path,
            whistle_start_s=float(times_c[0]), whistle_end_s=float(times_c[-1]),
            sample_rate=sr,
        )
        results.append({
            'id': i + 1,
            'type': whistle_type,
            'type_cn': TYPE_NAMES_CN.get(whistle_type, '未知'),
            'start_time': float(times_c[0]),
            'end_time': float(times_c[-1]),
            'min_freq_hz': float(np.min(freqs_c)),
            'max_freq_hz': float(np.max(freqs_c)),
            'duration_ms': float(times_c[-1] - times_c[0]) * 1000,
            'params': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in params.items()},
            'color': TYPE_COLORS.get(whistle_type, '#757575'),
            'contour': contour.tolist(),
        })

    # Step 5: 生成 Raven Pro 兼容 Selection Table
    sel_path = str(job_dir / f'{wav_name}.selections.txt')
    _generate_selection_table(results, sel_path)

    # JSON 结果
    json_path = str(job_dir / 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2, default=str)

    return {
        'job_id': job_id,
        'wav_name': wav_name,
        'results': results,
        'selection_url': f'/api/selection/{wav_name}_{job_id}',
        'sr': sr,
        'f_min': float(f[0]),
        'f_max': float(f[-1]),
        't_min': float(t[0]),
        't_max': float(t[-1]),
    }


def _generate_selection_table(results: list, output_path: str):
    """生成 Raven Pro 兼容的 Selection Table (.txt 制表符分隔)"""
    lines = [
        'Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tType\tDur_ms\tBF\tEF\tMinF\tMaxF\tDeltaF\tMeF\tNoIP\tNoG\tNoS\tNoH\tMFH'
    ]
    for r in results:
        p = r.get('params', {})
        line = '\t'.join([
            str(r['id']),                     # Selection
            'Spectrogram 1',                   # View
            '1',                               # Channel
            f"{r['start_time']:.6f}",          # Begin Time (s)
            f"{r['end_time']:.6f}",            # End Time (s)
            f"{r['min_freq_hz']:.1f}",         # Low Freq (Hz)
            f"{r['max_freq_hz']:.1f}",         # High Freq (Hz)
            r['type'],                         # Type
            f"{p.get('Dur', 0):.1f}",          # Dur_ms
            f"{p.get('BF', 0):.1f}",           # BF
            f"{p.get('EF', 0):.1f}",           # EF
            f"{p.get('MinF', 0):.1f}",         # MinF
            f"{p.get('MaxF', 0):.1f}",         # MaxF
            f"{p.get('DeltaF', 0):.1f}",       # DeltaF
            f"{p.get('MeF', 0):.1f}",          # MeF
            str(p.get('NoIP', 0)),             # NoIP
            str(p.get('NoG', 0)),              # NoG
            str(p.get('NoS', 0)),              # NoS
            str(p.get('NoH', 0)),              # NoH
            f"{p.get('MFH', 0):.1f}",          # MFH
        ])
        lines.append(line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _generate_clean_spectrogram(wav_path: str, output_path: str,
                                 min_freq: float = 1000, max_freq: float = 50000,
                                 n_fft: int = 4096, overlap: float = 0.75):
    """Raven Pro 风格声谱图"""
    render_spectrogram(
        wav_path, output_path,
        min_freq=min_freq, max_freq=max_freq,
        n_fft=n_fft, overlap=overlap,
        figsize=(30, 11), dpi=150,
    )


# ══════════════════════════════════════════════
#  API 路由
# ══════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """上传 WAV 并分析"""
    if 'file' not in request.files:
        return jsonify({'error': '未选择文件'}), 400

    file = request.files['file']
    if not file.filename or not file.filename.lower().endswith('.wav'):
        return jsonify({'error': '仅支持 .wav 文件'}), 400

    filename = secure_filename(file.filename)
    wav_path = UPLOAD_DIR / filename
    file.save(str(wav_path))

    threshold = float(request.form.get('threshold', -115))
    min_freq = float(request.form.get('min_freq', 1000))
    max_freq = float(request.form.get('max_freq', 50000))

    try:
        result = analyze_wav(str(wav_path), threshold_db=threshold,
                             min_freq=min_freq, max_freq=max_freq)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/static/spectrogram/<path:job_name>')
def serve_spectrogram(job_name):
    """提供纯声谱图"""
    # 从 job_name 找到对应的目录
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and d.name.startswith(job_name):
            img_path = d / 'spectrogram.png'
            if img_path.exists():
                return send_file(str(img_path), mimetype='image/png')
    return jsonify({'error': 'not found'}), 404


@app.route('/static/annotated/<path:job_name>')
def serve_annotated(job_name):
    """提供标注声谱图"""
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and d.name.startswith(job_name):
            img_path = d / 'annotated.png'
            if img_path.exists():
                return send_file(str(img_path), mimetype='image/png')
    return jsonify({'error': 'not found'}), 404


@app.route('/api/export/<path:job_name>')
def api_export(job_name):
    """导出 JSON 结果"""
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and d.name.startswith(job_name):
            json_path = d / 'results.json'
            if json_path.exists():
                return send_file(str(json_path), mimetype='application/json',
                                 as_attachment=True,
                                 download_name=f'{job_name}_whistles.json')
    return jsonify({'error': 'not found'}), 404


@app.route('/api/audio/<path:filename>')
def serve_audio(filename):
    """提供上传的音频文件给前端 spectrogram 组件"""
    wav_path = UPLOAD_DIR / secure_filename(filename)
    if wav_path.exists():
        return send_file(str(wav_path), mimetype='audio/wav')
    return jsonify({'error': 'not found'}), 404


@app.route('/api/export_csv/<path:job_name>')
def api_export_csv(job_name):
    """导出 CSV"""
    import io
    import csv
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and d.name.startswith(job_name):
            json_path = d / 'results.json'
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)

                si = io.StringIO()
                writer = csv.writer(si)
                param_keys = ['Dur', 'BF', 'Ft0.25', 'Ft0.5', 'Ft0.75', 'EF',
                              'MinF', 'MaxF', 'DeltaF', 'MeF',
                              'BS', 'ES', 'NoIP', 'NoG', 'NoS', 'NoH', 'MFH']
                writer.writerow(['ID', 'Type', 'Type_CN', 'Start_s', 'End_s',
                                 'Duration_ms', 'MinFreq_Hz', 'MaxFreq_Hz'] + param_keys)

                for r in results:
                    p = r.get('params', {})
                    row = [r['id'], r['type'], r['type_cn'],
                           r['start_time'], r['end_time'],
                           r['duration_ms'], r['min_freq_hz'], r['max_freq_hz']]
                    row += [p.get(k, '') for k in param_keys]
                    writer.writerow(row)

                output = si.getvalue()
                si.close()
                return output, 200, {
                    'Content-Type': 'text/csv; charset=utf-8-sig',
                    'Content-Disposition': f'attachment; filename={job_name}_whistles.csv'
                }
    return jsonify({'error': 'not found'}), 404


if __name__ == '__main__':
    print(f"\n  海豚哨声分析 Web 服务启动...")
    print(f"  打开浏览器访问: http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
