"""
可复现日志模块
============
每次分析自动记录：时间戳、参数、输入、版本、结果统计
输出: training_data/analysis_log.jsonl（每行一条JSON，可追加）
"""
import json, os, hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

LOG_PATH = Path(__file__).parent / 'training_data' / 'analysis_log.jsonl'
VERSION = '1.0.0'


def _get_file_hash(path: str, n_bytes: int = 4096) -> str:
    """快速文件哈希（读取头部）"""
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read(n_bytes)).hexdigest()
    except:
        return 'unknown'


def log_analysis(
    input_path: str,
    params: dict,
    results: list,
    duration_sec: float = 0,
) -> dict:
    """
    记录一次分析运行
    Args:
        input_path: WAV路径或文件夹路径
        params: {'threshold': -115, 'min_freq': 1000, 'max_freq': 50000}
        results: 分析结果列表
        duration_sec: 分析耗时
    Returns:
        日志条目 dict
    """
    # 文件统计
    files = set(r.get('file', '') for r in results)
    type_counts = {}
    for r in results:
        t = r.get('type', 'Unknown')
        type_counts[t] = type_counts.get(t, 0) + 1

    confidences = [r.get('confidence', 1.0) for r in results]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    n_low = sum(1 for c in confidences if c < 0.6)

    entry = {
        'timestamp': datetime.now().isoformat(),
        'version': VERSION,
        'input': str(input_path),
        'input_type': 'directory' if os.path.isdir(input_path) else 'file',
        'parameters': {
            'threshold_db': params.get('threshold', -115),
            'min_freq_hz': params.get('min_freq', 1000),
            'max_freq_hz': params.get('max_freq', 50000),
        },
        'results': {
            'n_files': len(files),
            'n_whistles': len(results),
            'type_distribution': type_counts,
            'avg_confidence': round(avg_conf, 3),
            'n_low_confidence': n_low,
        },
        'files_processed': sorted(files),
        'duration_sec': round(duration_sec, 1),
    }

    # 追加写入
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    return entry


def load_logs() -> list:
    """加载所有历史日志"""
    if not LOG_PATH.exists():
        return []
    logs = []
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return logs


def get_summary() -> dict:
    """获取累计统计"""
    logs = load_logs()
    if not logs:
        return {'total_runs': 0, 'total_whistles': 0, 'total_files': 0}

    all_types = {}
    total_whistles = 0
    files_set = set()

    for log in logs:
        r = log.get('results', {})
        total_whistles += r.get('n_whistles', 0)
        for f in log.get('files_processed', []):
            files_set.add(f)
        for t, c in r.get('type_distribution', {}).items():
            all_types[t] = all_types.get(t, 0) + c

    return {
        'total_runs': len(logs),
        'total_whistles': total_whistles,
        'total_files': len(files_set),
        'type_distribution': all_types,
        'first_run': logs[0]['timestamp'][:10],
        'last_run': logs[-1]['timestamp'][:10],
    }


def export_log_summary(output_path: str):
    """导出可读的日志摘要（用于论文方法部分）"""
    summary = get_summary()
    logs = load_logs()

    lines = [
        f"# 海豚哨声分析日志摘要",
        f"# 导出时间: {datetime.now().isoformat()}",
        f"# 软件版本: {VERSION}",
        "",
        f"## 总览",
        f"- 总运行次数: {summary['total_runs']}",
        f"- 总分析哨声数: {summary['total_whistles']}",
        f"- 总处理文件数: {summary['total_files']}",
        f"- 首次运行: {summary['first_run']}",
        f"- 最近运行: {summary['last_run']}",
        "",
        f"## 累计类型分布",
    ]
    for t in ['Flat', 'Down', 'Rise', 'Convex', 'U-shaped', 'Sine']:
        c = summary['type_distribution'].get(t, 0)
        pct = c / summary['total_whistles'] * 100 if summary['total_whistles'] else 0
        lines.append(f"- {t}: {c} ({pct:.1f}%)")

    lines.append("")
    lines.append("## 每次运行详情")
    lines.append("| # | 时间 | 文件数 | 哨声数 | 参数 | 平均置信度 |")
    lines.append("|---|---|---|---|---|---|")
    for i, log in enumerate(logs):
        r = log.get('results', {})
        p = log.get('parameters', {})
        lines.append(
            f"| {i+1} | {log['timestamp'][:19]} | {r.get('n_files',0)} | "
            f"{r.get('n_whistles',0)} | th={p.get('threshold_db','?')} "
            f"freq={p.get('min_freq_hz','?')}-{p.get('max_freq_hz','?')} | "
            f"{r.get('avg_confidence',0):.2f} |"
        )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return output_path
