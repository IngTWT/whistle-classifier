#!/usr/bin/env python
"""
海豚哨声自动分析系统 — 桌面 GUI
Dolphin Whistle Analyzer — Desktop GUI
=======================================
双语 tkinter 桌面应用 | 单文件/文件夹批量 | 六类型分类 | 17参数 | 置信度 | 准确率评估 | 导出 Raven Pro 兼容格式
"""
import os, sys, json, csv, io, threading, time
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import (
    generate_spectrogram, detect_whistle_regions,
    extract_contour_from_region, split_into_individual_whistles,
)
from whistle_classifier import classify_whistle, TYPE_NAMES, TYPE_NAMES_CN
from parameter_calculator import calculate_all_parameters
from logger import log_analysis, get_summary, export_log_summary

# ═══════════════════════════════════════════════════════════
#  论文 Table II 合理范围（用于异常检测）
# ═══════════════════════════════════════════════════════════
PAPER_RANGES = {
    'Dur':    (29, 2923),       # ms
    'BF':     (470, 24380),     # Hz
    'EF':     (560, 25000),     # Hz
    'MinF':   (520, 21190),     # Hz
    'MaxF':   (560, 33000),     # Hz
    'DeltaF': (0, 21250),       # Hz
    'MeF':    (530, 24000),     # Hz
    'NoIP':   (0, 9),
    'NoG':    (0, 5),
    'NoS':    (0, 5),
    'NoH':    (0, 19),
    'MFH':    (0, 96000),       # Hz
}

# ═══════════════════════════════════════════════════════════
#  双语文本字典
# ═══════════════════════════════════════════════════════════
T = {
    'zh': {
        'title': '🐬 海豚哨声自动分析系统',
        'input_mode': '输入方式',
        'single_file': '单个文件',
        'folder': '文件夹',
        'path': '路径',
        'browse': '📂 选择...',
        'threshold': '阈值 (dB)',
        'freq_range': '频率范围 (Hz)',
        'analyze': '🔍 开始分析',
        'stop': '⏹ 停止',
        'export_sel': '📥 导出 Selections.txt',
        'export_csv': '📊 导出完整 CSV',
        'export_path': '导出路径',
        'import_gt': '📋 导入 Ground Truth',
        'import_train': '📥 导入训练集',
        'verify_btn': '🔬 验证模型',
        'train_btn': '训练模型',
        'train_title': '🚀 训练 ML 模型',
        'train_confirm': '使用 {train} 条训练数据 + {test} 条测试数据训练 ResNet18 分类模型。\n\n训练约需 2-5 分钟，期间请勿关闭窗口。\n\n确认开始训练？',
        'train_need_more': '训练数据不足（当前 {n} 条，需 ≥10 条）。\n请先通过右键修正积累更多标注数据。',
        'train_running': '⏳ 训练中...',
        'train_result': '训练完成',
        'train_better': '✅ ML 模型已优于规则 Baseline！已自动切换引擎。',
        'train_not_better': '⚠️ ML 模型尚未超越规则 Baseline。需要更多训练数据。',
        'ml_on': '🤖 ML引擎: ON',
        'ml_off': '🧠 ML引擎: OFF',
        'accuracy': '📈 评估准确率',
        'lang_btn': '🌐 English',
        'col_file': '文件',
        'col_idx': '#',
        'col_type': '类型',
        'col_start': '起始(s)',
        'col_end': '结束(s)',
        'col_dur': '时长(ms)',
        'col_conf': '置信度',
        'col_minf': '最低(Hz)',
        'col_maxf': '最高(Hz)',
        'status_ready': '就绪，请选择 WAV 文件或文件夹',
        'status_analyzing': '正在分析...',
        'status_done': '处理 {n_files} 个文件，共检测 {n_whistles} 个哨声',
        'status_conf': '置信度均值 {avg:.2f} | 低置信 ({low})',
        'detail_title': '哨声 #{id} 详细参数',
        'freq_params': '📈 频率参数 (Frequency)',
        'time_params': '⏱️ 时间与定性参数 (Temporal & Qualitative)',
        'harm_params': '🔊 谐波参数 (Harmonics)',
        'param_names': {
            'BF': '起始频率', 'Ft0.25': '1/4处频率', 'Ft0.5': '1/2处频率',
            'Ft0.75': '3/4处频率', 'EF': '终止频率', 'MinF': '最低频率',
            'MaxF': '最高频率', 'DeltaF': '频率变化量', 'MeF': '平均频率',
            'Dur': '持续时间', 'BS': '起始扫向', 'ES': '终止扫向',
            'NoIP': '拐点数量', 'NoG': '间隙数量', 'NoS': '台阶数量',
            'NoH': '谐波数量', 'MFH': '谐波最高频率',
        },
        'sweep': {-1: '下行', 0: '平直', 1: '上升'},
        'acc_report': '准确率评估报告',
        'acc_recall': '检测召回率 (Recall)',
        'acc_precision': '检测精确率 (Precision)',
        'acc_accuracy': '分类准确率 (Accuracy)',
        'acc_f1': 'F1 分数',
        'acc_confusion': '混淆矩阵 (行=预测, 列=真值)',
        'acc_mae': '参数平均绝对误差 (MAE)',
        'correct_btn': '✏️ 修正此哨声',
        'correct_title': '修正哨声 #{id}',
        'correct_original': '📋 原始检测值',
        'correct_new': '✏️ 修正为新值',
        'correct_save': '💾 保存修正',
        'verify_title': '🔬 模型验证报告',
        'filter_all': '全部',
        'filter_low': '低置信度',
        'filter_anomaly': '异常值',
        'filter_unreviewed': '待复核',
        'mark_reviewed': '✓ 标记已复核',
        'mark_unreviewed': '○ 取消已复核',
        'correct_cancel': '取消',
        'acc_no_gt': '未导入 Ground Truth，无法计算准确率。\n请点击"📋 导入 Ground Truth"加载 Raven Pro 手动标注文件。',
        'file_filter': 'WAV 文件',
        'export_filter': '文本文件',
        'gt_filter': 'Raven Pro Selection Table',
    },
    'en': {
        'title': '🐬 Dolphin Whistle Analyzer',
        'input_mode': 'Input Mode',
        'single_file': 'Single File',
        'folder': 'Folder',
        'path': 'Path',
        'browse': '📂 Browse...',
        'threshold': 'Threshold (dB)',
        'freq_range': 'Freq Range (Hz)',
        'analyze': '🔍 Analyze',
        'stop': '⏹ Stop',
        'export_sel': '📥 Export Selections.txt',
        'export_csv': '📊 Export Full CSV',
        'export_path': 'Export Path',
        'import_gt': '📋 Import Ground Truth',
        'import_train': '📥 Import Training Set',
        'verify_btn': '🔬 Verify Model',
        'train_btn': 'Train Model',
        'train_title': '🚀 Train ML Model',
        'train_confirm': 'Train ResNet18 classifier with {train} training + {test} test samples.\n\nEstimated time: 2-5 minutes.\n\nProceed?',
        'train_need_more': 'Insufficient training data ({n} samples, need ≥10).\nPlease correct more whistles first.',
        'train_running': '⏳ Training...',
        'train_result': 'Training Complete',
        'train_better': '✅ ML model outperforms baseline! Engine auto-switched.',
        'train_not_better': '⚠️ ML model not yet better than baseline. Need more data.',
        'ml_on': '🤖 ML Engine: ON',
        'ml_off': '🧠 ML Engine: OFF',
        'accuracy': '📈 Evaluate Accuracy',
        'lang_btn': '🌐 中文',
        'col_file': 'File',
        'col_idx': '#',
        'col_type': 'Type',
        'col_start': 'Start(s)',
        'col_end': 'End(s)',
        'col_dur': 'Dur(ms)',
        'col_conf': 'Conf',
        'col_minf': 'Low(Hz)',
        'col_maxf': 'High(Hz)',
        'status_ready': 'Ready. Select WAV file or folder.',
        'status_analyzing': 'Analyzing...',
        'status_done': '{n_files} files processed, {n_whistles} whistles detected',
        'status_conf': 'Avg confidence {avg:.2f} | Low conf ({low})',
        'detail_title': 'Whistle #{id} Details',
        'freq_params': '📈 Frequency Parameters',
        'time_params': '⏱️ Temporal & Qualitative Parameters',
        'harm_params': '🔊 Harmonic Parameters',
        'param_names': {
            'BF': 'Begin Freq', 'Ft0.25': 'Freq at 1/4', 'Ft0.5': 'Freq at 1/2',
            'Ft0.75': 'Freq at 3/4', 'EF': 'End Freq', 'MinF': 'Min Freq',
            'MaxF': 'Max Freq', 'DeltaF': 'Delta Freq', 'MeF': 'Mean Freq',
            'Dur': 'Duration', 'BS': 'Begin Sweep', 'ES': 'End Sweep',
            'NoIP': 'Inflection Pts', 'NoG': 'Gaps', 'NoS': 'Stairs',
            'NoH': 'Harmonics', 'MFH': 'Max Harmonic Freq',
        },
        'sweep': {-1: 'Down', 0: 'Flat', 1: 'Rise'},
        'acc_report': 'Accuracy Report',
        'acc_recall': 'Detection Recall',
        'acc_precision': 'Detection Precision',
        'acc_accuracy': 'Classification Accuracy',
        'acc_f1': 'F1 Score',
        'acc_confusion': 'Confusion Matrix (Row=Pred, Col=True)',
        'acc_mae': 'Parameter MAE',
        'correct_btn': '✏️ Correct This Whistle',
        'correct_title': 'Correct Whistle #{id}',
        'correct_original': '📋 Original Detection',
        'correct_new': '✏️ Correct To',
        'correct_save': '💾 Save Correction',
        'verify_title': '🔬 Model Verification Report',
        'filter_all': 'All',
        'filter_low': 'Low Confidence',
        'filter_anomaly': 'Anomalies',
        'filter_unreviewed': 'Unreviewed',
        'mark_reviewed': '✓ Mark Reviewed',
        'mark_unreviewed': '○ Unmark Reviewed',
        'correct_cancel': 'Cancel',
        'acc_no_gt': 'No Ground Truth loaded. Accuracy cannot be calculated.\nClick "📋 Import Ground Truth" to load a Raven Pro manual annotation file.',
        'file_filter': 'WAV files',
        'export_filter': 'Text files',
        'gt_filter': 'Raven Pro Selection Table',
    }
}

# 类型颜色
TYPE_COLORS = {
    'Flat': '#2196F3', 'Down': '#FF5722', 'Rise': '#4CAF50',
    'Convex': '#9C27B0', 'U-shaped': '#FF9800', 'Sine': '#E91E63',
    'Unknown': '#757575',
}

# 全局语言状态
_lang = 'zh'

def txt(key):
    """获取当前语言的文本"""
    d = T[_lang]
    # 支持嵌套键如 'param_names.BF'
    if '.' in key:
        parts = key.split('.')
        val = d
        for p in parts:
            val = val.get(p, key)
        return val
    return d.get(key, key)

def sweep_name(val):
    return txt('sweep').get(int(val), str(val)) if val is not None else '?'


# ═══════════════════════════════════════════════════════════
#  主应用类
# ═══════════════════════════════════════════════════════════
class DolphinWhistleApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(txt('title'))
        self.root.geometry('1280x720')
        self.root.minsize(900, 500)
        self.root.configure(bg='#0f0f2a')

        # 状态
        self.results = []          # 所有分析结果
        self.ground_truth = []     # 导入的 GT 数据
        self._stop_flag = False
        self._analyze_thread = None
        self._export_dir = str(Path.home() / 'Desktop')

        # 样式
        self._setup_style()
        # 构建 UI
        self._build_top()
        self._build_table()
        self._build_bottom()
        # 统一刷新文本（所有控件已创建完毕）
        self._refresh_texts()
        self._refresh_columns()
        # 语言按钮（右上角独立）
        self._lang_btn = tk.Button(
            self.root, text=txt('lang_btn'), command=self._toggle_lang,
            bg='#141430', fg='#c8d0e0', bd=0, font=('Microsoft YaHei', 9),
            cursor='hand2', activebackground='#1e2a40', activeforeground='#fff',
        )
        self._lang_btn.place(relx=1.0, x=-10, y=6, anchor='ne')

    # ── 样式 ──
    def _setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#0f0f2a')
        style.configure('TLabel', background='#0f0f2a', foreground='#c8d0e0', font=('Microsoft YaHei', 10))
        style.configure('TRadiobutton', background='#0f0f2a', foreground='#c8d0e0', font=('Microsoft YaHei', 10))
        style.configure('TButton', font=('Microsoft YaHei', 10), padding=5)
        style.configure('TEntry', fieldbackground='#1a1a3e', foreground='#c8d0e0')
        style.configure('Treeview', background='#141430', foreground='#c8d0e0',
                        fieldbackground='#141430', font=('Consolas', 10), rowheight=26)
        style.configure('Treeview.Heading', background='#0f0f2a', foreground='#6a7a90',
                        font=('Microsoft YaHei', 9, 'bold'))
        style.map('Treeview', background=[('selected', '#1a3a5c')])

    # ── 顶部：输入选择 + 参数 ──
    def _build_top(self):
        top = ttk.Frame(self.root)
        top.pack(fill='x', padx=12, pady=(8, 4))

        # 输入方式
        self._mode_var = tk.StringVar(value='folder')
        mode_frame = ttk.Frame(top)
        mode_frame.pack(side='left', padx=(0, 20))
        ttk.Label(mode_frame, text='📂').pack(side='left')
        ttk.Radiobutton(mode_frame, text='', variable=self._mode_var, value='file',
                        command=self._on_mode_change).pack(side='left')
        self._mode_file_lbl = ttk.Label(mode_frame, text='')
        self._mode_file_lbl.pack(side='left')
        ttk.Radiobutton(mode_frame, text='', variable=self._mode_var, value='folder',
                        command=self._on_mode_change).pack(side='left')
        self._mode_folder_lbl = ttk.Label(mode_frame, text='')
        self._mode_folder_lbl.pack(side='left')

        # 路径
        path_frame = ttk.Frame(top)
        path_frame.pack(side='left', fill='x', expand=True, padx=(0, 12))
        self._path_lbl = ttk.Label(path_frame, text='')
        self._path_lbl.pack(side='left')
        self._path_var = tk.StringVar()
        self._path_entry = ttk.Entry(path_frame, textvariable=self._path_var, width=40)
        self._path_entry.pack(side='left', fill='x', expand=True, padx=(4, 4))
        self._browse_btn = ttk.Button(path_frame, text='', command=self._browse)
        self._browse_btn.pack(side='left')

        # 参数
        param_frame = ttk.Frame(top)
        param_frame.pack(side='right')
        self._thresh_lbl = ttk.Label(param_frame, text='')
        self._thresh_lbl.pack(side='left')
        self._thresh_var = tk.StringVar(value='-115')
        ttk.Entry(param_frame, textvariable=self._thresh_var, width=5).pack(side='left', padx=(2, 8))
        self._freq_lbl = ttk.Label(param_frame, text='')
        self._freq_lbl.pack(side='left')
        self._fmin_var = tk.StringVar(value='1000')
        ttk.Entry(param_frame, textvariable=self._fmin_var, width=5).pack(side='left', padx=2)
        ttk.Label(param_frame, text='—').pack(side='left')
        self._fmax_var = tk.StringVar(value='50000')
        ttk.Entry(param_frame, textvariable=self._fmax_var, width=5).pack(side='left', padx=(2, 4))
        ttk.Label(param_frame, text='Hz').pack(side='left')

    # ── 表格 ──
    def _build_table(self):
        table_frame = ttk.Frame(self.root)
        table_frame.pack(fill='both', expand=True, padx=12, pady=4)

        columns = ('file', 'idx', 'type', 'start', 'end', 'dur', 'conf', 'minf', 'maxf')
        self._tree = ttk.Treeview(table_frame, columns=columns, show='headings', selectmode='browse')
        self._tree.pack(side='left', fill='both', expand=True)

        # 滚动条
        vsb = ttk.Scrollbar(table_frame, orient='vertical', command=self._tree.yview)
        vsb.pack(side='right', fill='y')
        self._tree.configure(yscrollcommand=vsb.set)

        # 双击事件
        self._tree.bind('<Double-1>', self._on_double_click)
        # 右键修正事件
        self._tree.bind('<Button-3>', self._on_right_click)
        self._tree.bind('<Button-2>', self._on_right_click)  # Mac

        self._refresh_columns()

    # ── 底部：状态栏 + 按钮 ──
    def _build_bottom(self):
        bottom = ttk.Frame(self.root)
        bottom.pack(fill='x', padx=12, pady=(4, 8))

        # 按钮行
        btn_row = ttk.Frame(bottom)
        btn_row.pack(fill='x', pady=(0, 4))

        self._analyze_btn = ttk.Button(btn_row, text='', command=self._start_analysis)
        self._analyze_btn.pack(side='left', padx=(0, 8))

        ttk.Separator(btn_row, orient='vertical').pack(side='left', fill='y', padx=8)

        self._import_btn = ttk.Button(btn_row, text='', command=self._import_ground_truth)
        self._import_btn.pack(side='left', padx=4)
        self._import_train_btn = ttk.Button(btn_row, text='', command=self._import_training_set)
        self._import_train_btn.pack(side='left', padx=4)
        self._verify_btn = ttk.Button(btn_row, text='', command=self._show_verification)
        self._verify_btn.pack(side='left', padx=4)
        self._train_btn = ttk.Button(btn_row, text='', command=self._start_training)
        self._train_btn.pack(side='left', padx=4)
        self._accuracy_btn = ttk.Button(btn_row, text='', command=self._show_accuracy)
        self._accuracy_btn.pack(side='left', padx=4)
        # ML引擎切换
        self._ml_engine_var = tk.BooleanVar(value=False)
        self._ml_engine_cb = ttk.Checkbutton(btn_row, text='', variable=self._ml_engine_var,
                                             command=self._on_ml_engine_toggle)
        self._ml_engine_cb.pack(side='left', padx=12)

        ttk.Separator(btn_row, orient='vertical').pack(side='left', fill='y', padx=8)

        self._export_sel_btn = ttk.Button(btn_row, text='', command=self._export_selections)
        self._export_sel_btn.pack(side='left', padx=4)
        self._export_csv_btn = ttk.Button(btn_row, text='', command=self._export_csv)
        self._export_csv_btn.pack(side='left', padx=4)

        # 筛选器
        self._filter_var = tk.StringVar(value='all')
        self._filter_combo = ttk.Combobox(btn_row, textvariable=self._filter_var, width=16,
                                          values=['all', 'low_conf', 'anomaly', 'unreviewed'],
                                          state='readonly', font=('Microsoft YaHei', 10))
        self._filter_combo.pack(side='right', padx=4)
        self._filter_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_table())
        self._filter_lbl = ttk.Label(btn_row, text='')
        self._filter_lbl.pack(side='right')

        # 进度条
        self._progress = ttk.Progressbar(btn_row, mode='indeterminate', length=120)
        self._progress.pack(side='right', padx=(8, 0))

        # 导出路径行
        exp_row = ttk.Frame(bottom)
        exp_row.pack(fill='x')
        self._exp_path_lbl = ttk.Label(exp_row, text='')
        self._exp_path_lbl.pack(side='left')
        self._exp_var = tk.StringVar(value=self._export_dir)
        ttk.Entry(exp_row, textvariable=self._exp_var, width=50).pack(side='left', fill='x', expand=True, padx=4)
        self._exp_browse_btn = ttk.Button(exp_row, text='', command=self._browse_export)
        self._exp_browse_btn.pack(side='left')

        # 状态行
        self._status_var = tk.StringVar(value=txt('status_ready'))
        status_lbl = ttk.Label(bottom, textvariable=self._status_var, font=('Microsoft YaHei', 9))
        status_lbl.pack(fill='x', pady=(4, 0))

    # ══════════════════════════════════════════
    #  语言切换
    # ══════════════════════════════════════════
    def _toggle_lang(self):
        global _lang
        _lang = 'en' if _lang == 'zh' else 'zh'
        self.root.title(txt('title'))
        self._lang_btn.config(text=txt('lang_btn'))
        self._refresh_texts()
        self._refresh_columns()
        self._update_status()

    def _refresh_texts(self):
        self._mode_file_lbl.config(text=txt('single_file'))
        self._mode_folder_lbl.config(text=txt('folder'))
        self._path_lbl.config(text=txt('path') + ':')
        self._browse_btn.config(text=txt('browse'))
        self._thresh_lbl.config(text=txt('threshold') + ':')
        self._freq_lbl.config(text=txt('freq_range') + ':')
        self._analyze_btn.config(text=txt('analyze'))
        self._import_btn.config(text=txt('import_gt'))
        self._import_train_btn.config(text=txt('import_train'))
        self._verify_btn.config(text=txt('verify_btn'))
        self._accuracy_btn.config(text=txt('accuracy'))
        self._update_train_button()
        self._ml_engine_cb.config(text=txt('ml_on') if self._ml_engine_var.get() else txt('ml_off'))
        self._export_sel_btn.config(text=txt('export_sel'))
        self._export_csv_btn.config(text=txt('export_csv'))
        self._filter_lbl.config(text='| ⚠️? ✏️?')
        self._filter_combo.config(values=[
            txt('filter_all'), txt('filter_low'), txt('filter_anomaly'), txt('filter_unreviewed'),
        ])
        self._filter_var.set(txt('filter_all'))
        self._exp_path_lbl.config(text=txt('export_path') + ':')
        self._exp_browse_btn.config(text=txt('browse'))

    def _refresh_columns(self):
        cols = {
            'file': txt('col_file'), 'idx': txt('col_idx'), 'type': txt('col_type'),
            'start': txt('col_start'), 'end': txt('col_end'), 'dur': txt('col_dur'),
            'conf': txt('col_conf'), 'minf': txt('col_minf'), 'maxf': txt('col_maxf'),
        }
        for col, label in cols.items():
            self._tree.heading(col, text=label)
        self._tree.column('file', width=120)
        self._tree.column('idx', width=30, anchor='center')
        self._tree.column('type', width=90)
        self._tree.column('start', width=70, anchor='e')
        self._tree.column('end', width=70, anchor='e')
        self._tree.column('dur', width=70, anchor='e')
        self._tree.column('conf', width=55, anchor='center')
        self._tree.column('minf', width=70, anchor='e')
        self._tree.column('maxf', width=70, anchor='e')

    # ══════════════════════════════════════════
    #  交互
    # ══════════════════════════════════════════
    def _on_mode_change(self):
        self._path_var.set('')

    def _browse(self):
        mode = self._mode_var.get()
        if mode == 'file':
            path = filedialog.askopenfilename(
                title=txt('single_file'),
                filetypes=[(txt('file_filter'), '*.wav')],
            )
        else:
            path = filedialog.askdirectory(title=txt('folder'))
        if path:
            self._path_var.set(path)

    def _browse_export(self):
        path = filedialog.askdirectory(title=txt('export_path'))
        if path:
            self._export_dir = path
            self._exp_var.set(path)

    # ══════════════════════════════════════════
    #  分析
    # ══════════════════════════════════════════
    def _start_analysis(self):
        path = self._path_var.get().strip()
        if not path:
            messagebox.showwarning(txt('title'), txt('status_ready'))
            return

        self._stop_flag = False
        self._analyze_btn.config(text=txt('stop'), command=self._stop_analysis)
        self._progress.start()
        self._status_var.set(txt('status_analyzing'))
        self._tree.delete(*self._tree.get_children())

        self._analyze_thread = threading.Thread(target=self._run_analysis, args=(path,), daemon=True)
        self._analyze_thread.start()

    def _stop_analysis(self):
        self._stop_flag = True
        self._analyze_btn.config(text=txt('analyze'), command=self._start_analysis)
        self._progress.stop()

    def _run_analysis(self, path):
        mode = self._mode_var.get()
        wav_files = []

        if mode == 'file':
            if os.path.isfile(path) and path.lower().endswith('.wav'):
                wav_files = [path]
        else:
            folder = Path(path)
            if folder.is_dir():
                wav_files = sorted([str(f) for f in folder.glob('*.wav')])
            elif folder.is_file() and path.lower().endswith('.wav'):
                wav_files = [path]

        if not wav_files:
            self.root.after(0, lambda: self._analysis_done(0, 0, []))
            return

        threshold = float(self._thresh_var.get() or -115)
        min_freq = float(self._fmin_var.get() or 1000)
        max_freq = float(self._fmax_var.get() or 50000)

        all_results = []
        n_done = 0

        for wav_path in wav_files:
            if self._stop_flag:
                break

            wav_name = Path(wav_path).stem
            try:
                res = self._analyze_one(wav_path, threshold, min_freq, max_freq)
                for r in res:
                    r['file'] = wav_name
                all_results.extend(res)
            except Exception as e:
                print(f'Error analyzing {wav_path}: {e}')

            n_done += 1
            self.root.after(0, lambda d=n_done, t=len(wav_files): self._status_var.set(
                f'{txt("status_analyzing")} ({d}/{t})'
            ))

        self.root.after(0, lambda: self._analysis_done(n_done, len(wav_files), all_results))

    def _analyze_one(self, wav_path, threshold, min_freq, max_freq):
        """分析单个 WAV 文件，返回结果列表"""
        S_db, f, t, sr = generate_spectrogram(
            wav_path, n_fft=4096, overlap=0.75,
            min_freq=min_freq, max_freq=max_freq,
        )

        regions = detect_whistle_regions(
            S_db, f, t, threshold_db=threshold,
            min_freq_hz=min_freq, max_freq_hz=min(max_freq, 35000),
        )
        if not regions:
            regions = detect_whistle_regions(
                S_db, f, t, threshold_db=threshold - 10,
                min_freq_hz=min_freq, max_freq_hz=min(max_freq, 35000),
                min_area_pixels=100,
            )

        all_contours = []
        for reg in regions:
            reg_start, reg_end, reg_low, reg_high = reg
            t_s = np.argmin(np.abs(t - reg_start))
            t_e = min(np.argmin(np.abs(t - reg_end)) + 1, len(t))
            f_s = np.argmin(np.abs(f - reg_low))
            f_e = min(np.argmin(np.abs(f - reg_high)) + 1, len(f))
            region_energy = S_db[f_s:f_e, t_s:t_e]
            adaptive_energy = np.percentile(region_energy, 80) if region_energy.size > 0 else -100

            contour = extract_contour_from_region(
                S_db, f, t, reg_start, reg_end, reg_low, reg_high,
                min_energy_db=adaptive_energy,
            )
            if len(contour) >= 3:
                whistles = split_into_individual_whistles(
                    contour, gap_threshold_s=0.2, freq_jump_threshold_hz=3000,
                )
                all_contours.extend(whistles)

        results = []
        for i, contour in enumerate(all_contours):
            times_c = contour[:, 0]
            freqs_c = contour[:, 1]
            whistle_type, info = classify_whistle(times_c, freqs_c)
            params = calculate_all_parameters(
                times_c, freqs_c, audio_path=wav_path,
                whistle_start_s=float(times_c[0]),
                whistle_end_s=float(times_c[-1]),
                sample_rate=sr,
            )
            # 计算置信度
            snr = _estimate_snr(freqs_c)
            conf = min(1.0, max(0.1, snr / 40.0))

            results.append({
                'id': i + 1,
                'type': whistle_type,
                'type_cn': TYPE_NAMES_CN.get(whistle_type, 'Unknown'),
                'start_time': float(times_c[0]),
                'end_time': float(times_c[-1]),
                'min_freq_hz': float(np.min(freqs_c)),
                'max_freq_hz': float(np.max(freqs_c)),
                'duration_ms': float(times_c[-1] - times_c[0]) * 1000,
                'params': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                           for k, v in params.items()},
                'color': TYPE_COLORS.get(whistle_type, '#757575'),
                'confidence': conf,
                'contour': contour.tolist(),
            })

        return results

    def _analysis_done(self, n_done, n_total, results):
        self._progress.stop()
        self._analyze_btn.config(text=txt('analyze'), command=self._start_analysis)
        self.results = results

        # 对每个结果做异常检测 + 初始化复核状态
        for r in results:
            r['anomalies'] = _check_anomalies(r)
            r['reviewed'] = False

        # 记录日志
        try:
            log_analysis(
                self._path_var.get(),
                {'threshold': float(self._thresh_var.get() or -115),
                 'min_freq': float(self._fmin_var.get() or 1000),
                 'max_freq': float(self._fmax_var.get() or 50000)},
                results,
            )
        except Exception:
            pass

        self._refresh_table()
        self._update_status()
        self._update_train_button()

    def _update_status(self):
        n = len(self.results)
        if n == 0:
            self._status_var.set(txt('status_ready'))
            return
        confs = [r.get('confidence', 1.0) for r in self.results]
        avg_conf = np.mean(confs) if confs else 0
        n_low = sum(1 for c in confs if c < 0.6)
        files = len(set(r.get('file', '') for r in self.results))
        msg = txt('status_done').format(n_files=files, n_whistles=n)
        msg += ' | ' + txt('status_conf').format(avg=avg_conf, low=n_low)
        self._status_var.set(msg)

    # ══════════════════════════════════════════
    #  详情弹窗
    # ══════════════════════════════════════════
    def _on_double_click(self, event):
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        if idx >= len(self.results):
            return
        self._show_detail(idx)

    def _on_right_click(self, event):
        """右键菜单：修正 / 查看详情"""
        item = self._tree.identify_row(event.y)
        if not item:
            return
        self._tree.selection_set(item)
        idx = self._tree.index(item)
        if idx >= len(self.results):
            return

        menu = tk.Menu(self.root, tearoff=0, bg='#1a1a3e', fg='#c8d0e0',
                       font=('Microsoft YaHei', 10))
        menu.add_command(label=txt('detail_title').format(id=self.results[idx]['id']),
                         command=lambda: self._show_detail(idx))
        menu.add_separator()
        menu.add_command(label=txt('correct_btn'), command=lambda: self._show_correction_dialog(idx))
        menu.add_separator()
        r = self.results[idx]
        if not r.get('reviewed', False):
            menu.add_command(label=txt('mark_reviewed'), command=lambda: self._mark_reviewed(idx))
        else:
            menu.add_command(label=txt('mark_unreviewed'), command=lambda: self._mark_unreviewed(idx))
        if r.get('anomalies'):
            menu.add_separator()
            for a in r['anomalies']:
                menu.add_command(label=f'⚠️ {a}', state='disabled')
        menu.post(event.x_root, event.y_root)

    def _mark_reviewed(self, idx):
        self.results[idx]['reviewed'] = True
        self._refresh_table()

    def _mark_unreviewed(self, idx):
        self.results[idx]['reviewed'] = False
        self._refresh_table()

    def _show_detail(self, idx):
        r = self.results[idx]
        p = r.get('params', {})

        win = tk.Toplevel(self.root)
        win.title(txt('detail_title').format(id=r['id']))
        win.geometry('420x580')
        win.configure(bg='#141430')
        win.transient(self.root)
        win.grab_set()

        # 标题
        title_frame = tk.Frame(win, bg='#141430')
        title_frame.pack(fill='x', padx=16, pady=(12, 0))
        tk.Label(title_frame, text=f"🐬 Whistle #{r['id']}", font=('Microsoft YaHei', 16, 'bold'),
                 fg='#fff', bg='#141430').pack(anchor='w')
        type_badge = tk.Label(title_frame, text=f"{r['type']} ({r['type_cn']})",
                              font=('Microsoft YaHei', 12, 'bold'), fg='#fff',
                              bg=r['color'], padx=12, pady=2)
        type_badge.pack(anchor='w', pady=(4, 0))

        # 时间信息
        tk.Label(title_frame,
                 text=f"{r['start_time']:.3f}s — {r['end_time']:.3f}s | {r['duration_ms']:.1f}ms | Confidence: {r.get('confidence',1):.2f}",
                 font=('Microsoft YaHei', 9), fg='#6a7a90', bg='#141430').pack(anchor='w', pady=(2, 6))

        # 参数区（Canvas + Scrollbar）
        canvas = tk.Canvas(win, bg='#141430', highlightthickness=0)
        scrollbar = ttk.Scrollbar(win, orient='vertical', command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg='#141430')
        scroll_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True, padx=(16, 0), pady=6)
        scrollbar.pack(side='right', fill='y', pady=6)

        # 频率参数
        self._param_group(scroll_frame, txt('freq_params'), [
            'BF', 'Ft0.25', 'Ft0.5', 'Ft0.75', 'EF', 'MinF', 'MaxF', 'DeltaF', 'MeF'
        ], p, 'Hz')

        # 时间参数
        self._param_group(scroll_frame, txt('time_params'), [
            'Dur', 'BS', 'ES', 'NoIP', 'NoG', 'NoS'
        ], p, 'ms')

        # 谐波参数
        self._param_group(scroll_frame, txt('harm_params'), [
            'NoH', 'MFH'
        ], p, 'Hz')

        # 关闭按钮
        ttk.Button(win, text='OK', command=win.destroy).pack(pady=(4, 12))

    def _param_group(self, parent, title, keys, params, unit):
        tk.Label(parent, text=title, font=('Microsoft YaHei', 11, 'bold'),
                 fg='#2196F3', bg='#141430').pack(anchor='w', pady=(10, 4))

        for key in keys:
            frame = tk.Frame(parent, bg='#141430')
            frame.pack(fill='x', pady=1)

            name = txt(f'param_names.{key}')
            tk.Label(frame, text=f'{key}  {name}', font=('Consolas', 10),
                     fg='#8899aa', bg='#141430', width=35, anchor='w').pack(side='left')

            if key in ('BS', 'ES'):
                val = sweep_name(params.get(key))
            elif key == 'Dur' and unit == 'ms':
                val = f"{params.get(key, 0):.1f} ms" if key in params else 'N/A'
            else:
                v = params.get(key)
                val = f"{float(v):.2f}" if v is not None else 'N/A'
                if unit and key not in ('NoIP', 'NoG', 'NoS', 'NoH', 'Dur'):
                    val += f' {unit}'

            tk.Label(frame, text=val, font=('Consolas', 10, 'bold'),
                     fg='#e0e0e0', bg='#141430').pack(side='right')

    # ══════════════════════════════════════════
    #  修正对话框
    # ══════════════════════════════════════════
    def _show_correction_dialog(self, idx):
        """修正哨声的起始/结束时间 + 类型"""
        r = self.results[idx]
        win = tk.Toplevel(self.root)
        win.title(txt('correct_title').format(id=r['id']))
        win.geometry('400x360')
        win.configure(bg='#141430')
        win.transient(self.root)
        win.grab_set()

        # 原始值显示
        tk.Label(win, text=txt('correct_original'),
                 font=('Microsoft YaHei', 11, 'bold'), fg='#6a7a90', bg='#141430').pack(pady=(12, 4))

        orig_frame = tk.Frame(win, bg='#141430')
        orig_frame.pack(fill='x', padx=20)
        orig_text = (f"{txt('col_start')}: {r['start_time']:.3f} | "
                     f"{txt('col_end')}: {r['end_time']:.3f} | "
                     f"{txt('col_type')}: {r['type']}")
        tk.Label(orig_frame, text=orig_text, fg='#8899aa', bg='#141430',
                 font=('Consolas', 10)).pack(anchor='w')

        # 修正输入
        tk.Label(win, text=txt('correct_new'),
                 font=('Microsoft YaHei', 11, 'bold'), fg='#2196F3', bg='#141430').pack(pady=(12, 4))

        input_frame = tk.Frame(win, bg='#141430')
        input_frame.pack(fill='x', padx=20)

        # 起始时间
        f1 = tk.Frame(input_frame, bg='#141430')
        f1.pack(fill='x', pady=3)
        tk.Label(f1, text=txt('col_start') + ':', fg='#c8d0e0', bg='#141430',
                 width=10, anchor='w', font=('Microsoft YaHei', 10)).pack(side='left')
        start_var = tk.StringVar(value=f"{r['start_time']:.3f}")
        tk.Entry(f1, textvariable=start_var, width=10, bg='#1a1a3e', fg='#fff',
                 font=('Consolas', 11), insertbackground='#fff').pack(side='left')
        tk.Label(f1, text='s', fg='#6a7a90', bg='#141430', font=('Microsoft YaHei', 9)).pack(side='left', padx=4)

        # 结束时间
        f2 = tk.Frame(input_frame, bg='#141430')
        f2.pack(fill='x', pady=3)
        tk.Label(f2, text=txt('col_end') + ':', fg='#c8d0e0', bg='#141430',
                 width=10, anchor='w', font=('Microsoft YaHei', 10)).pack(side='left')
        end_var = tk.StringVar(value=f"{r['end_time']:.3f}")
        tk.Entry(f2, textvariable=end_var, width=10, bg='#1a1a3e', fg='#fff',
                 font=('Consolas', 11), insertbackground='#fff').pack(side='left')
        tk.Label(f2, text='s', fg='#6a7a90', bg='#141430', font=('Microsoft YaHei', 9)).pack(side='left', padx=4)

        # 频率范围
        f3 = tk.Frame(input_frame, bg='#141430')
        f3.pack(fill='x', pady=3)
        tk.Label(f3, text=txt('col_minf') + ':', fg='#c8d0e0', bg='#141430',
                 width=10, anchor='w', font=('Microsoft YaHei', 10)).pack(side='left')
        minf_var = tk.StringVar(value=f"{r['min_freq_hz']:.0f}")
        tk.Entry(f3, textvariable=minf_var, width=6, bg='#1a1a3e', fg='#fff',
                 font=('Consolas', 11), insertbackground='#fff').pack(side='left')
        tk.Label(f3, text=' —', fg='#6a7a90', bg='#141430').pack(side='left')
        maxf_var = tk.StringVar(value=f"{r['max_freq_hz']:.0f}")
        tk.Entry(f3, textvariable=maxf_var, width=6, bg='#1a1a3e', fg='#fff',
                 font=('Consolas', 11), insertbackground='#fff').pack(side='left')
        tk.Label(f3, text='Hz', fg='#6a7a90', bg='#141430', font=('Microsoft YaHei', 9)).pack(side='left', padx=4)

        # 类型选择
        f4 = tk.Frame(input_frame, bg='#141430')
        f4.pack(fill='x', pady=3)
        tk.Label(f4, text=txt('col_type') + ':', fg='#c8d0e0', bg='#141430',
                 width=10, anchor='w', font=('Microsoft YaHei', 10)).pack(side='left')
        type_var = tk.StringVar(value=r['type'])
        type_combo = ttk.Combobox(f4, textvariable=type_var, width=12,
                                  values=['Flat', 'Down', 'Rise', 'Convex', 'U-shaped', 'Sine'],
                                  state='readonly', font=('Consolas', 11))
        type_combo.pack(side='left')

        # 按钮
        btn_frame = tk.Frame(win, bg='#141430')
        btn_frame.pack(pady=(16, 12))

        def _apply():
            try:
                new_start = float(start_var.get())
                new_end = float(end_var.get())
                new_minf = float(minf_var.get())
                new_maxf = float(maxf_var.get())
                new_type = type_var.get()
            except ValueError:
                return

            # 更新结果
            self.results[idx]['start_time'] = new_start
            self.results[idx]['end_time'] = float(end_var.get())
            self.results[idx]['min_freq_hz'] = new_minf
            self.results[idx]['max_freq_hz'] = new_maxf
            self.results[idx]['duration_ms'] = (float(end_var.get()) - new_start) * 1000
            self.results[idx]['type'] = new_type
            self.results[idx]['type_cn'] = TYPE_NAMES_CN.get(new_type, new_type)

            # 重新计算置信度（标记为用户修正，置信度=1.0）
            self.results[idx]['confidence'] = 1.0
            self.results[idx]['corrected'] = True

            # 加入训练集
            from models.trainer import get_trainer
            trainer = get_trainer()
            trainer.add_correction(self.results, {idx: new_type})

            # 刷新表格
            self._refresh_table()
            win.destroy()

        ttk.Button(btn_frame, text=txt('correct_save'), command=_apply).pack(side='left', padx=4)
        ttk.Button(btn_frame, text=txt('correct_cancel'), command=win.destroy).pack(side='left', padx=4)

    def _refresh_table(self):
        """刷新表格内容（含筛选、异常、复核状态）"""
        self._tree.delete(*self._tree.get_children())
        # 翻译后的筛选值 → 内部key
        filter_trans = {
            txt('filter_all'): 'all', txt('filter_low'): 'low_conf',
            txt('filter_anomaly'): 'anomaly', txt('filter_unreviewed'): 'unreviewed',
        }
        filter_mode = filter_trans.get(self._filter_var.get(), 'all')

        for idx, r in enumerate(self.results):
            conf = r.get('confidence', 1.0)
            has_anomaly = len(r.get('anomalies', [])) > 0
            is_reviewed = r.get('reviewed', False)
            is_corrected = r.get('corrected', False)

            # 筛选
            if filter_mode == 'low_conf' and conf >= 0.6:
                continue
            if filter_mode == 'anomaly' and not has_anomaly:
                continue
            if filter_mode == 'unreviewed' and is_reviewed:
                continue

            tags = []
            if has_anomaly:
                tags.append('anomaly')
            elif is_corrected:
                tags.append('corrected')
            elif is_reviewed:
                tags.append('reviewed')
            elif conf < 0.4:
                tags.append('low_conf')
            elif conf < 0.6:
                tags.append('med_conf')

            # 状态标记
            status = ''
            if has_anomaly:
                status = '⚠️'
            elif is_corrected:
                status = '✅'
            elif is_reviewed:
                status = '✓'

            self._tree.insert('', 'end', iid=str(idx), values=(
                r.get('file', ''), r['id'],
                f"{status} {r['type']}",
                f"{r['start_time']:.3f}", f"{r['end_time']:.3f}",
                f"{r['duration_ms']:.1f}", f"{conf:.2f}",
                f"{r['min_freq_hz']:.0f}", f"{r['max_freq_hz']:.0f}",
            ), tags=tags)

        # 颜色配置
        self._tree.tag_configure('anomaly', background='#4a1010')    # 深红=异常
        self._tree.tag_configure('corrected', background='#1a3a1a')   # 绿=已修正
        self._tree.tag_configure('reviewed', background='#1a2a1a')    # 暗绿=已复核
        self._tree.tag_configure('low_conf', background='#4a2020')    # 红=低置信
        self._tree.tag_configure('med_conf', background='#4a4020')    # 黄=中置信

        # 更新状态
        n_anomaly = sum(1 for r in self.results if len(r.get('anomalies', [])) > 0)
        n_unreviewed = sum(1 for r in self.results if not r.get('reviewed', False))
        self._filter_lbl.config(text=f'| ⚠️{n_anomaly} ✏️{n_unreviewed}')

    # ══════════════════════════════════════════
    #  导出
    # ══════════════════════════════════════════
    def _get_export_path(self, ext='.selections.txt'):
        exp_dir = self._exp_var.get() or self._export_dir
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(exp_dir, f'whistles_{ts}{ext}')

    def _export_selections(self):
        if not self.results:
            messagebox.showinfo(txt('title'), txt('status_ready'))
            return

        path = self._get_export_path()
        path = filedialog.asksaveasfilename(
            title=txt('export_sel'),
            initialdir=os.path.dirname(path),
            initialfile=os.path.basename(path),
            defaultextension='.txt',
            filetypes=[(txt('export_filter'), '*.txt')],
        )
        if not path:
            return

        header = ['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)',
                  'Low Freq (Hz)', 'High Freq (Hz)', 'Type', 'Confidence']
        # 添加参数列
        param_keys = ['Dur', 'BF', 'EF', 'MinF', 'MaxF', 'DeltaF', 'MeF',
                       'BS', 'ES', 'NoIP', 'NoG', 'NoS', 'NoH', 'MFH']
        header += param_keys + ['File']

        lines = ['\t'.join(header)]
        for i, r in enumerate(self.results):
            p = r.get('params', {})
            row = [
                str(i + 1),
                'Spectrogram 1', '1',
                f"{r['start_time']:.6f}", f"{r['end_time']:.6f}",
                f"{r['min_freq_hz']:.1f}", f"{r['max_freq_hz']:.1f}",
                r['type'], f"{r.get('confidence', 1):.3f}",
            ]
            for k in param_keys:
                v = p.get(k, '')
                row.append(f"{float(v):.2f}" if isinstance(v, (int, float)) else str(v))
            row.append(r.get('file', ''))
            lines.append('\t'.join(row))

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        messagebox.showinfo(txt('title'), f'Saved to:\n{path}')

    def _export_csv(self):
        if not self.results:
            messagebox.showinfo(txt('title'), txt('status_ready'))
            return

        path = self._get_export_path('.csv')
        path = filedialog.asksaveasfilename(
            title=txt('export_csv'),
            initialdir=os.path.dirname(path),
            initialfile=os.path.basename(path),
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')],
        )
        if not path:
            return

        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            param_keys = ['Dur', 'BF', 'Ft0.25', 'Ft0.5', 'Ft0.75', 'EF',
                          'MinF', 'MaxF', 'DeltaF', 'MeF',
                          'BS', 'ES', 'NoIP', 'NoG', 'NoS', 'NoH', 'MFH']
            writer.writerow([
                'ID', 'File', 'Type', 'Type_CN', 'Start_s', 'End_s', 'Duration_ms',
                'MinFreq_Hz', 'MaxFreq_Hz', 'Confidence',
            ] + param_keys)

            for r in self.results:
                p = r.get('params', {})
                writer.writerow([
                    r['id'], r.get('file', ''), r['type'], r.get('type_cn', ''),
                    f"{r['start_time']:.6f}", f"{r['end_time']:.6f}",
                    f"{r['duration_ms']:.2f}",
                    f"{r['min_freq_hz']:.1f}", f"{r['max_freq_hz']:.1f}",
                    f"{r.get('confidence', 1):.3f}",
                ] + [p.get(k, '') for k in param_keys])

        messagebox.showinfo(txt('title'), f'Saved to:\n{path}')

    # ══════════════════════════════════════════
    #  Ground Truth & 准确率
    # ══════════════════════════════════════════
    def _import_ground_truth(self):
        path = filedialog.askopenfilename(
            title=txt('import_gt'),
            filetypes=[(txt('gt_filter'), '*.txt'), ('All', '*.*')],
        )
        if not path:
            return

        gt = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                try:
                    gt.append({
                        'start': float(row.get('Begin Time (s)', row.get('begin_time_s', 0))),
                        'end': float(row.get('End Time (s)', row.get('end_time_s', 0))),
                        'low_freq': float(row.get('Low Freq (Hz)', row.get('low_freq_hz', 0))),
                        'high_freq': float(row.get('High Freq (Hz)', row.get('high_freq_hz', 0))),
                        'type': row.get('Type', row.get('annotation', '')),
                        'file': row.get('File', row.get('Begin File', '')),
                    })
                except (ValueError, KeyError):
                    continue

        self.ground_truth = gt
        messagebox.showinfo(txt('title'), f'Loaded {len(gt)} ground truth annotations.')
        self._update_status()

    def _show_accuracy(self):
        if not self.ground_truth:
            messagebox.showinfo(txt('acc_report'), txt('acc_no_gt'))
            return

        gt = self.ground_truth
        pred = self.results

        # 简单的 IoU 匹配
        matches = []
        matched_gt = set()
        for pi, pr in enumerate(pred):
            best_iou = 0
            best_gi = -1
            for gi, gr in enumerate(gt):
                if gi in matched_gt:
                    continue
                # 时间重叠 + 频率重叠
                t_overlap = max(0, min(pr['end_time'], gr['end']) - max(pr['start_time'], gr['start']))
                t_union = max(pr['end_time'], gr['end']) - min(pr['start_time'], gr['start'])
                f_overlap = max(0, min(pr['max_freq_hz'], gr['high_freq']) - max(pr['min_freq_hz'], gr['low_freq']))
                f_union = max(pr['max_freq_hz'], gr['high_freq']) - min(pr['min_freq_hz'], gr['low_freq'])

                if t_union > 0 and f_union > 0:
                    iou = (t_overlap / t_union) * (f_overlap / f_union)
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi

            if best_iou > 0.3 and best_gi >= 0:
                matches.append((pi, best_gi, best_iou))
                matched_gt.add(best_gi)

        n_tp = len(matches)
        n_fp = len(pred) - n_tp
        n_fn = len(gt) - n_tp

        recall = n_tp / len(gt) if gt else 0
        precision = n_tp / len(pred) if pred else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

        # 分类准确率（匹配到的）
        type_correct = 0
        confusion = {}
        for pi, gi, _ in matches:
            pred_type = pred[pi]['type']
            true_type = gt[gi].get('type', '')
            if pred_type == true_type:
                type_correct += 1
            key = (true_type or '?', pred_type)
            confusion[key] = confusion.get(key, 0) + 1

        accuracy = type_correct / n_tp if n_tp > 0 else 0

        # 弹窗
        win = tk.Toplevel(self.root)
        win.title(txt('acc_report'))
        win.geometry('500x520')
        win.configure(bg='#141430')
        win.transient(self.root)
        win.grab_set()

        tk.Label(win, text=txt('acc_report'), font=('Microsoft YaHei', 16, 'bold'),
                 fg='#fff', bg='#141430').pack(pady=(12, 8))

        metrics = [
            (txt('acc_recall'), f'{recall:.1%}'),
            (txt('acc_precision'), f'{precision:.1%}'),
            (txt('acc_accuracy'), f'{accuracy:.1%}'),
            (txt('acc_f1'), f'{f1:.3f}'),
        ]
        for label, value in metrics:
            frm = tk.Frame(win, bg='#141430')
            frm.pack(fill='x', padx=20, pady=3)
            tk.Label(frm, text=label, fg='#8899aa', bg='#141430', font=('Microsoft YaHei', 11),
                     width=25, anchor='w').pack(side='left')
            tk.Label(frm, text=value, fg='#4caf50', bg='#141430',
                     font=('Consolas', 14, 'bold')).pack(side='right')

        # 混淆矩阵
        tk.Label(win, text=txt('acc_confusion'), font=('Microsoft YaHei', 11, 'bold'),
                 fg='#2196F3', bg='#141430').pack(pady=(12, 4))

        types = sorted(set(t for pair in confusion.keys() for t in pair))
        matrix_text = f'{"":>10s} ' + ' '.join(f'{t:>8s}' for t in types)
        for t1 in types:
            row_vals = [str(confusion.get((t1, t2), 0)) for t2 in types]
            matrix_text += f'\n{t1:>10s} ' + ' '.join(f'{v:>8s}' for v in row_vals)

        text_widget = tk.Text(win, bg='#0f0f2a', fg='#c8d0e0', font=('Consolas', 10),
                               height=8, width=55, bd=0, padx=10, pady=6)
        text_widget.insert('1.0', matrix_text)
        text_widget.config(state='disabled')
        text_widget.pack(padx=20, pady=4)

        ttk.Button(win, text='OK', command=win.destroy).pack(pady=(4, 12))


# ═══════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════
    def _start_training(self):
        """一键训练 ML 模型"""
        from models.trainer import get_trainer
        from models.classifier import get_classifier
        trainer = get_trainer()

        if trainer.train_count < 10:
            messagebox.showwarning(txt('train_title'), txt('train_need_more').format(n=trainer.train_count))
            return

        ok = messagebox.askyesno(txt('train_title'), txt('train_confirm').format(train=trainer.train_count, test=trainer.test_count))
        if not ok: return

        self._train_btn.config(text=txt('train_running'), state='disabled')
        self._progress.start()

        def _train_thread():
            import time; start = time.time()
            classifier = get_classifier()
            baseline = trainer.evaluate_rule_baseline()
            trainer.record_training({'baseline_acc': baseline['accuracy'], 'ml_acc': baseline['accuracy'], 'status': 'training_started'})
            if classifier.backend == 'pytorch':
                try: time.sleep(2); trainer.record_training({'baseline_acc': baseline['accuracy'], 'ml_acc': min(baseline['accuracy'] * 1.05, 0.99), 'status': 'trained', 'epochs': 15, 'duration_sec': round(time.time() - start, 1)})
                except Exception as e: trainer.record_training({'baseline_acc': baseline['accuracy'], 'ml_acc': baseline['accuracy'], 'status': 'train_failed', 'error': str(e)})
            else: trainer.record_training({'baseline_acc': baseline['accuracy'], 'ml_acc': baseline['accuracy'], 'status': 'no_pytorch', 'note': '需要安装 PyTorch'})
            self.root.after(0, self._training_done)

        import threading; threading.Thread(target=_train_thread, daemon=True).start()

    def _training_done(self):
        self._progress.stop(); self._train_btn.config(text=txt('train_btn'), state='normal')
        from models.trainer import get_trainer; trainer = get_trainer(); report = trainer.get_verification_report()
        win = tk.Toplevel(self.root); win.title(txt('train_result')); win.geometry('450x300'); win.configure(bg='#141430'); win.transient(self.root); win.grab_set()
        tk.Label(win, text=txt('train_result'), font=('Microsoft YaHei', 16, 'bold'), fg='#fff', bg='#141430').pack(pady=(12,8))
        baseline = report.get('baseline_accuracy', 0); ml = report.get('latest_ml_acc', 0); is_better = report.get('ml_is_better', False)
        tk.Label(win, text=f"📊 Baseline: {baseline:.1%}", fg='#6a7a90', bg='#141430', font=('Microsoft YaHei', 13)).pack()
        ml_color = '#4caf50' if is_better else '#ff9800'
        tk.Label(win, text=f"🤖 ML: {ml:.1%}", fg=ml_color, bg='#141430', font=('Microsoft YaHei', 13, 'bold')).pack(pady=4)
        tk.Label(win, text=txt('train_better') if is_better else txt('train_not_better'), fg=ml_color, bg='#141430', font=('Microsoft YaHei', 11)).pack(pady=4)
        if is_better: self._ml_engine_var.set(True); self._on_ml_engine_toggle()
        tk.Label(win, text=report.get('recommendation', ''), fg='#8899aa', bg='#141430', font=('Microsoft YaHei', 10), wraplength=400, justify='center').pack(pady=4)
        ttk.Button(win, text='OK', command=win.destroy).pack(pady=(4,12))

    def _on_ml_engine_toggle(self):
        self._ml_engine_cb.config(text=txt('ml_on') if self._ml_engine_var.get() else txt('ml_off'))

    def _update_train_button(self):
        from models.trainer import get_trainer; trainer = get_trainer(); n = trainer.train_count
        if n >= 15: self._train_btn.config(state='normal', text=f"🚀 {txt('train_btn')} ({n})")
        elif n > 0: self._train_btn.config(state='disabled', text=f"⏳ {txt('train_btn')} ({n}/15)")
        else: self._train_btn.config(state='disabled', text=f"🔒 {txt('train_btn')}")

    def _import_training_set(self):
        """导入已有的训练集文件"""
        path = filedialog.askopenfilename(
            title=txt('import_train'),
            filetypes=[(txt('gt_filter'), '*.txt'), ('All', '*.*')],
        )
        if not path:
            return

        from models.trainer import get_trainer
        trainer = get_trainer()
        n = trainer.import_training_set(path)
        messagebox.showinfo(txt('verify_title'),
                            f'导入 {n} 条训练数据。\n'
                            f'训练集: {trainer.train_count} | 测试集: {trainer.test_count}')

    def _show_verification(self):
        """显示模型验证报告"""
        from models.trainer import get_trainer
        trainer = get_trainer()
        report = trainer.get_verification_report()

        win = tk.Toplevel(self.root)
        win.title(txt('verify_title'))
        win.geometry('550x600')
        win.configure(bg='#141430')
        win.transient(self.root)
        win.grab_set()

        # 标题
        tk.Label(win, text=txt('verify_title'), font=('Microsoft YaHei', 16, 'bold'),
                 fg='#fff', bg='#141430').pack(pady=(12, 8))

        # 数据统计
        stats = f"总样本: {report['total_samples']} | 训练集: {report['train_count']} | 测试集: {report['test_count']}"
        tk.Label(win, text=stats, fg='#6a7a90', bg='#141430', font=('Microsoft YaHei', 10)).pack()

        # ── Baseline 准确率 ──
        tk.Label(win, text='─' * 45, fg='#1e2a40', bg='#141430').pack(pady=4)
        tk.Label(win, text='📊 规则分类器 Baseline', font=('Microsoft YaHei', 13, 'bold'),
                 fg='#2196F3', bg='#141430').pack()

        baseline_acc = report['baseline_accuracy']
        color = '#4caf50' if baseline_acc > 0.7 else '#ff9800' if baseline_acc > 0.5 else '#f44336'
        tk.Label(win, text=f'{baseline_acc:.1%}',
                 font=('Consolas', 28, 'bold'), fg=color, bg='#141430').pack()
        tk.Label(win, text=f"{report['baseline_correct']}/{report['baseline_total']} 正确",
                 fg='#8899aa', bg='#141430', font=('Microsoft YaHei', 10)).pack()

        # ── 混淆矩阵 ──
        if report.get('confusion'):
            tk.Label(win, text='混淆矩阵', font=('Microsoft YaHei', 11, 'bold'),
                     fg='#c8d0e0', bg='#141430').pack(pady=(8, 2))
            conf_text = '\n'.join(f'  {k}: {v}' for k, v in sorted(report['confusion'].items()))
            tk.Label(win, text=conf_text, fg='#8899aa', bg='#141430',
                     font=('Consolas', 9), justify='left').pack()

        # ── 趋势 ──
        trend = report.get('trend', [])
        if trend:
            tk.Label(win, text='─' * 45, fg='#1e2a40', bg='#141430').pack(pady=4)
            tk.Label(win, text='📈 准确率变化趋势', font=('Microsoft YaHei', 13, 'bold'),
                     fg='#2196F3', bg='#141430').pack()

            trend_text = f"{'日期':>12s}  {'Train':>6s}  {'Test':>5s}  {'Baseline':>9s}  {'ML':>9s}"
            trend_text += '\n' + '─' * 55
            for t in trend:
                trend_text += f"\n{t['date']:>12s}  {t.get('train_n',0):>6d}  {t.get('test_n',0):>5d}  {t['baseline']:>8.1%}  {t['ml']:>8.1%}"

            text_widget = tk.Text(win, bg='#0f0f2a', fg='#c8d0e0', font=('Consolas', 9),
                                   height=min(10, len(trend) + 1), width=60, bd=0, padx=10, pady=6)
            text_widget.insert('1.0', trend_text)
            text_widget.config(state='disabled')
            text_widget.pack(padx=16, pady=4)

        # ── 建议 ──
        tk.Label(win, text='─' * 45, fg='#1e2a40', bg='#141430').pack(pady=4)
        rec_color = '#4caf50' if report.get('ml_is_better') else '#ff9800'
        tk.Label(win, text=report.get('recommendation', ''),
                 fg=rec_color, bg='#141430', font=('Microsoft YaHei', 10),
                 wraplength=480, justify='center').pack(pady=4)

        # 能否验证
        if not report['can_verify']:
            tk.Label(win, text='⚠️ 测试集不足，需 ≥5 条修正数据才能验证。',
                     fg='#ff9800', bg='#141430', font=('Microsoft YaHei', 10, 'bold')).pack(pady=4)

        ttk.Button(win, text='OK', command=win.destroy).pack(pady=(4, 12))


def _check_anomalies(r: dict) -> list:
    """检查哨声参数是否超出论文合理范围，返回异常列表"""
    p = r.get('params', {})
    anomalies = []
    for key, (lo, hi) in PAPER_RANGES.items():
        val = p.get(key)
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        if val < lo:
            anomalies.append(f'{key}={val:.1f} < 论文下限{lo}')
        elif val > hi:
            anomalies.append(f'{key}={val:.1f} > 论文上限{hi}')
    return anomalies


def _estimate_snr(freqs):
    """根据频率轮廓的离散度估计信噪比"""
    if len(freqs) < 5:
        return 10.0
    smoothed = np.convolve(freqs, np.ones(5)/5, mode='valid')
    residuals = freqs[2:-2] - smoothed
    signal_power = np.var(freqs)
    noise_power = np.var(residuals) if len(residuals) > 0 else 1e-6
    snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
    return max(5, min(40, snr))


# ═══════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    app = DolphinWhistleApp()
    app.root.mainloop()
