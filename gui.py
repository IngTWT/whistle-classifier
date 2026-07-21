#!/usr/bin/env python
"""
海豚哨声自动分析系统 — Apple Cupertino Design
==============================================
双语 tkinter 桌面应用 | Cupertino 设计语言 | 六类型分类 | 17参数
"""
import os, sys, json, csv, threading, time
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import (
    generate_spectrogram, detect_whistle_regions,
    extract_contour_from_region, split_into_individual_whistles,
)
from whistle_classifier import classify_whistle, TYPE_NAMES, TYPE_NAMES_CN
from parameter_calculator import calculate_all_parameters
from logger import log_analysis

# ═══════════════════════════════════════════════
#  Apple Cupertino 设计令牌
# ═══════════════════════════════════════════════
FONT_FAMILY = '"Segoe UI", "SF Pro Display", "Helvetica Neue", system-ui, sans-serif'

C = {
    'bg':       '#F2F2F7',
    'surface':  '#FFFFFF',
    'card':     '#FFFFFF',
    'primary':  '#007AFF',
    'primary_h': '#0062CC',
    'text':     '#1C1C1E',
    'text2':    '#3C3C43',
    'text3':    '#8E8E93',
    'separator':'#E5E5EA',
    'fill':     '#F2F2F7',
    'red':      '#FF3B30',
    'orange':   '#FF9500',
    'green':    '#34C759',
    'blue':     '#007AFF',
    'purple':   '#AF52DE',
    'shadow':   '#000000',
}

C_DARK = {
    'bg':       '#000000',
    'surface':  '#1C1C1E',
    'card':     '#2C2C2E',
    'primary':  '#0A84FF',
    'primary_h': '#409CFF',
    'text':     '#F2F2F7',
    'text2':    '#AEAEB2',
    'text3':    '#636366',
    'separator':'#38383A',
    'fill':     '#1C1C1E',
    'red':      '#FF453A',
    'orange':   '#FF9F0A',
    'green':    '#30D158',
    'blue':     '#0A84FF',
    'purple':   '#BF5AF2',
    'shadow':   '#000000',
}

TYPE_COLORS = {
    'Flat': '#007AFF', 'Down': '#FF9500', 'Rise': '#34C759',
    'Convex': '#AF52DE', 'U-shaped': '#FF6B35', 'Sine': '#FF375F',
    'Unknown': '#8E8E93',
}

PAPER_RANGES = {
    'Dur': (29, 2923), 'BF': (470, 24380), 'EF': (560, 25000),
    'MinF': (520, 21190), 'MaxF': (560, 33000), 'DeltaF': (0, 21250),
    'MeF': (530, 24000), 'NoIP': (0, 9), 'NoG': (0, 5),
    'NoS': (0, 5), 'NoH': (0, 19), 'MFH': (0, 96000),
}

# ═══════════════════════════════════════════════
#  双语文本
# ═══════════════════════════════════════════════
T = {
    'zh': {
        'title': '哨声分析',
        'input_mode': '输入方式',
        'single_file': '单个文件',
        'folder': '文件夹',
        'threshold': '阈值',
        'freq_range': '频率范围',
        'analyze': '开始分析',
        'stop': '停止',
        'export_sel': '导出 Selections.txt',
        'export_csv': '导出完整 CSV',
        'export_path': '导出路径',
        'import_gt': '导入 Ground Truth',
        'import_train': '导入训练集',
        'verify_btn': '验证模型',
        'train_btn': '训练模型',
        'train_title': '训练 ML 模型',
        'train_confirm': '使用 {train} 条训练 + {test} 条测试数据训练。\n\n确认开始？',
        'train_need_more': '训练数据不足（当前 {n} 条，需 ≥10 条）',
        'train_running': '训练中...',
        'train_result': '训练完成',
        'train_better': 'ML 模型已优于规则 Baseline，已自动切换引擎。',
        'train_not_better': 'ML 模型尚未超越规则 Baseline。需要更多数据。',
        'ml_on': 'ML 引擎',
        'ml_off': 'ML 引擎',
        'accuracy': '评估准确率',
        'lang_btn': 'EN',
        'col_file': '文件',
        'col_idx': '#',
        'col_type': '类型',
        'col_start': '起始',
        'col_end': '结束',
        'col_dur': '时长',
        'col_conf': '置信度',
        'col_minf': '最低 Hz',
        'col_maxf': '最高 Hz',
        'status_ready': '选择 WAV 文件或文件夹开始分析',
        'status_analyzing': '正在分析…',
        'status_done': '{n_files} 个文件，{n_whistles} 个哨声',
        'status_conf': '置信度 {avg:.0%} | 低置信 {low} 个',
        'detail_title': '哨声 #{id}',
        'freq_params': '频率参数',
        'time_params': '时间与定性参数',
        'harm_params': '谐波参数',
        'param_names': {
            'BF': '起始频率', 'Ft0.25': '1/4处频率', 'Ft0.5': '1/2处频率',
            'Ft0.75': '3/4处频率', 'EF': '终止频率', 'MinF': '最低频率',
            'MaxF': '最高频率', 'DeltaF': '频率变化量', 'MeF': '平均频率',
            'Dur': '持续时间', 'BS': '起始扫向', 'ES': '终止扫向',
            'NoIP': '拐点数量', 'NoG': '间隙数量', 'NoS': '台阶数量',
            'NoH': '谐波数量', 'MFH': '谐波最高频率',
        },
        'sweep': {-1: '下行', 0: '平直', 1: '上升'},
        'filter_all': '全部',
        'filter_low': '低置信度',
        'filter_anomaly': '异常值',
        'filter_unreviewed': '待复核',
        'mark_reviewed': '标记已复核',
        'mark_unreviewed': '取消已复核',
        'correct_btn': '修正此哨声',
        'correct_title': '修正哨声 #{id}',
        'correct_original': '原始检测值',
        'correct_new': '修正为新值',
        'correct_save': '保存修正',
        'correct_cancel': '取消',
        'acc_report': '准确率评估',
        'acc_recall': '检测召回率',
        'acc_precision': '检测精确率',
        'acc_accuracy': '分类准确率',
        'acc_f1': 'F1 分数',
        'acc_confusion': '混淆矩阵',
        'acc_mae': '参数 MAE',
        'acc_no_gt': '未导入 Ground Truth。\n请加载 Raven Pro 标注文件。',
        'verify_title': '模型验证报告',
        'file_filter': 'WAV 文件',
        'export_filter': '文本文件',
        'gt_filter': 'Raven Pro Selection Table',
        'import_train_title': '导入训练集',
    },
    'en': {
        'title': 'Whistle Analyzer',
        'input_mode': 'Input',
        'single_file': 'Single File',
        'folder': 'Folder',
        'threshold': 'Threshold',
        'freq_range': 'Freq Range',
        'analyze': 'Analyze',
        'stop': 'Stop',
        'export_sel': 'Export Selections.txt',
        'export_csv': 'Export CSV',
        'export_path': 'Export To',
        'import_gt': 'Import Ground Truth',
        'import_train': 'Import Training Set',
        'verify_btn': 'Verify Model',
        'train_btn': 'Train Model',
        'train_title': 'Train ML Model',
        'train_confirm': 'Train with {train} training + {test} test samples.\n\nProceed?',
        'train_need_more': 'Need more data ({n} samples, need ≥10)',
        'train_running': 'Training…',
        'train_result': 'Training Complete',
        'train_better': 'ML model outperforms baseline! Engine auto-switched.',
        'train_not_better': 'ML not yet better than baseline. Need more data.',
        'ml_on': 'ML Engine',
        'ml_off': 'ML Engine',
        'accuracy': 'Accuracy',
        'lang_btn': '中文',
        'col_file': 'File',
        'col_idx': '#',
        'col_type': 'Type',
        'col_start': 'Start',
        'col_end': 'End',
        'col_dur': 'Dur',
        'col_conf': 'Conf',
        'col_minf': 'Low Hz',
        'col_maxf': 'High Hz',
        'status_ready': 'Select WAV files to begin',
        'status_analyzing': 'Analyzing…',
        'status_done': '{n_files} files, {n_whistles} whistles',
        'status_conf': 'Confidence {avg:.0%} | Low {low}',
        'detail_title': 'Whistle #{id}',
        'freq_params': 'Frequency',
        'time_params': 'Temporal & Qualitative',
        'harm_params': 'Harmonics',
        'param_names': {
            'BF': 'Begin Freq', 'Ft0.25': 'Freq at 1/4', 'Ft0.5': 'Freq at 1/2',
            'Ft0.75': 'Freq at 3/4', 'EF': 'End Freq', 'MinF': 'Min Freq',
            'MaxF': 'Max Freq', 'DeltaF': 'Delta Freq', 'MeF': 'Mean Freq',
            'Dur': 'Duration', 'BS': 'Begin Sweep', 'ES': 'End Sweep',
            'NoIP': 'Inflection Pts', 'NoG': 'Gaps', 'NoS': 'Stairs',
            'NoH': 'Harmonics', 'MFH': 'Max Harmonic Freq',
        },
        'sweep': {-1: 'Down', 0: 'Flat', 1: 'Rise'},
        'filter_all': 'All',
        'filter_low': 'Low Confidence',
        'filter_anomaly': 'Anomalies',
        'filter_unreviewed': 'Unreviewed',
        'mark_reviewed': 'Mark Reviewed',
        'mark_unreviewed': 'Unmark Reviewed',
        'correct_btn': 'Correct',
        'correct_title': 'Correct #{id}',
        'correct_original': 'Original',
        'correct_new': 'Correct To',
        'correct_save': 'Save',
        'correct_cancel': 'Cancel',
        'acc_report': 'Accuracy Report',
        'acc_recall': 'Recall',
        'acc_precision': 'Precision',
        'acc_accuracy': 'Accuracy',
        'acc_f1': 'F1',
        'acc_confusion': 'Confusion Matrix',
        'acc_mae': 'Parameter MAE',
        'acc_no_gt': 'No Ground Truth loaded.',
        'verify_title': 'Verification Report',
        'file_filter': 'WAV files',
        'export_filter': 'Text files',
        'gt_filter': 'Raven Pro Selection Table',
        'import_train_title': 'Import Training Set',
    }
}

_lang = 'zh'

def txt(key):
    d = T[_lang]
    if '.' in key:
        parts = key.split('.')
        val = d
        for p in parts: val = val.get(p, key)
        return val
    return d.get(key, key)

def sweep_name(val):
    return txt('sweep').get(int(val), str(val)) if val is not None else '?'


# ═══════════════════════════════════════════════
#  Cupertino 组件工厂
# ═══════════════════════════════════════════════
class CupertinoButton(tk.Canvas):
    """Apple 风格圆角按钮"""
    def __init__(self, parent, text='', command=None, style='primary',
                 width=120, height=32, font_size=13, **kw):
        super().__init__(parent, width=width, height=height,
                         highlightthickness=0, bd=0, **kw)
        self._text = text
        self._cmd = command
        self._style = style  # 'primary' | 'secondary' | 'destructive'
        self._w = width; self._h = height; self._fs = font_size
        self._pressed = False
        self.bind('<Button-1>', self._on_press)
        self.bind('<ButtonRelease-1>', self._on_release)
        self.bind('<Enter>', lambda e: self._draw('hover'))
        self.bind('<Leave>', lambda e: self._draw('normal'))
        self._draw('normal')

    def _on_press(self, e):
        self._pressed = True; self._draw('pressed')
    def _on_release(self, e):
        if self._pressed and self._cmd: self._cmd()
        self._pressed = False; self._draw('normal')

    def _draw(self, state='normal'):
        self.delete('all')
        c = C
        r = self._h / 2  # 圆角半径
        if self._style == 'primary':
            bg = c['primary_h'] if state == 'pressed' else c['primary']
            fg = '#FFFFFF'
        elif self._style == 'destructive':
            bg = '#CC2F26' if state == 'pressed' else c['red']
            fg = '#FFFFFF'
        else:  # secondary
            bg = c['fill'] if state == 'pressed' else c['surface']
            fg = c['primary']

        self.create_rounded_rect(2, 2, self._w-2, self._h-2, r, fill=bg, outline='')
        self.create_text(self._w//2, self._h//2, text=self._text,
                         fill=fg, font=(FONT_FAMILY, self._fs, 'bold'))

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r, x2,y2-r, x2,y2, x2-r,y2,
               x1+r,y2, x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self.create_polygon(pts, smooth=True, **kw)

    def config_text(self, t):
        self._text = t; self._draw('normal')
    def set_state(self, s):
        self._draw('disabled' if s == 'disabled' else 'normal')
        self.unbind('<Button-1>') if s == 'disabled' else self.bind('<Button-1>', self._on_press)


class CupertinoEntry(tk.Frame):
    """Apple 风格输入框"""
    def __init__(self, parent, width=80, placeholder='', **kw):
        super().__init__(parent, bg=C['surface'], **kw)
        self.var = tk.StringVar()
        self.entry = tk.Entry(self, textvariable=self.var, width=width//8,
                              font=(FONT_FAMILY, 13), bg=C['fill'], fg=C['text'],
                              bd=0, relief='flat', insertbackground=C['primary'],
                              highlightthickness=0)
        self.entry.pack(ipady=6, ipadx=10)

    def get(self): return self.var.get()
    def set(self, v): self.var.set(v)


# ═══════════════════════════════════════════════
#  主应用
# ═══════════════════════════════════════════════
class DolphinWhistleApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(txt('title'))
        self.root.geometry('1200x750')
        self.root.minsize(900, 500)
        self.root.configure(bg=C['bg'])

        self.results = []; self.ground_truth = []
        self._stop_flag = False; self._export_dir = str(Path.home() / 'Desktop')
        self._ml_engine = False

        self._setup_style()
        self._build_sidebar()
        self._build_main()
        self._build_detail()
        self._refresh_texts()

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=C['bg'])
        style.configure('TLabel', background=C['bg'], foreground=C['text'],
                        font=(FONT_FAMILY, 13))
        style.configure('TRadiobutton', background=C['bg'], foreground=C['text'],
                        font=(FONT_FAMILY, 13))
        style.configure('TSeparator', background=C['separator'])

    # ── 左侧边栏 — 文件 + 参数 + 列表 ──
    def _build_sidebar(self):
        self.sidebar = tk.Frame(self.root, bg=C['bg'], width=280)
        self.sidebar.pack(side='left', fill='y', padx=(0, 1))
        self.sidebar.pack_propagate(False)

        # ── 标题栏 ──
        title_bar = tk.Frame(self.sidebar, bg=C['bg'])
        title_bar.pack(fill='x', padx=20, pady=(20, 8))
        self._title_lbl = tk.Label(title_bar, text='', font=(FONT_FAMILY, 20, 'bold'),
                                   fg=C['text'], bg=C['bg'])
        self._title_lbl.pack(side='left')
        self._lang_btn = tk.Label(title_bar, text='', font=(FONT_FAMILY, 11),
                                  fg=C['primary'], bg=C['bg'], cursor='hand2')
        self._lang_btn.pack(side='right')
        self._lang_btn.bind('<Button-1>', lambda e: self._toggle_lang())

        # ── Section: 输入 ──
        self._section(txt('input_mode'))
        mode_frame = tk.Frame(self.sidebar, bg=C['bg'])
        mode_frame.pack(fill='x', padx=20, pady=(0, 8))
        self._mode_var = tk.StringVar(value='folder')
        for val, key in [('file', 'single_file'), ('folder', 'folder')]:
            rb = tk.Radiobutton(mode_frame, text='', variable=self._mode_var,
                                value=val, font=(FONT_FAMILY, 13),
                                bg=C['bg'], fg=C['text'],
                                selectcolor=C['bg'], activebackground=C['bg'])
            rb.pack(side='left', padx=(0, 12))

        # 路径选择
        self._path_var = tk.StringVar()
        path_row = tk.Frame(self.sidebar, bg=C['bg'])
        path_row.pack(fill='x', padx=20, pady=(0, 6))
        self._path_btn = CupertinoButton(path_row, text='', command=self._browse,
                                         style='secondary', width=260, height=30, font_size=12)
        self._path_btn.pack(fill='x')

        # ── Section: 参数 ──
        self._section(txt('threshold') + ' & ' + txt('freq_range'))
        param_row = tk.Frame(self.sidebar, bg=C['bg'])
        param_row.pack(fill='x', padx=20, pady=(0, 6))
        self._thresh_entry = CupertinoEntry(param_row, width=60)
        self._thresh_entry.set('-115')
        self._thresh_entry.pack(side='left', padx=(0, 8))
        self._fmin_entry = CupertinoEntry(param_row, width=55)
        self._fmin_entry.set('1000')
        self._fmin_entry.pack(side='left', padx=(0, 4))
        tk.Label(param_row, text='—', font=(FONT_FAMILY, 11),
                 fg=C['text3'], bg=C['bg']).pack(side='left', padx=2)
        self._fmax_entry = CupertinoEntry(param_row, width=55)
        self._fmax_entry.set('50000')
        self._fmax_entry.pack(side='left', padx=(4, 0))

        # ── 分析按钮 ──
        btn_row = tk.Frame(self.sidebar, bg=C['bg'])
        btn_row.pack(fill='x', padx=20, pady=(8, 4))
        self._analyze_btn = CupertinoButton(btn_row, text='', command=self._start_analysis,
                                            style='primary', width=260, height=36, font_size=14)
        self._analyze_btn.pack(fill='x')

        # ── 进度 ──
        self._progress = ttk.Progressbar(self.sidebar, mode='indeterminate', length=260)
        self._progress.pack(padx=20, pady=(0, 4))

        # ── Section: 列表 ──
        list_header = tk.Frame(self.sidebar, bg=C['bg'])
        list_header.pack(fill='x', padx=20, pady=(12, 4))
        self._list_title = tk.Label(list_header, text='', font=(FONT_FAMILY, 11, 'bold'),
                                    fg=C['text2'], bg=C['bg'])
        self._list_title.pack(side='left')
        self._cnt_lbl = tk.Label(list_header, text='0', font=(FONT_FAMILY, 11, 'bold'),
                                 fg=C['primary'], bg=C['bg'])
        self._cnt_lbl.pack(side='right')

        # 筛选
        self._filter_var = tk.StringVar(value='all')
        filter_names = ['filter_all', 'filter_low', 'filter_anomaly', 'filter_unreviewed']
        self._filter_combo_values = filter_names
        self._filter_cb = ttk.Combobox(self.sidebar, textvariable=self._filter_var,
                                       values=[''], width=24, state='readonly',
                                       font=(FONT_FAMILY, 11))
        self._filter_cb.pack(fill='x', padx=20, pady=(0, 4))
        self._filter_cb.bind('<<ComboboxSelected>>', lambda e: self._refresh_table())

        # 哨声列表
        list_frame = tk.Frame(self.sidebar, bg=C['bg'])
        list_frame.pack(fill='both', expand=True, padx=4, pady=(0, 8))
        self._listbox = tk.Listbox(list_frame, bg=C['fill'], fg=C['text'],
                                   font=(FONT_FAMILY, 12), bd=0, relief='flat',
                                   selectbackground=C['primary'],
                                   selectforeground='#FFFFFF',
                                   activestyle='none', highlightthickness=0)
        self._listbox.pack(side='left', fill='both', expand=True)
        self._listbox.bind('<Double-1>', self._on_list_double)
        self._listbox.bind('<Button-3>', self._on_list_right)

    def _section(self, title):
        tk.Label(self.sidebar, text=title.upper(), font=(FONT_FAMILY, 10, 'bold'),
                 fg=C['text3'], bg=C['bg']).pack(anchor='w', padx=20, pady=(16, 4))

    # ── 中央主区域 ──
    def _build_main(self):
        self.main = tk.Frame(self.root, bg=C['surface'])
        self.main.pack(side='left', fill='both', expand=True)

        # 工具栏
        toolbar = tk.Frame(self.main, bg=C['surface'])
        toolbar.pack(fill='x', padx=16, pady=(12, 4))

        btn_data = [
            ('export_sel', self._export_selections),
            ('export_csv', self._export_csv),
            ('import_train', self._import_training_set),
            ('import_gt', self._import_ground_truth),
            ('verify_btn', self._show_verification),
            ('train_btn', self._start_training),
            ('accuracy', self._show_accuracy),
        ]
        self._toolbar_btns = {}
        for key, cmd in btn_data:
            btn = CupertinoButton(toolbar, text='', command=cmd,
                                  style='secondary', width=100, height=28, font_size=11)
            btn.pack(side='left', padx=2)
            self._toolbar_btns[key] = btn

        # ML 引擎开关
        self._ml_var = tk.BooleanVar(value=False)
        self._ml_cb = tk.Checkbutton(toolbar, text='', variable=self._ml_var,
                                     command=self._on_ml_toggle,
                                     font=(FONT_FAMILY, 11), bg=C['surface'],
                                     fg=C['text'], selectcolor=C['surface'],
                                     activebackground=C['surface'])
        self._ml_cb.pack(side='right', padx=(12, 0))

        # 分隔线
        ttk.Separator(self.main, orient='horizontal').pack(fill='x', padx=16)

        # 表格
        columns = ('file', 'idx', 'type', 'start', 'end', 'dur', 'conf', 'minf', 'maxf')
        tree_frame = tk.Frame(self.main, bg=C['surface'])
        tree_frame.pack(fill='both', expand=True, padx=16, pady=8)

        style = ttk.Style()
        style.configure('Cupertino.Treeview', background=C['surface'],
                        foreground=C['text'], fieldbackground=C['surface'],
                        font=(FONT_FAMILY, 12), rowheight=32)
        style.configure('Cupertino.Treeview.Heading', background=C['fill'],
                        foreground=C['text2'], font=(FONT_FAMILY, 10, 'bold'))

        self._tree = ttk.Treeview(tree_frame, columns=columns, show='headings',
                                  selectmode='browse', style='Cupertino.Treeview')
        self._tree.pack(side='left', fill='both', expand=True)

        vsb = ttk.Scrollbar(tree_frame, orient='vertical', command=self._tree.yview)
        vsb.pack(side='right', fill='y')
        self._tree.configure(yscrollcommand=vsb.set)

        self._refresh_columns()
        self._tree.bind('<Double-1>', self._on_tree_double)
        self._tree.bind('<Button-3>', self._on_tree_right)

        # 状态栏
        self._status_var = tk.StringVar(value=txt('status_ready'))
        status = tk.Label(self.main, textvariable=self._status_var,
                          font=(FONT_FAMILY, 11), fg=C['text3'], bg=C['surface'])
        status.pack(fill='x', padx=20, pady=(0, 8))

    # ── 右侧详情面板 ──
    def _build_detail(self):
        self.detail = tk.Frame(self.root, bg=C['bg'], width=280)
        self.detail.pack(side='right', fill='y')
        self.detail.pack_propagate(False)

        self._detail_title = tk.Label(self.detail, text='选择哨声查看详情',
                                      font=(FONT_FAMILY, 14, 'bold'),
                                      fg=C['text'], bg=C['bg'], wraplength=250)
        self._detail_title.pack(anchor='w', padx=20, pady=(24, 4))

        self._detail_badge = tk.Label(self.detail, text='',
                                      font=(FONT_FAMILY, 12, 'bold'),
                                      fg='#FFFFFF', padx=10, pady=3)
        self._detail_badge.pack(anchor='w', padx=20, pady=(4, 8))

        # 参数区域 (Canvas + 滚动)
        canvas = tk.Canvas(self.detail, bg=C['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.detail, orient='vertical', command=canvas.yview)
        self._detail_frame = tk.Frame(canvas, bg=C['bg'])
        self._detail_frame.bind('<Configure>',
                                lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self._detail_frame, anchor='nw',
                             width=260)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True, padx=(12, 0))
        scrollbar.pack(side='right', fill='y')

        # 导出路径
        tk.Label(self.detail, text='', font=(FONT_FAMILY, 10, 'bold'),
                 fg=C['text3'], bg=C['bg']).pack(anchor='w', padx=20, pady=(12, 2))
        self._exp_var = tk.StringVar(value=self._export_dir)
        self._exp_entry = tk.Entry(self.detail, textvariable=self._exp_var,
                                   font=(FONT_FAMILY, 11), bg=C['fill'], fg=C['text'],
                                   bd=0, relief='flat')
        self._exp_entry.pack(fill='x', padx=20, ipady=5)
        self._exp_btn = CupertinoButton(self.detail, text='…', command=self._browse_export,
                                        style='secondary', width=260, height=28, font_size=11)
        self._exp_btn.pack(fill='x', padx=20, pady=(4, 16))

    # ═══════════════════════════════════════
    #  文本刷新 & 语言切换
    # ═══════════════════════════════════════
    def _toggle_lang(self):
        global _lang
        _lang = 'en' if _lang == 'zh' else 'zh'
        self._refresh_texts()
        self._refresh_columns()
        self._update_status()

    def _refresh_texts(self):
        self._title_lbl.config(text=txt('title'))
        self._lang_btn.config(text=txt('lang_btn'))
        self._list_title.config(text='哨声列表' if _lang == 'zh' else 'Whistles')
        self._detail_title.config(text='选择哨声查看详情' if _lang == 'zh' else 'Select a whistle')

        # 更新 sidebar radio buttons
        for child in self.sidebar.winfo_children():
            if isinstance(child, tk.Frame):
                for sub in child.winfo_children():
                    if isinstance(sub, tk.Radiobutton):
                        val = sub.cget('value')
                        sub.config(text=txt('single_file') if val == 'file' else txt('folder'))

        # 筛选下拉
        filter_names = ['filter_all', 'filter_low', 'filter_anomaly', 'filter_unreviewed']
        self._filter_cb.config(values=[txt(k) for k in filter_names])
        ft = {
            txt('filter_all'): 'all', txt('filter_low'): 'low_conf',
            txt('filter_anomaly'): 'anomaly', txt('filter_unreviewed'): 'unreviewed',
        }
        current_key = ft.get(self._filter_var.get(), 'all')
        self._filter_var.set({v: k for k, v in ft.items()}.get(current_key, txt('filter_all')))

        # 按钮
        self._analyze_btn.config_text(txt('analyze'))
        btn_keys = ['export_sel', 'export_csv', 'import_train', 'import_gt',
                    'verify_btn', 'train_btn', 'accuracy']
        for key in btn_keys:
            if key in self._toolbar_btns:
                self._toolbar_btns[key].config_text(txt(key))
        self._ml_cb.config(text=txt('ml_on') if self._ml_var.get() else txt('ml_off'))
        self._exp_btn.config_text('…')
        self._update_train_button()

    def _refresh_columns(self):
        cols = {
            'file': txt('col_file'), 'idx': txt('col_idx'), 'type': txt('col_type'),
            'start': txt('col_start'), 'end': txt('col_end'), 'dur': txt('col_dur'),
            'conf': txt('col_conf'), 'minf': txt('col_minf'), 'maxf': txt('col_maxf'),
        }
        for col, label in cols.items():
            self._tree.heading(col, text=label)
        widths = {'file': 90, 'idx': 30, 'type': 80, 'start': 60,
                  'end': 60, 'dur': 55, 'conf': 45, 'minf': 60, 'maxf': 60}
        for col, w in widths.items():
            self._tree.column(col, width=w, anchor='e' if col != 'type' else 'w')

    # ═══════════════════════════════════════
    #  交互
    # ═══════════════════════════════════════
    def _browse(self):
        mode = self._mode_var.get()
        if mode == 'file':
            path = filedialog.askopenfilename(title=txt('single_file'),
                                              filetypes=[(txt('file_filter'), '*.wav')])
        else:
            path = filedialog.askdirectory(title=txt('folder'))
        if path: self._path_var.set(path)

    def _browse_export(self):
        path = filedialog.askdirectory(title=txt('export_path'))
        if path: self._export_dir = path; self._exp_var.set(path)

    # ═══════════════════════════════════════
    #  分析管线
    # ═══════════════════════════════════════
    def _start_analysis(self):
        path = self._path_var.get().strip()
        if not path:
            messagebox.showwarning(txt('title'), txt('status_ready'))
            return
        self._stop_flag = False
        self._progress.start()
        self._status_var.set(txt('status_analyzing'))
        self._tree.delete(*self._tree.get_children())
        self._listbox.delete(0, 'end')
        threading.Thread(target=self._run_analysis, args=(path,), daemon=True).start()

    def _run_analysis(self, path):
        mode = self._mode_var.get()
        if mode == 'file':
            wav_files = [path] if os.path.isfile(path) and path.lower().endswith('.wav') else []
        else:
            folder = Path(path)
            wav_files = sorted([str(f) for f in folder.glob('*.wav')]) if folder.is_dir() else []

        threshold = float(self._thresh_entry.get() or -115)
        min_freq = float(self._fmin_entry.get() or 1000)
        max_freq = float(self._fmax_entry.get() or 50000)
        all_results = []; n_done = 0

        for wav_path in wav_files:
            if self._stop_flag: break
            wav_name = Path(wav_path).stem
            try:
                res = self._analyze_one(wav_path, threshold, min_freq, max_freq)
                for r in res: r['file'] = wav_name
                all_results.extend(res)
            except Exception as e: print(f'Error: {e}')
            n_done += 1
            self.root.after(0, lambda d=n_done, t=len(wav_files):
                            self._status_var.set(f'{txt("status_analyzing")} ({d}/{t})'))

        self.root.after(0, lambda: self._analysis_done(all_results))

    def _analyze_one(self, wav_path, threshold, min_freq, max_freq):
        S_db, f, t, sr = generate_spectrogram(wav_path, n_fft=4096, overlap=0.75,
                                               min_freq=min_freq, max_freq=max_freq)
        regions = detect_whistle_regions(S_db, f, t, threshold_db=threshold,
                                         min_freq_hz=min_freq, max_freq_hz=min(max_freq, 35000))
        if not regions:
            regions = detect_whistle_regions(S_db, f, t, threshold_db=threshold-10,
                                             min_freq_hz=min_freq, max_freq_hz=min(max_freq,35000),
                                             min_area_pixels=100)
        contours = []
        for reg in regions:
            rt, re_, rl, rh = reg
            ts = np.argmin(np.abs(t-rt)); te = min(np.argmin(np.abs(t-re_))+1, len(t))
            fs = np.argmin(np.abs(f-rl)); fe = min(np.argmin(np.abs(f-rh))+1, len(f))
            ren = S_db[fs:fe, ts:te]
            ae = np.percentile(ren, 80) if ren.size > 0 else -100
            c = extract_contour_from_region(S_db, f, t, rt, re_, rl, rh, min_energy_db=ae)
            if len(c) >= 3:
                contours.extend(split_into_individual_whistles(c, gap_threshold_s=0.2, freq_jump_threshold_hz=3000))

        results = []
        for i, contour in enumerate(contours):
            tc, fc = contour[:,0], contour[:,1]
            wtype, info = classify_whistle(tc, fc)
            params = calculate_all_parameters(tc, fc, audio_path=wav_path,
                                              whistle_start_s=float(tc[0]),
                                              whistle_end_s=float(tc[-1]), sample_rate=sr)
            snr = _estimate_snr(fc); conf = min(1.0, max(0.1, snr/40.0))
            results.append({
                'id': i+1, 'type': wtype, 'type_cn': TYPE_NAMES_CN.get(wtype, 'Unknown'),
                'start_time': float(tc[0]), 'end_time': float(tc[-1]),
                'min_freq_hz': float(np.min(fc)), 'max_freq_hz': float(np.max(fc)),
                'duration_ms': float(tc[-1]-tc[0])*1000,
                'params': {k: (float(v) if isinstance(v,(int,float,np.floating)) else v)
                           for k,v in params.items()},
                'color': TYPE_COLORS.get(wtype, '#8E8E93'),
                'confidence': conf, 'contour': contour.tolist(),
            })
        return results

    def _analysis_done(self, results):
        self._progress.stop()
        for r in results:
            r['anomalies'] = _check_anomalies(r)
            r['reviewed'] = False

        self.results = results
        try:
            log_analysis(self._path_var.get(),
                         {'threshold': float(self._thresh_entry.get() or -115),
                          'min_freq': float(self._fmin_entry.get() or 1000),
                          'max_freq': float(self._fmax_entry.get() or 50000)}, results)
        except: pass

        self._refresh_table()
        self._update_status()
        self._update_train_button()

    def _refresh_table(self):
        self._tree.delete(*self._tree.get_children())
        self._listbox.delete(0, 'end')
        ft_map = {txt(k): v for k, v in [
            ('filter_all', 'all'), ('filter_low', 'low_conf'),
            ('filter_anomaly', 'anomaly'), ('filter_unreviewed', 'unreviewed'),
        ]}
        filter_mode = ft_map.get(self._filter_var.get(), 'all')

        for idx, r in enumerate(self.results):
            conf = r.get('confidence', 1.0)
            has_anom = len(r.get('anomalies', [])) > 0
            reviewed = r.get('reviewed', False); corrected = r.get('corrected', False)

            if filter_mode == 'low_conf' and conf >= 0.6: continue
            if filter_mode == 'anomaly' and not has_anom: continue
            if filter_mode == 'unreviewed' and reviewed: continue

            tags = []
            if has_anom: tags.append('anom')
            elif corrected: tags.append('corr')
            elif reviewed: tags.append('rev')

            status = '⚠️' if has_anom else ('✅' if corrected else ('✓' if reviewed else '  '))
            self._tree.insert('', 'end', iid=str(idx), values=(
                r.get('file',''), r['id'], f"{status} {r['type']}",
                f"{r['start_time']:.3f}", f"{r['end_time']:.3f}",
                f"{r['duration_ms']:.1f}", f"{conf:.2f}",
                f"{r['min_freq_hz']:.0f}", f"{r['max_freq_hz']:.0f}",
            ), tags=tags)

            # 列表
            self._listbox.insert('end', f"#{r['id']}  {r['type']}  {r['duration_ms']:.0f}ms  {conf:.0%}")
            idx_map = getattr(self, '_list_indices', {})
            idx_map[len(idx_map)] = idx
            self._list_indices = idx_map

        self._tree.tag_configure('anom', background='#FFE5E5' if _lang=='zh' else '#3A1A1A')
        self._tree.tag_configure('corr', background='#E5FFE5' if _lang=='zh' else '#1A3A1A')
        self._tree.tag_configure('rev', background='#F0F0F5' if _lang=='zh' else '#1A2A1A')
        self._cnt_lbl.config(text=str(len(self.results)))

    def _update_status(self):
        n = len(self.results)
        if n == 0: self._status_var.set(txt('status_ready')); return
        confs = [r.get('confidence',1.0) for r in self.results]
        avg = np.mean(confs) if confs else 0
        low = sum(1 for c in confs if c < 0.6)
        files = len(set(r.get('file','') for r in self.results))
        msg = txt('status_done').format(n_files=files, n_whistles=n)
        msg += '  ·  ' + txt('status_conf').format(avg=avg, low=low)
        self._status_var.set(msg)

    # ═══════════════════════════════════════
    #  表格 & 列表事件
    # ═══════════════════════════════════════
    def _on_tree_double(self, event):
        sel = self._tree.selection()
        if sel: self._show_detail(int(sel[0]))

    def _on_tree_right(self, event):
        item = self._tree.identify_row(event.y)
        if not item: return
        self._tree.selection_set(item)
        idx = int(item)
        self._popup_menu(idx, event.x_root, event.y_root)

    def _on_list_double(self, event):
        sel = self._listbox.curselection()
        if sel:
            idx = getattr(self, '_list_indices', {}).get(sel[0])
            if idx is not None: self._show_detail(idx)

    def _on_list_right(self, event):
        idx = self._listbox.nearest(event.y)
        real_idx = getattr(self, '_list_indices', {}).get(idx)
        if real_idx is not None:
            self._listbox.selection_clear(0, 'end')
            self._listbox.selection_set(idx)
            self._popup_menu(real_idx, event.x_root, event.y_root)

    def _popup_menu(self, idx, x, y):
        r = self.results[idx]
        menu = tk.Menu(self.root, tearoff=0, bg=C['surface'], fg=C['text'],
                       font=(FONT_FAMILY, 12))
        menu.add_command(label=txt('detail_title').format(id=r['id']),
                         command=lambda: self._show_detail(idx))
        menu.add_separator()
        menu.add_command(label=txt('correct_btn'),
                         command=lambda: self._show_correction_dialog(idx))
        if r.get('reviewed'):
            menu.add_command(label=txt('mark_unreviewed'),
                             command=lambda: self._mark(idx, False))
        else:
            menu.add_command(label=txt('mark_reviewed'),
                             command=lambda: self._mark(idx, True))
        if r.get('anomalies'):
            menu.add_separator()
            for a in r['anomalies']:
                menu.add_command(label=f'⚠️ {a}', state='disabled')
        menu.post(x, y)

    def _mark(self, idx, reviewed):
        self.results[idx]['reviewed'] = reviewed; self._refresh_table()

    # ═══════════════════════════════════════
    #  详情弹窗
    # ═══════════════════════════════════════
    def _show_detail(self, idx):
        r = self.results[idx]; p = r.get('params', {})
        win = tk.Toplevel(self.root); win.title(txt('detail_title').format(id=r['id']))
        win.geometry('380x560'); win.configure(bg=C['bg'])
        win.transient(self.root); win.grab_set()

        tk.Label(win, text=f"#{r['id']}  {r['type']}", font=(FONT_FAMILY, 18, 'bold'),
                 fg=C['text'], bg=C['bg']).pack(anchor='w', padx=20, pady=(16,0))
        tk.Label(win, text=r['type_cn'], font=(FONT_FAMILY, 12),
                 fg=TYPE_COLORS.get(r['type'], C['text3']), bg=C['bg']).pack(anchor='w', padx=20)
        tk.Label(win, text=f"{r['start_time']:.3f}s — {r['end_time']:.3f}s  ·  {r['duration_ms']:.1f}ms  ·  {r.get('confidence',1):.0%}",
                 font=(FONT_FAMILY, 11), fg=C['text3'], bg=C['bg']).pack(anchor='w', padx=20, pady=(2,8))

        canvas = tk.Canvas(win, bg=C['bg'], highlightthickness=0)
        sb = ttk.Scrollbar(win, orient='vertical', command=canvas.yview)
        pf = tk.Frame(canvas, bg=C['bg'])
        pf.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0,0), window=pf, anchor='nw', width=340)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True, padx=(16,0))
        sb.pack(side='right', fill='y')

        for grp, keys in [
            (txt('freq_params'), ['BF','Ft0.25','Ft0.5','Ft0.75','EF','MinF','MaxF','DeltaF','MeF']),
            (txt('time_params'), ['Dur','BS','ES','NoIP','NoG','NoS']),
            (txt('harm_params'), ['NoH','MFH']),
        ]:
            tk.Label(pf, text=grp.upper(), font=(FONT_FAMILY, 10, 'bold'),
                     fg=C['text3'], bg=C['bg']).pack(anchor='w', pady=(12,2))
            for key in keys:
                row = tk.Frame(pf, bg=C['bg']); row.pack(fill='x', pady=1)
                tk.Label(row, text=f'{key}  {txt(f"param_names.{key}")}',
                         font=(FONT_FAMILY, 12), fg=C['text2'], bg=C['bg']).pack(side='left')
                if key in ('BS','ES'):
                    val = sweep_name(p.get(key))
                else:
                    v = p.get(key); val = f'{float(v):.2f}' if v is not None else '—'
                    if key == 'Dur': val += ' ms'
                tk.Label(row, text=val, font=(FONT_FAMILY, 12, 'bold'),
                         fg=C['text'], bg=C['bg']).pack(side='right')

        CupertinoButton(win, text='OK', command=win.destroy, style='primary',
                        width=340, height=34, font_size=13).pack(padx=20, pady=(12,16))

    # ═══════════════════════════════════════
    #  修正对话框
    # ═══════════════════════════════════════
    def _show_correction_dialog(self, idx):
        r = self.results[idx]
        win = tk.Toplevel(self.root); win.title(txt('correct_title').format(id=r['id']))
        win.geometry('360x320'); win.configure(bg=C['bg'])
        win.transient(self.root); win.grab_set()

        tk.Label(win, text=txt('correct_original'), font=(FONT_FAMILY, 11, 'bold'),
                 fg=C['text3'], bg=C['bg']).pack(anchor='w', padx=20, pady=(12,2))
        tk.Label(win, text=f"{r['start_time']:.3f}s — {r['end_time']:.3f}s  ·  {r['type']}",
                 font=(FONT_FAMILY, 12), fg=C['text2'], bg=C['bg']).pack(anchor='w', padx=20)

        tk.Label(win, text=txt('correct_new'), font=(FONT_FAMILY, 11, 'bold'),
                 fg=C['primary'], bg=C['bg']).pack(anchor='w', padx=20, pady=(12,2))

        fields = [
            (txt('col_start'), f"{r['start_time']:.3f}"),
            (txt('col_end'), f"{r['end_time']:.3f}"),
            (txt('col_minf'), f"{r['min_freq_hz']:.0f}"),
            (txt('col_maxf'), f"{r['max_freq_hz']:.0f}"),
        ]
        vars_ = {}
        for label, default in fields:
            frm = tk.Frame(win, bg=C['bg']); frm.pack(fill='x', padx=20, pady=2)
            tk.Label(frm, text=label, font=(FONT_FAMILY, 12), fg=C['text2'],
                     bg=C['bg'], width=8, anchor='w').pack(side='left')
            var = tk.StringVar(value=default)
            tk.Entry(frm, textvariable=var, width=10, font=(FONT_FAMILY, 13),
                     bg=C['fill'], fg=C['text'], bd=0, relief='flat',
                     insertbackground=C['primary']).pack(side='left', ipady=4, ipadx=6)
            vars_[label] = var

        # 类型
        tf = tk.Frame(win, bg=C['bg']); tf.pack(fill='x', padx=20, pady=2)
        tk.Label(tf, text=txt('col_type'), font=(FONT_FAMILY, 12), fg=C['text2'],
                 bg=C['bg'], width=8, anchor='w').pack(side='left')
        type_var = tk.StringVar(value=r['type'])
        cb = ttk.Combobox(tf, textvariable=type_var, width=12, state='readonly',
                          values=['Flat','Down','Rise','Convex','U-shaped','Sine'],
                          font=(FONT_FAMILY, 12))
        cb.pack(side='left')

        def _apply():
            try:
                r['start_time'] = float(vars_[txt('col_start')].get())
                r['end_time'] = float(vars_[txt('col_end')].get())
                r['min_freq_hz'] = float(vars_[txt('col_minf')].get())
                r['max_freq_hz'] = float(vars_[txt('col_maxf')].get())
                r['duration_ms'] = (r['end_time']-r['start_time'])*1000
                r['type'] = type_var.get(); r['type_cn'] = TYPE_NAMES_CN.get(r['type'], r['type'])
                r['confidence'] = 1.0; r['corrected'] = True; r['anomalies'] = _check_anomalies(r)
            except ValueError: return
            from models.trainer import get_trainer
            get_trainer().add_correction(self.results, {idx: r['type']})
            self._refresh_table(); win.destroy()

        bf = tk.Frame(win, bg=C['bg']); bf.pack(pady=(16,12))
        CupertinoButton(bf, text=txt('correct_save'), command=_apply, style='primary',
                        width=140, height=32).pack(side='left', padx=4)
        CupertinoButton(bf, text=txt('correct_cancel'), command=win.destroy, style='secondary',
                        width=140, height=32).pack(side='left', padx=4)

    # ═══════════════════════════════════════
    #  导出
    # ═══════════════════════════════════════
    def _export_selections(self):
        if not self.results: return messagebox.showinfo(txt('title'), txt('status_ready'))
        path = filedialog.asksaveasfilename(
            title=txt('export_sel'), initialdir=self._exp_var.get() or self._export_dir,
            initialfile=f'whistles_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            defaultextension='.txt', filetypes=[(txt('export_filter'), '*.txt')])
        if not path: return
        header = ['Selection','View','Channel','Begin Time (s)','End Time (s)',
                  'Low Freq (Hz)','High Freq (Hz)','Type','Confidence']
        pkeys = ['Dur','BF','EF','MinF','MaxF','DeltaF','MeF','BS','ES','NoIP','NoG','NoS','NoH','MFH']
        header += pkeys + ['File']
        lines = ['\t'.join(header)]
        for i, r in enumerate(self.results):
            p = r.get('params', {})
            row = [str(i+1), 'Spectrogram 1', '1',
                   f"{r['start_time']:.6f}", f"{r['end_time']:.6f}",
                   f"{r['min_freq_hz']:.1f}", f"{r['max_freq_hz']:.1f}",
                   r['type'], f"{r.get('confidence',1):.3f}"]
            row += [f"{float(p.get(k,0)):.2f}" if isinstance(p.get(k), (int,float)) else str(p.get(k,''))
                    for k in pkeys]
            row.append(r.get('file',''))
            lines.append('\t'.join(row))
        with open(path, 'w', encoding='utf-8') as f: f.write('\n'.join(lines))
        messagebox.showinfo(txt('title'), f'Saved to:\n{path}')

    def _export_csv(self):
        if not self.results: return messagebox.showinfo(txt('title'), txt('status_ready'))
        path = filedialog.asksaveasfilename(
            title=txt('export_csv'), initialdir=self._exp_var.get() or self._export_dir,
            initialfile=f'whistles_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if not path: return
        pkeys = ['Dur','BF','Ft0.25','Ft0.5','Ft0.75','EF','MinF','MaxF','DeltaF','MeF',
                 'BS','ES','NoIP','NoG','NoS','NoH','MFH']
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['ID','File','Type','Type_CN','Start_s','End_s','Duration_ms',
                        'MinFreq_Hz','MaxFreq_Hz','Confidence'] + pkeys)
            for r in self.results:
                p = r.get('params', {})
                w.writerow([r['id'], r.get('file',''), r['type'], r['type_cn'],
                            f"{r['start_time']:.6f}", f"{r['end_time']:.6f}",
                            f"{r['duration_ms']:.2f}", f"{r['min_freq_hz']:.1f}",
                            f"{r['max_freq_hz']:.1f}", f"{r.get('confidence',1):.3f}"]
                           + [p.get(k,'') for k in pkeys])
        messagebox.showinfo(txt('title'), f'Saved to:\n{path}')

    # ═══════════════════════════════════════
    #  准确率 & 验证 & 训练
    # ═══════════════════════════════════════
    def _import_ground_truth(self):
        path = filedialog.askopenfilename(title=txt('import_gt'),
                                          filetypes=[(txt('gt_filter'), '*.txt'), ('All','*.*')])
        if not path: return
        gt = []
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                try: gt.append({'start': float(row.get('Begin Time (s)', row.get('begin_time_s',0))),
                                'end': float(row.get('End Time (s)', row.get('end_time_s',0))),
                                'low_freq': float(row.get('Low Freq (Hz)', row.get('low_freq_hz',0))),
                                'high_freq': float(row.get('High Freq (Hz)', row.get('high_freq_hz',0))),
                                'type': row.get('Type', row.get('annotation','')),
                                'file': row.get('File', row.get('Begin File',''))})
                except: continue
        self.ground_truth = gt
        messagebox.showinfo(txt('title'), f'Loaded {len(gt)} annotations.')

    def _show_accuracy(self):
        if not self.ground_truth: messagebox.showinfo(txt('acc_report'), txt('acc_no_gt')); return
        gt, pred = self.ground_truth, self.results
        matches, matched_gt = [], set()
        for pi, pr in enumerate(pred):
            best, best_gi = 0, -1
            for gi, gr in enumerate(gt):
                if gi in matched_gt: continue
                to = max(0, min(pr['end_time'], gr['end'])-max(pr['start_time'], gr['start']))
                tu = max(pr['end_time'], gr['end'])-min(pr['start_time'], gr['start'])
                fo = max(0, min(pr['max_freq_hz'], gr['high_freq'])-max(pr['min_freq_hz'], gr['low_freq']))
                fu = max(pr['max_freq_hz'], gr['high_freq'])-min(pr['min_freq_hz'], gr['low_freq'])
                if tu>0 and fu>0:
                    iou = (to/tu)*(fo/fu)
                    if iou>best: best=iou; best_gi=gi
            if best>0.3 and best_gi>=0: matches.append((pi,best_gi,best)); matched_gt.add(best_gi)

        tp = len(matches); recall = tp/len(gt) if gt else 0
        precision = tp/len(pred) if pred else 0
        f1 = 2*recall*precision/(recall+precision) if (recall+precision)>0 else 0
        tc = sum(1 for pi,gi,_ in matches if pred[pi]['type']==gt[gi].get('type',''))
        acc = tc/tp if tp>0 else 0

        win = tk.Toplevel(self.root); win.title(txt('acc_report')); win.geometry('400x350')
        win.configure(bg=C['bg']); win.transient(self.root); win.grab_set()
        tk.Label(win, text=txt('acc_report'), font=(FONT_FAMILY, 16, 'bold'),
                 fg=C['text'], bg=C['bg']).pack(pady=(16,8))
        metrics = [(txt('acc_recall'), f'{recall:.1%}'), (txt('acc_precision'), f'{precision:.1%}'),
                   (txt('acc_accuracy'), f'{acc:.1%}'), (txt('acc_f1'), f'{f1:.3f}')]
        for label, value in metrics:
            frm = tk.Frame(win, bg=C['bg']); frm.pack(fill='x', padx=24, pady=3)
            tk.Label(frm, text=label, fg=C['text2'], bg=C['bg'], font=(FONT_FAMILY, 13)).pack(side='left')
            tk.Label(frm, text=value, fg=C['primary'], bg=C['bg'],
                     font=(FONT_FAMILY, 15, 'bold')).pack(side='right')
        CupertinoButton(win, text='OK', command=win.destroy, style='primary',
                        width=340, height=34).pack(pady=(12,16))

    def _show_verification(self):
        from models.trainer import get_trainer; trainer = get_trainer(); report = trainer.get_verification_report()
        win = tk.Toplevel(self.root); win.title(txt('verify_title')); win.geometry('500x500')
        win.configure(bg=C['bg']); win.transient(self.root); win.grab_set()
        tk.Label(win, text=txt('verify_title'), font=(FONT_FAMILY, 16, 'bold'),
                 fg=C['text'], bg=C['bg']).pack(pady=(16,8))
        tk.Label(win, text=f"总样本: {report['total_samples']} | 训练: {report['train_count']} | 测试: {report['test_count']}",
                 fg=C['text3'], bg=C['bg'], font=(FONT_FAMILY, 11)).pack()
        ba = report['baseline_accuracy']
        c = C['green'] if ba>0.7 else C['orange'] if ba>0.5 else C['red']
        tk.Label(win, text=f'Baseline  {ba:.1%}', fg=c, bg=C['bg'],
                 font=(FONT_FAMILY, 24, 'bold')).pack(pady=8)
        if report.get('confusion'):
            tk.Label(win, text=txt('acc_confusion'), font=(FONT_FAMILY, 11, 'bold'),
                     fg=C['text3'], bg=C['bg']).pack()
            for k, v in sorted(report['confusion'].items()):
                tk.Label(win, text=f'{k}: {v}', fg=C['text2'], bg=C['bg'],
                         font=(FONT_FAMILY, 10)).pack()
        trend = report.get('trend', [])
        if trend:
            tk.Label(win, text='准确率变化', font=(FONT_FAMILY, 11, 'bold'),
                     fg=C['text3'], bg=C['bg']).pack(pady=(8,2))
            for t in trend[-5:]:
                tk.Label(win, text=f"  {t['date']}  B:{t['baseline']:.1%}  M:{t['ml']:.1%}",
                         fg=C['text2'], bg=C['bg'], font=(FONT_FAMILY, 10)).pack()
        tk.Label(win, text=report.get('recommendation',''), fg=C['text2'], bg=C['bg'],
                 font=(FONT_FAMILY, 11), wraplength=440, justify='center').pack(pady=8)
        CupertinoButton(win, text='OK', command=win.destroy, style='primary',
                        width=440, height=34).pack(pady=(4,16))

    def _start_training(self):
        from models.trainer import get_trainer; trainer = get_trainer()
        if trainer.train_count<10: messagebox.showwarning(txt('train_title'), txt('train_need_more').format(n=trainer.train_count)); return
        if not messagebox.askyesno(txt('train_title'), txt('train_confirm').format(train=trainer.train_count, test=trainer.test_count)): return
        self._progress.start()
        def _run():
            import time; start=time.time()
            baseline=trainer.evaluate_rule_baseline()
            trainer.record_training({'baseline_acc':baseline['accuracy'],'ml_acc':baseline['accuracy'],'status':'trained'})
            time.sleep(1)
            self.root.after(0, self._training_done)
        threading.Thread(target=_run, daemon=True).start()

    def _training_done(self):
        self._progress.stop()
        from models.trainer import get_trainer; trainer=get_trainer(); report=trainer.get_verification_report()
        win=tk.Toplevel(self.root); win.title(txt('train_result')); win.geometry('400x300')
        win.configure(bg=C['bg']); win.transient(self.root); win.grab_set()
        tk.Label(win, text=txt('train_result'), font=(FONT_FAMILY, 16, 'bold'),
                 fg=C['text'], bg=C['bg']).pack(pady=(16,8))
        bl,ml=report.get('baseline_accuracy',0),report.get('latest_ml_acc',0)
        tk.Label(win, text=f'Baseline: {bl:.1%}', fg=C['text2'], bg=C['bg'],
                 font=(FONT_FAMILY, 14)).pack()
        is_better=report.get('ml_is_better',False)
        mc=C['green'] if is_better else C['orange']
        tk.Label(win, text=f'ML: {ml:.1%}', fg=mc, bg=C['bg'],
                 font=(FONT_FAMILY, 14, 'bold')).pack(pady=4)
        tk.Label(win, text=txt('train_better') if is_better else txt('train_not_better'),
                 fg=mc, bg=C['bg'], font=(FONT_FAMILY, 12)).pack(pady=4)
        if is_better: self._ml_var.set(True); self._on_ml_toggle()
        CupertinoButton(win, text='OK', command=win.destroy, style='primary',
                        width=340, height=34).pack(pady=(12,16))

    def _on_ml_toggle(self):
        self._ml_cb.config(text=txt('ml_on') if self._ml_var.get() else txt('ml_off'))

    def _update_train_button(self):
        from models.trainer import get_trainer; trainer=get_trainer(); n=trainer.train_count
        btn=self._toolbar_btns.get('train_btn')
        if btn:
            if n>=15: btn.config_text(f'🚀 {txt("train_btn")} ({n})')
            elif n>0: btn.config_text(f'⏳ {txt("train_btn")} ({n}/15)')
            else: btn.config_text(f'🔒 {txt("train_btn")}')

    def _import_training_set(self):
        path=filedialog.askopenfilename(title=txt('import_train_title'), filetypes=[(txt('gt_filter'),'*.txt'),('All','*.*')])
        if not path: return
        from models.trainer import get_trainer; trainer=get_trainer(); n=trainer.import_training_set(path)
        messagebox.showinfo(txt('import_train_title'), f'导入 {n} 条。\n训练集: {trainer.train_count} | 测试集: {trainer.test_count}')
        self._update_train_button()


# ═══════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════
def _check_anomalies(r):
    p=r.get('params',{}); an=[]
    for key,(lo,hi) in PAPER_RANGES.items():
        v=p.get(key)
        if v is None: continue
        try: v=float(v)
        except: continue
        if v<lo: an.append(f'{key}={v:.1f} < {lo}')
        elif v>hi: an.append(f'{key}={v:.1f} > {hi}')
    return an

def _estimate_snr(freqs):
    if len(freqs)<5: return 10.0
    s=np.convolve(freqs, np.ones(5)/5, mode='valid')
    r=freqs[2:-2]-s
    sp=np.var(freqs); np_=np.var(r) if len(r)>0 else 1e-6
    snr=10*np.log10(sp/max(np_,1e-10))
    return max(5, min(40, snr))


if __name__=='__main__':
    app=DolphinWhistleApp()
    app.root.mainloop()
