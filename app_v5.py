import io
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import streamlit as st

def to_compact_percent(label):
    """Convert 10.000% → 10%, 62.50% → 62.5%, 1%CC → 1%CC, 12.0%extra → 12%extra"""
    s = str(label).replace(' ', '')
    # e.g. "10%", "62.5%", "10%CC", "62.5%CC", "12.0%extra"
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)%([A-Za-z]*)", s)
    if m:
        num, suffix = m.group(1), m.group(2)
        # Only display decimals if needed
        numf = float(num)
        if abs(numf - int(numf)) < 1e-8:
            num_str = f"{int(numf)}"
        else:
            num_str = f"{numf}".rstrip('0').rstrip('.')  # minimal decimals
        return f"{num_str}%" + suffix
    return label

def parse_base_percent(label):
    """Extract base percent (e.g. from 10%CC → 10%). Returns string."""
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)%", str(label).replace(' ', ''))
    if m:
        numf = float(m.group(1))
        if abs(numf - int(numf)) < 1e-8:
            num_str = f"{int(numf)}"
        else:
            num_str = f"{numf}".rstrip('0').rstrip('.')
        return f"{num_str}%"
    return label

def duration_to_minutes(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().lower()
    m = re.match(r'^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-z]+)\s*$', s)
    if not m:
        try:
            return float(s)
        except Exception:
            raise ValueError(f"Unrecognized duration format: {x!r}")
    val, unit = m.groups()
    val = float(val)
    if unit.startswith('min'):
        return val
    if unit.startswith('hour') or unit in ('h', 'hr', 'hrs'):
        return val * 60.0
    if unit.startswith('day') or unit == 'd':
        return val * 1440.0
    return val

def load_ifd_xlsx(uploaded_file):
    # Find header row with "Duration"
    df = None
    for header_row in range(0, 8):
        uploaded_file.seek(0)
        try:
            cand = pd.read_excel(uploaded_file, sheet_name=0, header=header_row)
        except Exception:
            continue
        if 'Duration' in cand.columns:
            df = cand
            break
    if df is None:
        uploaded_file.seek(0)
        raw = pd.read_excel(uploaded_file, sheet_name=0, header=None)
        header_idx = raw.index[raw.iloc[:,0].astype(str).str.strip().str.lower() == 'duration']
        if len(header_idx) == 0:
            raise ValueError("Could not find a 'Duration' header in the workbook.")
        h = int(header_idx[0])
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, sheet_name=0, header=h)
    first_col = list(df.columns)[0]
    if first_col != 'Duration':
        df = df.rename(columns={first_col: 'Duration'})
    df = df.dropna(subset=['Duration']).copy()
    df['duration_min'] = df['Duration'].map(duration_to_minutes)
    df = df.dropna(subset=['duration_min']).sort_values('duration_min').reset_index(drop=True)

    # Accept all data columns
    data_cols = [c for c in df.columns if c not in ('Duration', 'duration_min')]
    prob_info, other_cols = [], []
    for c in data_cols:
        compact_label = to_compact_percent(c)
        base = parse_base_percent(compact_label)
        try:
            value = float(base.rstrip('%'))
            prob_info.append((c, value, 'CC' in compact_label or 'cc' in compact_label, compact_label, base))
        except:
            other_cols.append((c, compact_label))
    # Sort numerically, CC curves after original
    prob_info_sorted = sorted(prob_info, key=lambda x: (x[1], x[2]))
    ordered_cols, labels, bases = [], [], []
    for orig_col, value, is_cc, label, base in prob_info_sorted:
        ordered_cols.append(orig_col)
        labels.append(label)
        bases.append(base)
    for orig_col, label in other_cols:
        ordered_cols.append(orig_col)
        labels.append(label)
        bases.append(label)
    return df[['duration_min'] + ordered_cols], ordered_cols, labels, bases

def load_ifd_from_text_csv(uploaded_file):
    uploaded_file.seek(0)
    text = uploaded_file.read().decode('utf-8', errors='replace')
    rows = []
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line: continue
        m = re.match(r'^Duration:\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)', line)
        if m:
            if current: rows.append(current)
            val, unit = m.groups()
            current = {'duration_min': duration_to_minutes(f"{val} {unit}")}
            continue
        m = re.match(r'^(.+?)\s*:\s*([0-9]+(?:\.[0-9]+)?)$', line)
        if m and current is not None:
            k, v = m.groups()
            current[k.strip()] = float(v)
    if current: rows.append(current)
    if not rows: raise ValueError("No data found in CSV. Expected 'Duration:' blocks.")

    df = pd.DataFrame(rows).sort_values('duration_min').reset_index(drop=True)
    data_cols = [c for c in df.columns if c != 'duration_min']
    prob_info, other_cols = [], []
    for c in data_cols:
        compact_label = to_compact_percent(c)
        base = parse_base_percent(compact_label)
        try:
            value = float(base.rstrip('%'))
            prob_info.append((c, value, 'CC' in compact_label or 'cc' in compact_label, compact_label, base))
        except:
            other_cols.append((c, compact_label))
    prob_info_sorted = sorted(prob_info, key=lambda x: (x[1], x[2]))
    ordered_cols, labels, bases = [], [], []
    for orig_col, value, is_cc, label, base in prob_info_sorted:
        ordered_cols.append(orig_col)
        labels.append(label)
        bases.append(base)
    for orig_col, label in other_cols:
        ordered_cols.append(orig_col)
        labels.append(label)
        bases.append(label)
    return df[['duration_min'] + ordered_cols], ordered_cols, labels, bases

def configure_xticks(ax, xmin, xmax):
    def within(a, lo, hi):
        return [t for t in a if lo <= t <= hi]
    minute_ticks = [1, 2, 3, 5, 10, 15, 20, 30, 45]
    hour_ticks   = [60, 120, 180, 360, 720, 1080, 1440]
    day_ticks    = [2*1440, 3*1440, 5*1440, 7*1440, 10*1440]
    xticks = sorted(set(within(minute_ticks, xmin, xmax) +
                        within(hour_ticks, xmin, xmax) +
                        within(day_ticks, xmin, xmax)))
    if xticks: ax.set_xticks(xticks)
    def tick_fmt(val, pos):
        if val <= 0: return ''
        if val < 60:
            return f'{int(val)}'
        if val < 1440:
            v = val / 60.0
            return f'{v:.0f}' if abs(v - round(v)) < 1e-6 else f'{v:.1f}'
        v = val / 1440.0
        return f'{v:.0f}' if abs(v - round(v)) < 1e-6 else f'{v:.1f}'
    ax.xaxis.set_major_formatter(FuncFormatter(tick_fmt))

def build_log_mm_ticks(ymin, ymax):
    ticks = [0.2, 0.5]
    emax = int(np.ceil(np.log10(max(ymax, 1))))
    for e in range(0, emax + 1):
        for m in (1, 2, 3, 5):
            ticks.append(m * (10 ** e))
    return sorted([t for t in ticks if ymin <= t <= ymax * 1.001])

def nice_mm_formatter():
    def fmt(v, pos):
        if v < 1:
            return f'{v:.1f}'.rstrip('0').rstrip('.')
        if abs(v - round(v)) < 1e-6:
            return f'{int(round(v))}'
        return f'{v:g}'
    return FuncFormatter(fmt)

def plot_ifd(df, ordered_cols, labels, bases):
    plt.rcParams.update({
        'axes.grid': True,
        'grid.linestyle': ':',
        'grid.color': '#777777',
        'grid.alpha': 0.6
    })
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    x = df['duration_min'].values

    # Unique base keys for color assignment
    unique_bases = []
    for base in bases:
        if base not in unique_bases:
            unique_bases.append(base)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    base_to_color = {base: color_cycle[i % len(color_cycle)] for i, base in enumerate(unique_bases)}

    for idx, (col, label, base) in enumerate(zip(ordered_cols, labels, bases)):
        # 'CC' columns get dashed, others solid
        is_cc = 'CC' in label.upper()
        color = base_to_color.get(base, f'C{idx}')
        linestyle = '--' if is_cc else '-'
        y = df[col].astype(float).values
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, lw=2)

    y_min = float(df[ordered_cols].to_numpy().min())
    y_max = float(df[ordered_cols].to_numpy().max())
    ax.set_ylim(y_min * 0.9, y_max * 1.05)
    y_ticks = build_log_mm_ticks(y_min, y_max)
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(nice_mm_formatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel('Depth (mm)', fontsize=12)

    xmin, xmax = x.min(), x.max()
    ax.set_xlim(xmin, xmax)
    configure_xticks(ax, xmin, xmax)
    for xv in (60, 1440):
        if xmin <= xv <= xmax:
            ax.axvline(x=xv, color='k', lw=0.8, alpha=0.5)
    leg = ax.legend(title='AEP', loc='lower right', frameon=True)
    leg.get_frame().set_alpha(0.9)

    band = inset_axes(ax, width="100%", height="4%", loc="lower center",
                      bbox_to_anchor=(0, -0.08, 1, 1),
                      bbox_transform=ax.transAxes, borderpad=0)
    band.set_xscale('log')
    band.set_xlim(ax.get_xlim())
    band.set_ylim(0, 1)
    band.axis('off')
    def draw_box(x0, x1, label):
        left = max(x0, xmin); right = min(x1, xmax)
        if right <= left: return
        rect = Rectangle((left, 0), right-left, 1, fill=False, edgecolor='k', lw=1.0)
        band.add_patch(rect)
        xc = math.sqrt(left * right)
        band.text(xc, 0.5, label, ha='center', va='center', fontsize=10)
    draw_box(1, 60, 'minutes')
    draw_box(60, 1440, 'hours')
    draw_box(1440, max(xmax, 7*1440), 'days')
    fig.text(0.55, -0.05, 'Duration', ha='center', va='center', fontsize=12)
    fig.tight_layout()
    return fig

# ---- Streamlit UI ----

st.set_page_config(page_title="IFD Plotter", layout="centered")
st.title("IFD Plotter (log–log) v2")
st.caption("Upload your IFD Excel file (or suitable CSV). AEP curves (e.g. 10%, 10%CC) will be matched in color. Legend labels are compact.")

uploaded = st.file_uploader("Upload your IFD_data.xlsx or .csv", type=["xlsx", "xls", "csv"])

if uploaded is not None:
    try:
        name = uploaded.name.lower()
        if name.endswith((".xlsx", ".xls")):
            df, ordered_cols, labels, bases = load_ifd_xlsx(uploaded)
        else:
            df, ordered_cols, labels, bases = load_ifd_from_text_csv(uploaded)

        fig = plot_ifd(df, ordered_cols, labels, bases)
        st.pyplot(fig, clear_figure=True, use_container_width=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        st.download_button(
            "Download PNG",
            buf,
            file_name=(Path(uploaded.name).stem + "_plot.png"),
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Could not build the plot: {e}")