"""
HD2 Log Viewer — Streamlit Edition
Original Tkinter app by ERRORX2. Converted to Streamlit.
"""

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Constants & keyword sets
# ──────────────────────────────────────────────────────────────────────────────

CURRENT_VERSION = "1.4.1"
GITHUB_REPO = "ERRORX2/HD2-LOG-VIEWER"

_EXCLUDE_RAW = frozenset([
    '[MB]', '[GB]', '[A]', 'PWM', '(STATIC)',
    'THERMAL LIMIT', 'POWER LIMIT', 'TDC LIMIT', 'PPT LIMIT', 'EDC LIMIT',
    'FREQUENCY LIMIT', 'CLOCK LIMIT',
])
_EXCLUDE_NAME = frozenset([
    'FREQUENCYLIMIT', 'ACCUMULATED', 'EFFECTIVECLOCK', 'REQUESTEDCLOCK',
    'TARGETTEMP', 'LIMITTEMP', 'THERMALLIMIT', 'POWERLIMIT',
    'CURRENTLIMIT', 'TDCLIMIT', 'PPTLIMIT', 'EDCLIMIT',
])
_THROTTLE_KW   = ('THROTTLING', 'RELIABILITY', 'PERFCAP')
_SMART_KW      = ('REMAINING LIFE', 'WEAR LEVEL', 'ENDURANCE REMAINING')
_WHEA_KW       = ('WHEA', 'MACHINE CHECK', 'MCE', 'MCA')
_ERROR_KW      = ('ECC', 'BAD SECTOR', 'REALLOCATED', 'PENDING SECTOR',
                  'UNCORRECTABLE', 'CRC ERROR')
_RAIL_SKIP     = ('GPU PCIE', 'PCIE', '12VHPWR', 'INPUT')
_TEMP_TRIGGERS = frozenset(['TEMP', '°C', 'HOTSPOT', 'TDIE', 'TCTL'])

TIME_COLUMN_CANDIDATES = ['time', 'date', 'timestamp', 'elapsed', 'clock', '#']
TIME_FORMATS = ['%H:%M:%S', '%H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
                '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%H:%M']

GROUPS_FILE = Path("groups.json")

# ──────────────────────────────────────────────────────────────────────────────
# groups.json persistence  (mirrors the original save_config / load_config)
# ──────────────────────────────────────────────────────────────────────────────

def save_config():
    """Write presets + settings to groups.json next to the script."""
    config = {
        "groups": st.session_state.custom_groups,
        "settings": {
            "temp_limits":  st.session_state.temp_limits,
            "volt_rails":   {k: list(v) for k, v in st.session_state.volt_rails.items()},
            "misc":         st.session_state.misc,
        }
    }
    try:
        GROUPS_FILE.write_text(json.dumps(config, indent=4), encoding="utf-8")
    except Exception:
        pass


def load_config():
    """Read groups.json and return (groups, temp_limits, volt_rails, misc).
    Falls back to defaults if the file doesn't exist or is malformed."""
    if not GROUPS_FILE.exists():
        return {}, dict(DEFAULT_TEMP_LIMITS), dict(DEFAULT_VOLT_RAILS), dict(DEFAULT_MISC)
    try:
        data = json.loads(GROUPS_FILE.read_text(encoding="utf-8"))
        groups = data.get("groups", {})
        sets   = data.get("settings", {})
        tl = {**DEFAULT_TEMP_LIMITS, **sets.get("temp_limits", {})}
        vr = {k: tuple(v) for k, v in
              {**{k: list(v) for k, v in DEFAULT_VOLT_RAILS.items()},
               **sets.get("volt_rails", {})}.items()}
        misc = {**DEFAULT_MISC, **sets.get("misc", {})}
        return groups, tl, vr, misc
    except Exception:
        return {}, dict(DEFAULT_TEMP_LIMITS), dict(DEFAULT_VOLT_RAILS), dict(DEFAULT_MISC)

# ──────────────────────────────────────────────────────────────────────────────
# Default thresholds
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_TEMP_LIMITS = {
    'HOTSPOT': 95.0, 'HOT SPOT': 95.0,
    'GPU': 88.0,
    'VRAM': 95.0, 'MEMORY': 95.0,
    'VRM': 110.0,
    'CORE': 95.0,
    'TDIE': 95.0, 'TCTL': 95.0,
    'CCD': 90.0, 'CCX': 90.0,
    'IOD': 95.0,
    'SOCKET': 95.0,
    'COOLANT': 45.0, 'LIQUID': 45.0, 'WATER': 45.0,
    'SSD': 65.0, 'NVME': 65.0, 'HDD': 55.0,
    'DRIVE': 70.0,
    'TEMPERATURE': 70.0,
    'CHIPSET': 90.0, 'PCH': 90.0,
    'MOSFET': 110.0, 'CHOKE': 110.0,
}
DEFAULT_VOLT_RAILS = {
    '+12V': (11.4, 12.6),
    '+5V':  (4.75, 5.25),
    '+3.3V': (3.13, 3.46),
}
DEFAULT_MISC = {
    'cpu_volt_lo': 0.8,  'cpu_volt_hi': 1.55,
    'gpu_volt_max': 1.1,
    'dram_volt_lo': 1.1, 'dram_volt_hi': 1.55,
    'fan_min_rpm': 400.0,
    'cpu_power_max': 300.0, 'gpu_power_max': 500.0, 'total_power_max': 600.0,
    'latency_max_ms': 50.0, 'frametime_max_ms': 100.0, 'fps_min': 10.0,
    'coolant_max': 45.0, 'memory_load_max': 95.0,
    'drive_spare_min': 10.0, 'drive_life_min': 10.0,
    'vcore_droop_max': 0.3, 'clock_instability': 0.35, 'throttle_threshold': 0.9,
    'sig_cpu_thermal_pct': 0.85, 'sig_cpu_thermal_samples': 10,
    'sig_fan_stall_rpm': 100.0, 'sig_fan_min_spinning': 200.0,
    'sig_fan_hot_cpu_c': 70.0, 'sig_fan_hot_gpu_c': 65.0,
    'sig_drive_temp_max': 70.0, 'sig_vrm_temp_max': 105.0,
    'sig_ram_exhaust_pct': 95.0, 'sig_vram_overflow_pct': 98.0,
    'sig_cpu_bn_gpu_pct': 60.0, 'sig_cpu_bn_cpu_pct': 85.0, 'sig_cpu_bn_samples': 10,
    'sig_stutter_mult': 3.0, 'sig_stutter_min_hits': 5,
    'sig_tdr_clock_frac': 0.5, 'sig_ppt_sat_pct': 0.98, 'sig_ppt_sat_samples': 15,
    'sig_clock_stretch_mhz': 500.0, 'sig_disk_busy_pct': 99.9, 'sig_disk_busy_samples': 3,
    'sig_v12_lo': 11.4, 'sig_v5_lo': 4.75, 'sig_v5_hi': 5.25,
    'sig_v33_lo': 3.14, 'sig_v33_hi': 3.47,
}

# ──────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────────────

def init_session():
    # Load persisted config from groups.json exactly once per process startup.
    if '_config_loaded' not in st.session_state:
        groups, tl, vr, misc = load_config()
        st.session_state._config_loaded = True
        st.session_state.custom_groups  = groups
        st.session_state.temp_limits    = tl
        st.session_state.volt_rails     = vr
        st.session_state.misc           = misc

    # Transient keys — only set if missing
    transient_defaults = {
        'df': None,
        'time_col': '',
        'time_series': None,
        'filename': '',
        'ref_df': None,
        'selected_cols': [],
    }
    for k, v in transient_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_misc(key):
    return st.session_state.misc.get(key, DEFAULT_MISC[key])


# ──────────────────────────────────────────────────────────────────────────────
# CSV loading & time detection
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(uploaded) -> Tuple[pd.DataFrame, str, Optional[pd.Series]]:
    raw = uploaded.read()
    # Detect delimiter
    try:
        sample = raw[:2048].decode('latin-1', errors='ignore')
        dialect = csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
    except Exception:
        sep = ','

    df = pd.DataFrame()
    for enc in ['utf-8-sig', 'latin-1', 'cp1252']:
        try:
            import io
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep,
                             on_bad_lines='skip', engine='python')
            if not df.empty:
                break
        except Exception:
            continue

    if df.empty:
        raise ValueError("Could not parse the CSV file.")

    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
    time_col, time_series = _detect_time_column(df)

    for col in df.columns:
        if col == time_col:
            continue
        try:
            s = df[col].astype(str).str.replace(',', '.', regex=False)
            cleaned = s.str.replace(r'[^\d\.\-eE]', '', regex=True)
            df[col] = pd.to_numeric(cleaned, errors='coerce')
        except Exception:
            continue

    # Trim trailing all-zero rows
    while len(df) > 1:
        numeric_cols = [c for c in df.columns if c != time_col]
        check = df.iloc[-1][numeric_cols]
        if (check == 0).sum() + check.isna().sum() > len(numeric_cols) / 2:
            df = df.iloc[:-1]
        else:
            break

    df.ffill(inplace=True)
    if time_series is not None:
        time_series = time_series.iloc[:len(df)].reset_index(drop=True)
    df = df.reset_index(drop=True)

    return df, time_col, time_series


def _detect_time_column(df: pd.DataFrame):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    found_col = None
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in cols_lower:
            found_col = cols_lower[candidate]
            break
    if not found_col:
        return '', None

    raw = df[found_col].astype(str).str.strip()
    for fmt in TIME_FORMATS:
        try:
            parsed = pd.to_datetime(raw, format=fmt, errors='coerce')
            if parsed.notna().sum() > len(parsed) * 0.8:
                first = parsed.dropna().iloc[0]
                return found_col, parsed - first
        except Exception:
            continue
    try:
        parsed = pd.to_datetime(raw, errors='coerce')
        if parsed.notna().sum() > len(parsed) * 0.8:
            first = parsed.dropna().iloc[0]
            return found_col, parsed - first
    except Exception:
        pass
    return '', None


# ──────────────────────────────────────────────────────────────────────────────
# Category helpers
# ──────────────────────────────────────────────────────────────────────────────

UI_ORDER = [
    "Temperatures (°C)", "Utilization / Load (%)", "Clock Speeds (MHz)",
    "Power / Wattage (W)", "Voltage (V)", "Fan Speeds (RPM)"
]


def get_category(n: str) -> str:
    u = n.upper()
    if '°C' in u or 'TEMP' in u: return "Temperatures (°C)"
    if '%' in u or 'USAGE' in u or 'UTILIZATION' in u: return "Utilization / Load (%)"
    if 'MHZ' in u or 'CLOCK' in u: return "Clock Speeds (MHz)"
    if ' W' in u or 'WATT' in u or 'POWER' in u: return "Power / Wattage (W)"
    if ' V' in u or 'VOLT' in u or 'VCORE' in u: return "Voltage (V)"
    if 'RPM' in u or 'FAN' in u: return "Fan Speeds (RPM)"
    if any(x in u for x in ['GPU', 'NVIDIA', 'GEFORCE', 'AMD', 'RTX', 'GTX']): return "Graphics Card (GPU)"
    if any(x in u for x in ['CPU', 'CORE ', 'AMD RYZEN', 'INTEL']): return "Processor (CPU)"
    return "Other Sensors"


def group_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for col in df.columns:
        cat = get_category(col)
        groups.setdefault(cat, []).append(col)
    return groups


def sorted_categories(groups: dict) -> List[str]:
    return [c for c in UI_ORDER if c in groups] + \
           sorted([c for c in groups if c not in UI_ORDER])


# ──────────────────────────────────────────────────────────────────────────────
# Out-of-spec detection
# ──────────────────────────────────────────────────────────────────────────────

def sustained(df: pd.DataFrame, col: str, threshold: float, n_samples: int = 5,
              above: bool = True) -> bool:
    if col not in df.columns:
        return False
    s = df[col].ffill().fillna(0).values
    count = 0
    for v in s:
        if (v >= threshold if above else v <= threshold):
            count += 1
            if count >= n_samples:
                return True
        else:
            count = 0
    return False


def is_critical(df: pd.DataFrame, col: str) -> bool:
    tl = st.session_state.temp_limits
    vr = st.session_state.volt_rails
    misc = st.session_state.misc
    raw = col.upper()
    name = raw.replace(' ', '')
    series = df[col].dropna()
    if series.empty:
        return False

    if any(x in raw for x in _EXCLUDE_RAW) or any(x in name for x in _EXCLUDE_NAME):
        return False
    if 'FRAME TIME' in raw or 'FRAMETIME' in raw:
        if '1% HIGH' in raw and '0.1%' not in raw:
            return series.max() > misc['frametime_max_ms']
        return False
    if 'FRAMERATE' in raw or ' FPS' in raw or 'FRAMES PER SECOND' in raw:
        if '0.1%' in raw and 'LOW' in raw and 'PRESENTED' not in raw:
            return series.min() <= misc['fps_min'] and series.max() > 0
        return False
    if 'LATENCY' in raw or 'RENDER TIME' in raw or 'GPU BUSY' in raw or 'CPU BUSY' in raw:
        return series.max() > misc['latency_max_ms']
    if '[MS]' in raw:
        return False
    if any(x in raw for x in _THROTTLE_KW):
        return series.max() >= misc['throttle_threshold']
    if 'YES/NO' in raw:
        if 'DRIVE FAILURE' in raw or 'DRIVE WARNING' in raw:
            return series.max() >= 1.0
        if ('THERMAL' in raw or 'POWER' in raw) and 'PERFORMANCE LIMIT' in raw:
            return series.max() >= 1.0
        return False
    if 'TOTAL ERRORS' in raw:
        return series.max() > 0
    if 'AVAILABLE SPARE' in raw and '[%]' in raw:
        return series.min() < misc['drive_spare_min']
    if any(x in raw for x in _SMART_KW):
        return series.min() < misc['drive_life_min']
    if '[%]' in raw:
        if 'LIMIT' in raw:
            return False
        if ('MEMORY' in raw or 'RAM' in raw) and ('USAGE' in raw or 'LOAD' in raw):
            return series.max() >= misc['memory_load_max']
        if 'DECODE' in raw or 'ENCODE' in raw or 'VIDEO' in raw:
            return False
    if any(x in raw for x in _WHEA_KW):
        return series.max() > 0
    if any(x in raw for x in _ERROR_KW):
        return series.max() > 0
    if '[W]' in raw and 'STATIC' not in raw and 'LIMIT' not in raw and 'PPT' not in raw:
        if 'CPU' in raw and sustained(df, col, misc['cpu_power_max'], 5):
            return True
        if 'GPU' in raw and sustained(df, col, misc['gpu_power_max'], 5):
            return True
        if 'TOTAL' in raw and sustained(df, col, misc['total_power_max'], 5):
            return True
    for rail, (low, high) in vr.items():
        if rail in raw:
            if any(x in raw for x in _RAIL_SKIP):
                continue
            return series.min() < low or series.max() > high
    if 'VCORE' in raw or 'CPU CORE VOLTAGE' in raw:
        lo, hi = misc['cpu_volt_lo'], misc['cpu_volt_hi']
        return series.min() < lo or series.max() > hi or \
               (series.max() - series.min() > misc['vcore_droop_max'])
    if 'VID' in raw and 'GPU' not in raw and 'VIDEO' not in raw:
        lo, hi = misc['cpu_volt_lo'], misc['cpu_volt_hi']
        return series.min() < lo or series.max() > hi
    if ('DRAM VOLTAGE' in raw or 'DIMM VOLTAGE' in raw or 'VDIMM' in raw) and 'GPU' not in raw:
        lo, hi = misc['dram_volt_lo'], misc['dram_volt_hi']
        return series.min() < lo or series.max() > hi
    if 'GPU CORE VOLTAGE' in raw and 'GFX' not in raw:
        return series.max() > misc['gpu_volt_max']
    if 'RPM' in raw or 'FAN SPEED' in raw:
        if series.max() > misc['fan_min_rpm'] and series.min() < misc['fan_min_rpm']:
            return True
    if any(x in name for x in _TEMP_TRIGGERS):
        matched_limit, matched_len = None, 0
        for key, limit in tl.items():
            key_norm = key.upper().replace(' ', '')
            if key_norm in name and len(key_norm) > matched_len:
                matched_limit, matched_len = limit, len(key_norm)
        return sustained(df, col, matched_limit if matched_limit is not None else 90.0, 3)
    if 'PHYSICAL MEMORY' in raw and 'LOAD' in raw:
        return series.max() >= misc['memory_load_max']
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Column helper (like _col / _col_any / _col_excl in the original)
# ──────────────────────────────────────────────────────────────────────────────

def col_find(df: pd.DataFrame, *keywords) -> Optional[str]:
    kw = [k.upper() for k in keywords]
    for c in df.columns:
        u = c.upper()
        if all(k in u for k in kw):
            return c
    return None


def col_excl(df: pd.DataFrame, keywords, excl=()) -> Optional[str]:
    kw   = [k.upper() for k in keywords]
    skip = [e.upper() for e in excl]
    for c in df.columns:
        u = c.upper()
        if all(k in u for k in kw) and not any(e in u for e in skip):
            return c
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Hardware signature analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_signatures(df: pd.DataFrame) -> List[dict]:
    tl   = st.session_state.temp_limits
    vr   = st.session_state.volt_rails
    misc = st.session_state.misc
    hits = []

    def add(name, severity, description, evidence):
        hits.append({'name': name, 'severity': severity,
                     'description': description,
                     'evidence': [str(e) for e in evidence if e and str(e).strip()]})

    def mx(c):  return df[c].max() if c and c in df.columns else 0
    def avg(c): return df[c].mean() if c and c in df.columns else 0

    cpu_temp      = col_find(df, 'TCTL') or col_find(df, 'TDIE') or col_find(df, 'CPU', 'TEMP')
    cpu_usage_col = col_find(df, 'CPU', 'USAGE') or col_find(df, 'CPU', 'UTIL') or col_find(df, 'CPU', 'LOAD')
    cpu_power     = col_find(df, 'CPU', 'PACKAGE') or col_find(df, 'CPU', 'PPT') or col_find(df, 'CPU', 'POWER')
    cpu_clock     = col_find(df, 'CORE', 'CLOCK') or col_find(df, 'CPU', 'CLOCK')
    throttle      = col_find(df, 'THROTTLE') or col_find(df, 'PROCHOT')

    gpu_hotspot   = col_excl(df, ('GPU', 'HOT'), excl=('CPU', 'LIMIT')) or \
                    col_excl(df, ('GPU', 'TEMP'), excl=('CPU',))
    gpu_usage_col = col_find(df, 'GPU', 'USAGE') or col_find(df, 'GPU', 'LOAD')
    gpu_clock     = col_find(df, 'GPU', 'CLOCK') or col_find(df, 'GPU', 'FREQUENCY')
    gpu_pwr_limit = col_find(df, 'Performance Limit - Power') or col_excl(df, ('PERFCAP', 'PWR'))
    gpu_power     = col_find(df, 'GPU', 'POWER') or col_find(df, 'BOARD', 'POWER')
    gpu_mem_usage = col_find(df, 'GPU', 'MEMORY', 'USAGE') or col_find(df, 'VRAM', 'USAGE')
    gpu_clk_col   = col_find(df, 'GPU Clock [MHz]')
    vram_j_temp   = col_find(df, 'GPU Memory Junction Temperature [°C]')
    pcie_errors   = col_find(df, 'PCI Express Error Counters')
    ft_col        = col_find(df, 'FRAME TIME') or col_find(df, 'FRAMETIME')
    gpu_wait_ms   = col_find(df, 'GPU Wait')
    gpu_busy_ms   = col_find(df, 'GPU Busy')
    ram_load      = col_find(df, 'PHYSICAL', 'MEMORY', 'LOAD')
    chipset_t     = col_find(df, 'Chipset') or col_find(df, 'Motherboard [°C]')
    usb_v_col     = col_find(df, 'USB VCC') or col_find(df, 'USB Voltage')
    fclk_col      = col_find(df, 'FCLK')
    uclk_col      = next((c for c in df.columns if 'UCLK' in c), None)
    mclk_col      = col_find(df, 'MCLK') or col_find(df, 'MEMORY CLOCK')
    gpu_mem_ded   = col_find(df, 'GPU D3D Memory Dedicated')
    gpu_mem_dyn   = col_find(df, 'GPU D3D Memory Dynamic')
    gpu_bus_col   = col_find(df, 'GPU Bus Load') or col_find(df, 'Bus Load')
    ppt_limit     = col_find(df, 'CPU', 'PPT', 'LIMIT')
    v12           = col_find(df, '+12V')
    vrm_temp      = col_find(df, 'VRM') or col_find(df, 'MOS')
    disk_busy     = col_find(df, 'TOTAL', 'ACTIVE', 'TIME') or col_find(df, 'DISK', 'BUSY')
    pcie_width    = col_find(df, 'GPU', 'PCIE', 'WIDTH')
    pcie_gen      = col_find(df, 'GPU', 'PCIE', 'GENERATION')
    gpu_12v_v     = col_find(df, 'GPU 12VHPWR Voltage') or col_find(df, 'GPU 12V Input Voltage')
    gpu_12v_w     = col_find(df, 'GPU 12VHPWR Power') or col_find(df, 'GPU Power [W]')
    drive_act     = col_find(df, 'Total Activity [%]') or col_find(df, 'Read Activity [%]')
    drive_warn    = col_find(df, 'Drive Warning [Yes/No]')
    cpu_utility   = col_find(df, 'CPU USAGE') or col_find(df, 'CPU UTILIZATION')
    whea          = col_find(df, 'WHEA')
    is_laptop     = any(k in ''.join(df.columns).upper() for k in ['BATTERY', 'CHARGE', 'AC ADAPTER'])

    # 1 CPU thermal
    if cpu_temp:
        limit = tl.get('TDIE', 95.0)
        if sustained(df, cpu_temp, limit * 0.92, misc['sig_cpu_thermal_samples']):
            sev = 'CRITICAL'
        elif sustained(df, cpu_temp, limit * 0.85, misc['sig_cpu_thermal_samples']):
            sev = 'WARNING'
        else:
            sev = None
        if sev:
            add("CPU Thermal Throttling", sev,
                "CPU is hitting its thermal ceiling. ADVICE: Check CPU cooler mounting, "
                "re-apply thermal paste, or ensure your AIO pump hasn't failed.",
                [f"Peak Temp: {mx(cpu_temp):.1f}°C", f"Limit: {limit:.0f}°C"])

    # 2 GPU thermals
    if gpu_hotspot:
        hs_max = mx(gpu_hotspot)
        hs_limit = tl.get('HOTSPOT', 95.0)
        gpu_edge = col_excl(df, ('GPU', 'TEMP'), excl=('HOTSPOT', 'MEMORY', 'CPU'))
        delta_val = 0
        if gpu_edge:
            delta_val = (df[gpu_hotspot] - df[gpu_edge]).max()
        evidence = [f"Hotspot Max: {hs_max:.1f}°C", f"Thermal Delta: {delta_val:.1f}°C"]
        if hs_max >= hs_limit:
            add("GPU Overheating (Hotspot)", "CRITICAL",
                "GPU Hotspot is at dangerous levels. ADVICE: Increase fan curves and check airflow.",
                evidence + [f"Hardware Limit: {hs_limit}°C"])
        elif hs_max > (hs_limit - 10) or delta_val >= 21.0:
            msg = ("High thermal delta detected (re-paste GPU)." if delta_val >= 21.0
                   else "GPU Hotspot approaching dangerous levels.")
            add("GPU Thermal Warning", "WARNING", msg, evidence)

    # 3 PSU +12V sag
    if v12:
        v_min = df[v12].min()
        if v_min < misc['sig_v12_lo']:
            sev = "CRITICAL" if v_min < 11.2 else "WARNING"
            add("PSU +12V Rail Sag", sev,
                "12V rail sagging below safe limits. Check PCIe/EPS cables. PSU may be failing.",
                [f"Min Voltage: {v_min:.2f}V", f"Limit: {misc['sig_v12_lo']}V"])

    # 4 GPU driver TDR
    if gpu_usage_col and gpu_clock:
        low_usage = df[gpu_usage_col] < 5
        clock_stall = (df[gpu_clock].rolling(3).std() < 1.0) & (df[gpu_clock] > 0)
        tdr_mask = (low_usage & clock_stall).rolling(5).sum() >= 3
        if (df[gpu_usage_col].rolling(10).mean() > 20).any() and tdr_mask.any():
            add("GPU Driver TDR (Timeout)", "CRITICAL",
                "GPU driver timeout pattern detected. Likely driver stall or reset.",
                [f"Confirmed Events: {int(tdr_mask.sum())} samples"])

    # 5 Drive overheating
    drive_temps = [c for c in df.columns if 'TEMP' in c.upper()
                   and any(k in c.upper() for k in ['DRIVE', 'NVME', 'SSD', 'HDD'])]
    for d_col in drive_temps:
        peak = mx(d_col)
        u = d_col.upper()
        if any(k in u for k in ['HDD', 'HARD DRIVE', 'ST']):
            crit_limit, warn_limit, dtype = 55.0, 45.0, "HDD"
        else:
            crit_limit = misc['sig_drive_temp_max']
            warn_limit = crit_limit - 10
            dtype = "SSD/NVMe"
        if peak >= crit_limit:
            add("Storage Thermal Critical", "CRITICAL",
                f"Critical heat on {dtype}. ADVICE: Power off immediately.",
                [f"Peak: {peak:.1f}°C", f"Limit: {crit_limit}°C"])
        elif peak > warn_limit:
            add("Storage Overheating", "WARNING",
                f"High temp on {dtype}. Improve airflow.",
                [f"Peak: {peak:.1f}°C"])

    # 6 WHEA
    if whea and mx(whea) > 0:
        add("Hardware (WHEA) Errors", "CRITICAL",
            "Windows detected physical hardware errors. Check RAM XMP or CPU undervolts.",
            [f"Total Errors: {int(mx(whea))}"])

    # 7 CPU power limit
    if cpu_power and ppt_limit and sustained(df, cpu_power, mx(ppt_limit) * misc['sig_ppt_sat_pct'],
                                              misc['sig_ppt_sat_samples']):
        add("CPU Power Limit Reached", "WARNING",
            "CPU performance capped by power limits. Increase PPT/PL1/PL2 in BIOS if temps allow.",
            [f"Power Sustained: {avg(cpu_power):.1f}W"])

    # 8 Fan failure
    for col in df.columns:
        if ('FAN' in col.upper() or 'RPM' in col.upper()) and '[%]' not in col:
            fan_s = df[col].ffill().fillna(0)
            if fan_s.max() > misc['sig_fan_min_spinning']:
                is_stalled = fan_s < misc['sig_fan_stall_rpm']
                is_hot = pd.Series(False, index=df.index)
                if gpu_hotspot: is_hot |= (df[gpu_hotspot] > misc['sig_fan_hot_gpu_c'])
                if cpu_temp:    is_hot |= (df[cpu_temp] > misc['sig_fan_hot_cpu_c'])
                if (is_stalled & is_hot).rolling(3).sum().max() >= 3:
                    add("Fan Stall Detected", "CRITICAL",
                        f"Fan '{col}' stopped while hot. Check for cable obstruction or failing motor.",
                        ["RPM hit 0 during load samples."])
                    break

    # 9 VRAM overflow
    if gpu_mem_usage:
        vram_val = mx(gpu_mem_usage)
        is_pct = '[%]' in gpu_mem_usage or vram_val <= 100.0
        if (vram_val > misc['sig_vram_overflow_pct'] if is_pct
                else vram_val > df[gpu_mem_usage].max() * misc['sig_vram_overflow_pct'] / 100):
            add("GPU VRAM Overflow", "WARNING",
                "GPU out of video memory. Lower Texture Quality or render resolution.",
                [f"VRAM Usage: {vram_val:.1f}{'%' if is_pct else ' MB'}"])

    # 10 S.M.A.R.T.
    life_cols = [c for c in df.columns if 'REMAINING LIFE' in c.upper() or 'DRIVE HEALTH' in c.upper()]
    for l_col in life_cols:
        life = df[l_col].min()
        if life <= 5.0:
            add("SSD Lifespan Critical", "CRITICAL",
                f"Drive at {life:.1f}% life. Replace immediately.",
                [f"Remaining Life: {life:.1f}%"])
        elif life <= 20.0:
            add("SSD Wear Warning", "WARNING",
                f"Drive at {life:.1f}% health. Plan replacement.",
                [f"Remaining Life: {life:.1f}%"])

    # 11 RAM exhaustion
    if ram_load and mx(ram_load) > misc['sig_ram_exhaust_pct']:
        add("System RAM Exhaustion", "WARNING",
            "Physical RAM nearly full. Close background apps or upgrade RAM.",
            [f"Max Load: {mx(ram_load):.1f}%"])

    # 12 Virtual memory
    v_load = col_find(df, 'VIRTUAL', 'MEMORY', 'LOAD')
    if v_load and mx(v_load) > 98:
        add("Virtual Memory Limit", "CRITICAL",
            "Commit limit full. Set Page File to System Managed.",
            [f"Commit: {mx(v_load):.1f}%"])

    # 13 CPU bottleneck
    if gpu_usage_col and cpu_usage_col:
        bn = (df[gpu_usage_col] < misc['sig_cpu_bn_gpu_pct']) & \
             (df[cpu_usage_col] > misc['sig_cpu_bn_cpu_pct'])
        if bn.rolling(misc['sig_cpu_bn_samples']).sum().max() >= misc['sig_cpu_bn_samples']:
            add("CPU Bottleneck", "WARNING",
                "CPU maxed while GPU is idling. Increase graphics settings or close background apps.",
                [f"Avg GPU during spike: {df.loc[bn, gpu_usage_col].mean():.1f}%"])

    # 14 VRM overheating
    if vrm_temp and mx(vrm_temp) > misc['sig_vrm_temp_max']:
        add("VRM Overheating", "CRITICAL",
            "Motherboard power delivery too hot. Improve airflow.",
            [f"Max: {mx(vrm_temp):.1f}°C"])

    # 15 Micro-stuttering
    if ft_col:
        ft_s = df[ft_col].ffill().dropna()
        stutter_limit = ft_s.median() * misc['sig_stutter_mult']
        stutters = ft_s[ft_s > stutter_limit]
        if len(stutters) > misc['sig_stutter_min_hits']:
            add("Micro-Stuttering Detected", "WARNING",
                "Frequent frametime spikes. Cap FPS or enable G-Sync/FreeSync.",
                [f"Worst Spike: {stutters.max():.1f}ms", f"Events: {len(stutters)}"])

    # 16 VRAM swapping
    if gpu_mem_ded and gpu_mem_dyn:
        vram_full = df[gpu_mem_ded] > df[gpu_mem_ded].max() * 0.95
        if (vram_full & (df[gpu_mem_dyn] > 512)).any():
            add("VRAM Swapping", "WARNING",
                "GPU using slow system RAM. Reduce texture quality.",
                [f"Dynamic: {df[gpu_mem_dyn].max():.0f} MB"])

    # 17 PCIe errors
    if pcie_errors and df[pcie_errors].sum() > 0:
        add("PCIe Bus Signal Instability", "CRITICAL",
            "Hardware-level PCIe errors. Reseat GPU or replace riser cable.",
            [f"Total Errors: {df[pcie_errors].sum()}"])

    # 18 VRAM junction temp
    if vram_j_temp:
        t = df[vram_j_temp].max()
        if t > 102:
            add("VRAM Thermal Throttling",
                "CRITICAL" if t > 106 else "WARNING",
                f"GDDR6X VRAM at {t:.1f}°C. Check GPU backplate airflow.",
                [f"Max VRAM Temp: {t:.1f}°C"])

    # 19 PSU +5V / +3.3V
    for r_name, lo, hi in [('+5V', misc['sig_v5_lo'], misc['sig_v5_hi']),
                            ('+3.3V', misc['sig_v33_lo'], misc['sig_v33_hi'])]:
        col = col_find(df, r_name)
        if col and (df[col].min() < lo or df[col].max() > hi):
            add(f"PSU {r_name} Rail Unstable", "WARNING",
                f"Rail {r_name} out of spec. May cause USB disconnects or drive errors.",
                [f"Range: {df[col].min():.2f}V – {df[col].max():.2f}V",
                 f"Spec: {lo}V – {hi}V"])

    # 20 GPU power limit oscillation
    if gpu_pwr_limit and gpu_clk_col:
        limit_active = df[gpu_pwr_limit].apply(lambda x: 1 if x == 'Yes' else 0)
        toggles = limit_active.diff().abs().sum()
        if toggles > 5:
            add("GPU Power Limit Oscillation", "WARNING",
                "GPU ping-ponging against power limit. Undervolt or increase power limit.",
                [f"Toggles: {toggles:.0f}"])

    # 21 Chipset / USB
    if chipset_t and (df[chipset_t] > 80).any():
        add("Chipset Thermal Throttling", "WARNING",
            "Chipset overheating. Ensure GPU isn't blocking chipset airflow.",
            [f"Max: {df[chipset_t].max():.1f}°C"])
    if usb_v_col and df[usb_v_col].min() < 4.75:
        add("USB Rail Voltage Sag", "CRITICAL",
            "USB 5V rail below limits. Causes peripheral disconnects.",
            [f"Min: {df[usb_v_col].min():.2f}V"])

    # 22 GPU power connector safety
    if gpu_12v_v and gpu_12v_w:
        mask = df[gpu_12v_w] > 300
        if mask.any():
            min_v = df.loc[mask, gpu_12v_v].min()
            if min_v < 11.7:
                add("GPU Power Connector Safety Risk", "CRITICAL" if min_v < 11.5 else "WARNING",
                    f"Connector voltage dropped to {min_v:.2f}V under load. Risk of melting/fire. "
                    "Reseat the 12VHPWR/12-pin cable.",
                    [f"Voltage Drop: {min_v:.2f}V", f"Load: {mx(gpu_12v_w):.1f}W"])

    # 23 Background process interference
    if cpu_usage_col and gpu_usage_col:
        os_j = (df[cpu_usage_col] > 70) & (df[gpu_usage_col] < 40)
        if os_j.rolling(misc['sig_cpu_bn_samples']).sum().max() >= misc['sig_cpu_bn_samples']:
            add("Background Process Interference", "INFO",
                "High CPU activity not driven by GPU. Background apps stealing cycles. "
                "Close unneeded apps.",
                [f"Avg CPU: {df.loc[os_j, cpu_usage_col].mean():.1f}%",
                 f"Avg GPU: {df.loc[os_j, gpu_usage_col].mean():.1f}%"])

    # 24 Storage congestion
    if disk_busy and (df[disk_busy] >= misc['sig_disk_busy_pct']).rolling(
            misc['sig_disk_busy_samples']).sum().max() >= misc['sig_disk_busy_samples']:
        add("Storage Congestion", "INFO",
            "Drive 100% busy. Check for Windows Update or Antivirus.",
            ["Persistent 100% disk usage detected."])

    hits.sort(key=lambda r: {'CRITICAL': 0, 'WARNING': 1, 'INFO': 2}.get(r['severity'], 3))
    return hits


# ──────────────────────────────────────────────────────────────────────────────
# X-axis helper
# ──────────────────────────────────────────────────────────────────────────────

def format_elapsed(seconds: float) -> str:
    try:
        s = int(seconds)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
    except Exception:
        return str(seconds)


def get_x_axis(use_time: bool):
    df = st.session_state.df
    ts = st.session_state.time_series
    if use_time and ts is not None and len(ts) == len(df):
        x_vals = ts.dt.total_seconds().values
        return x_vals, True
    return df.index.values, False


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def build_figure(sel, mode_multi, mode_delta, mode_time, mode_heatmap, mode_compare,
                 is_dark) -> plt.Figure:
    df = st.session_state.df
    ref_df = st.session_state.ref_df
    vr = st.session_state.volt_rails

    bg_color   = "#121212" if is_dark else "white"
    text_color = "white"   if is_dark else "black"
    grid_color = "#333333" if is_dark else "#e0e0e0"

    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor(bg_color)

    x_vals, use_time = get_x_axis(mode_time)

    def fmt_xticks(ax):
        if use_time:
            import matplotlib.ticker as ticker
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: format_elapsed(v)))
            ax.tick_params(axis='x', labelrotation=30)

    def draw_spec_zones(ax, col_name):
        u = col_name.upper()
        for rail, (low, high) in vr.items():
            if rail in u:
                ax.axhspan(low - 0.2, low, color='red', alpha=0.1)
                ax.axhspan(high, high + 0.2, color='red', alpha=0.1)
                ax.axhline(y=low, color='#ff4d4d', ls='--', lw=1, alpha=0.5)
                ax.axhline(y=high, color='#ff4d4d', ls='--', lw=1, alpha=0.5)
                break

    colors = plt.cm.tab10.colors

    # ── Heatmap ───────────────────────────────────────────────────────────────
    if mode_heatmap:
        if not sel:
            ax = fig.add_subplot(111)
            ax.set_facecolor("#1e1e1e" if is_dark else "#fdfdfd")
            ax.text(0.5, 0.5, "No Sensors Selected", ha='center', va='center', color='gray')
            return fig

        band_colors = [(0.00, '#1a7a3a'), (0.55, '#2ecc71'), (0.60, '#f1c40f'),
                       (0.80, '#e67e22'), (0.85, '#922b21'), (1.00, '#641e16')]
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'threshold_map', [(p, c) for p, c in band_colors], N=512)

        matrix, labels = [], []
        for col in sel:
            data = df[col].ffill().fillna(0).values.astype(float)
            mn, mx2 = data.min(), data.max()
            norm = np.clip((data - mn) / (mx2 - mn + 1e-9) * 0.85, 0, 0.85)
            matrix.append(norm)
            short = col
            for br in ['[°C]', '[%]', '[MHz]', '[W]', '[V]', '[RPM]', '[ms]', '[FPS]']:
                short = short.replace(br, '').strip()
            labels.append(short[:45])

        matrix = np.array(matrix)
        extent = [x_vals[0], x_vals[-1], len(sel) - 0.5, -0.5]
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg_color)
        ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                  extent=extent, interpolation='nearest', origin='upper')
        for i in range(1, len(sel)):
            ax.axhline(i - 0.5, color=grid_color, lw=1.0)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, color=text_color, fontsize=7)
        ax.tick_params(axis='y', length=0, colors=text_color)
        ax.tick_params(axis='x', colors=text_color, labelsize=8)
        fmt_xticks(ax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_ticks([0.0, 0.30, 0.60, 0.85, 1.0])
        cbar.set_ticklabels(['Safe', 'Normal', 'Warning', 'At Limit', 'Critical'])
        cbar.ax.yaxis.set_tick_params(color=text_color, labelsize=7)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_color(text_color)
        ax.set_title("Sensor Heatmap — Green: safe | Yellow: warning | Red: at/above limit",
                     color=text_color, fontsize=8)
        fig.tight_layout()
        return fig

    # ── No selection ─────────────────────────────────────────────────────────
    if not sel:
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1e1e1e" if is_dark else "#fdfdfd")
        ax.text(0.5, 0.5, "No Sensors Selected", ha='center', va='center', color='gray')
        return fig

    # ── Multi mode ────────────────────────────────────────────────────────────
    if mode_multi:
        groups = {}
        for col in sel:
            cat = get_category(col)
            groups.setdefault(cat, []).append(col)
        active_cats = [c for c in sorted_categories(groups) if c in groups]
        axes = []
        color_idx = 0
        for i, cat_name in enumerate(active_cats):
            ax = fig.add_subplot(len(active_cats), 1, i + 1,
                                 sharex=axes[0] if axes else None)
            axes.append(ax)
            ax.set_facecolor("#1e1e1e" if is_dark else "#fdfdfd")
            ax.set_ylabel(cat_name, color=text_color, fontsize=7, fontweight='bold')
            for col in groups[cat_name]:
                c = colors[color_idx % len(colors)]
                if mode_compare and ref_df is not None and col in ref_df.columns:
                    ax.plot(ref_df.index.values, ref_df[col], ls='--', lw=1, alpha=0.35, color=c)
                s = df[col].dropna()
                lbl = f"{col}\nMin:{s.min():.1f} Avg:{s.mean():.1f} Max:{s.max():.1f}"
                ax.plot(x_vals, df[col], label=lbl, lw=1.5, color=c)
                draw_spec_zones(ax, col)
                color_idx += 1
            ax.grid(True, linestyle=':', alpha=0.4, color=grid_color)
            ax.tick_params(colors=text_color, labelsize=7)
            fmt_xticks(ax)
            leg = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                            fontsize='xx-small', frameon=False)
            if leg:
                for t in leg.get_texts(): t.set_color(text_color)
        for a in axes[:-1]:
            plt.setp(a.get_xticklabels(), visible=False)
        fig.subplots_adjust(right=0.78, hspace=0.3)
        return fig

    # ── Delta mode ────────────────────────────────────────────────────────────
    if mode_delta and len(sel) >= 2:
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1e1e1e" if is_dark else "#fdfdfd")
        s1, s2 = df[sel[0]], df[sel[1]]
        delta = (s1 - s2).abs()
        ax.plot(x_vals, s1, label=sel[0], alpha=0.4, ls='--')
        ax.plot(x_vals, s2, label=sel[1], alpha=0.4, ls='--')
        ax.plot(x_vals, delta,
                label=f"Δ ({sel[0]} – {sel[1]})\nMin:{delta.min():.1f} Avg:{delta.mean():.1f} Max:{delta.max():.1f}",
                color="#ffcc00", lw=2)
        ax.grid(True, linestyle=':', alpha=0.4, color=grid_color)
        ax.tick_params(colors=text_color, labelsize=8)
        fmt_xticks(ax)
        leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                        fontsize='x-small', frameon=False)
        if leg:
            for t in leg.get_texts(): t.set_color(text_color)
        fig.subplots_adjust(right=0.80)
        fig.tight_layout()
        return fig

    # ── Standard overlay ──────────────────────────────────────────────────────
    ax = fig.add_subplot(111)
    ax.set_facecolor("#1e1e1e" if is_dark else "#fdfdfd")
    for i, col in enumerate(sel):
        c = colors[i % len(colors)]
        if mode_compare and ref_df is not None and col in ref_df.columns:
            ax.plot(ref_df.index.values, ref_df[col], ls='--', lw=1, alpha=0.35, color=c)
        s = df[col].dropna()
        lbl = f"{col}\nMin:{s.min():.1f} Avg:{s.mean():.1f} Max:{s.max():.1f}"
        ax.plot(x_vals, df[col], label=lbl, lw=1.5, color=c)
        draw_spec_zones(ax, col)
    ax.grid(True, linestyle=':', alpha=0.4, color=grid_color)
    ax.tick_params(colors=text_color, labelsize=8)
    fmt_xticks(ax)
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                    fontsize='x-small', frameon=False)
    if leg:
        for t in leg.get_texts(): t.set_color(text_color)
    fig.subplots_adjust(right=0.80)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Detection limits editor (rendered inline in an expander)
# ──────────────────────────────────────────────────────────────────────────────

def render_limits_editor():
    tl   = st.session_state.temp_limits
    vr   = st.session_state.volt_rails
    misc = st.session_state.misc

    st.markdown("#### Temperature Limits (°C)")
    temp_display = [
        ("GPU Core", "GPU"), ("GPU Hotspot", "HOTSPOT"), ("GPU VRAM", "VRAM"),
        ("GPU VRM", "VRM"), ("CPU Core", "CORE"), ("CPU Tdie/Tctl", "TDIE"),
        ("CPU CCD/CCX", "CCD"), ("CPU Socket", "SOCKET"), ("Coolant/Liquid", "COOLANT"),
        ("SSD/NVMe", "SSD"), ("HDD", "HDD"), ("Chipset/PCH", "CHIPSET"),
    ]
    cols = st.columns(3)
    for idx, (label, key) in enumerate(temp_display):
        if key in tl:
            new_val = cols[idx % 3].number_input(label, value=float(tl[key]),
                                                  key=f"tl_{key}", step=1.0)
            tl[key] = new_val
            # sync aliases
            if key == 'HOTSPOT': tl['HOT SPOT'] = new_val
            if key == 'TDIE':    tl['TCTL'] = new_val
            if key == 'CCD':     tl['CCX'] = new_val
            if key == 'COOLANT': tl['LIQUID'] = tl['WATER'] = new_val
            if key == 'SSD':     tl['NVME'] = new_val
            if key == 'CHIPSET': tl['PCH'] = new_val

    st.markdown("#### Voltage Rails (V)")
    for rail, (lo, hi) in list(vr.items()):
        c1, c2 = st.columns(2)
        new_lo = c1.number_input(f"{rail} Low", value=float(lo), key=f"vr_{rail}_lo", step=0.01, format="%.2f")
        new_hi = c2.number_input(f"{rail} High", value=float(hi), key=f"vr_{rail}_hi", step=0.01, format="%.2f")
        vr[rail] = (new_lo, new_hi)

    st.markdown("#### Power / Latency")
    c1, c2, c3 = st.columns(3)
    misc['cpu_power_max']    = c1.number_input("CPU Power Max (W)",    value=float(misc['cpu_power_max']),    key="m_cpu_pwr")
    misc['gpu_power_max']    = c2.number_input("GPU Power Max (W)",    value=float(misc['gpu_power_max']),    key="m_gpu_pwr")
    misc['total_power_max']  = c3.number_input("Total Power Max (W)",  value=float(misc['total_power_max']),  key="m_tot_pwr")
    c1, c2, c3 = st.columns(3)
    misc['frametime_max_ms'] = c1.number_input("Frametime Max (ms)",  value=float(misc['frametime_max_ms']),  key="m_ft")
    misc['fps_min']          = c2.number_input("Min FPS (0.1% Low)",  value=float(misc['fps_min']),           key="m_fps")
    misc['latency_max_ms']   = c3.number_input("Latency Max (ms)",    value=float(misc['latency_max_ms']),    key="m_lat")

    st.markdown("#### Misc")
    c1, c2, c3 = st.columns(3)
    misc['fan_min_rpm']       = c1.number_input("Fan Stall RPM",      value=float(misc['fan_min_rpm']),       key="m_fan")
    misc['memory_load_max']   = c2.number_input("RAM/VRAM Load Max%", value=float(misc['memory_load_max']),   key="m_mem")
    misc['throttle_threshold']= c3.number_input("Throttle Sensitivity",value=float(misc['throttle_threshold']),key="m_thr",step=0.05,format="%.2f")

    # Auto-save every time this function runs (number_inputs already mutated session state)
    save_config()

    if st.button("Reset to Defaults", key="reset_limits"):
        st.session_state.temp_limits = dict(DEFAULT_TEMP_LIMITS)
        st.session_state.volt_rails  = dict(DEFAULT_VOLT_RAILS)
        st.session_state.misc        = dict(DEFAULT_MISC)
        save_config()
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="HD2 Log Viewer", layout="wide",
                       page_icon="📊")
    init_session()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title(f"📊 HD2 Log Viewer  v{CURRENT_VERSION}")
    st.caption("Hardware telemetry analyzer — HWiNFO / PresentMon / AIDA64 CSV")

    # ── File upload ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload CSV log", type=["csv"], label_visibility="collapsed")
    if uploaded:
        if uploaded.name != st.session_state.filename:
            with st.spinner("Loading CSV…"):
                try:
                    df, tc, ts = load_csv(uploaded)
                    st.session_state.df        = df
                    st.session_state.time_col  = tc
                    st.session_state.time_series = ts
                    st.session_state.filename  = uploaded.name
                    st.session_state.selected_cols = []
                    st.session_state.ref_df    = None
                except Exception as e:
                    st.error(f"Failed to load CSV: {e}")
                    return

    if st.session_state.df is None:
        st.info("Upload a CSV log file to get started.")
        return

    df = st.session_state.df

    # ──────────────────────────────────────────────────────────────────────────
    # Sidebar — controls
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"**File:** `{st.session_state.filename}`")
        st.markdown(f"**Rows:** {len(df):,}  |  **Columns:** {len(df.columns):,}")
        has_time = bool(st.session_state.time_col)
        if has_time:
            st.success(f"⏱ Time column: `{st.session_state.time_col}`")

        st.divider()
        st.subheader("View Settings")
        is_dark     = st.toggle("🌙 Dark Theme",   value=True)
        mode_multi  = st.toggle("📊 Multi-axis",   value=False)
        mode_delta  = st.toggle("Δ Delta Mode",    value=False,
                                disabled=mode_multi,
                                help="Compare two selected sensors. Disable Multi first.")
        mode_time   = st.toggle("🕒 Time X-axis",  value=False, disabled=not has_time)
        mode_heatmap= st.toggle("🌡 Heatmap",      value=False)
        mode_compare= st.toggle("🔍 Compare Ref",  value=False,
                                disabled=(st.session_state.ref_df is None))

        if st.button("📌 Set Reference", help="Snapshot current selection as comparison baseline"):
            st.session_state.ref_df = df.copy()
            st.success("Reference saved!")

        st.divider()
        st.subheader("Sensor Presets")
        # Save current selection as preset
        preset_name = st.text_input("Preset name", placeholder="e.g. GPU temps",
                                    label_visibility="collapsed")
        if st.button("💾 Save Preset") and preset_name:
            st.session_state.custom_groups[preset_name] = list(st.session_state.selected_cols)
            save_config()
            st.session_state._last_loaded_preset = None
            st.success(f"Saved '{preset_name}'")

        if st.session_state.custom_groups:
            preset_options = ["— select —"] + sorted(st.session_state.custom_groups.keys())
            chosen_preset = st.selectbox("Load preset",
                options=preset_options,
                key="chosen_preset",
                label_visibility="collapsed")
            if (chosen_preset != "— select —"
                    and chosen_preset != st.session_state.get("_last_loaded_preset")):
                st.session_state._last_loaded_preset = chosen_preset
                valid = [c for c in st.session_state.custom_groups[chosen_preset]
                         if c in df.columns]
                st.session_state.selected_cols = valid
                st.rerun()

            # Import / export via JSON text area
            with st.expander("Import / Export preset JSON"):
                export_data = json.dumps({
                    "name": "export",
                    "sensors": st.session_state.selected_cols
                }, indent=2)
                st.code(export_data, language="json")
                paste = st.text_area("Paste preset JSON to import", height=80)
                if st.button("Import") and paste:
                    try:
                        data = json.loads(paste)
                        nm = data.get("name", "Imported")
                        st.session_state.custom_groups[nm] = data.get("sensors", [])
                        save_config()
                        st.session_state._last_loaded_preset = None
                        st.success(f"Imported '{nm}'")
                        st.rerun()
                    except Exception:
                        st.error("Invalid JSON")

            # Delete presets
            del_preset = st.selectbox("Delete preset",
                options=["— select —"] + sorted(st.session_state.custom_groups.keys()),
                key="del_preset_select", label_visibility="collapsed")
            if st.button("🗑 Delete Preset") and del_preset != "— select —":
                del st.session_state.custom_groups[del_preset]
                save_config()
                st.session_state._last_loaded_preset = None
                st.rerun()

        st.divider()
        if st.button("⬇ Export PNG", help="Save the current chart"):
            fig = build_figure(st.session_state.selected_cols, mode_multi, mode_delta,
                               mode_time, mode_heatmap, mode_compare, is_dark)
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            st.download_button("📥 Download PNG", buf.getvalue(),
                               file_name="hd2_log_chart.png", mime="image/png")

    # ──────────────────────────────────────────────────────────────────────────
    # Main area layout: sensor list | chart
    # ──────────────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 3], gap="medium")

    with col_left:
        st.subheader("Sensor Selection")

        # Detect issues button
        show_issues_only = st.toggle("🚨 Show Out-of-Spec Only", value=False)

        search = st.text_input("🔍 Search sensors", placeholder="Type to filter…",
                               label_visibility="collapsed")

        groups = group_columns(df)
        cats   = sorted_categories(groups)

        # Build issue set once
        if show_issues_only:
            issue_cols = {col for col in df.columns if is_critical(df, col)}
        else:
            issue_cols = set()

        new_selection = list(st.session_state.selected_cols)

        for cat in cats:
            cols_in_cat = groups.get(cat, [])
            # Apply search filter
            q = search.upper().replace(" ", "")
            if q:
                cols_in_cat = [c for c in cols_in_cat if q in c.upper().replace(" ", "")]
            # Apply issues filter
            if show_issues_only:
                cols_in_cat = [c for c in cols_in_cat if c in issue_cols]
            if not cols_in_cat:
                continue

            with st.expander(f"**{cat.upper()}** ({len(cols_in_cat)})", expanded=False):
                for col in sorted(cols_in_cat):
                    is_alert = col in issue_cols if show_issues_only else is_critical(df, col)
                    label = f"⚠️ {col}" if is_alert else col
                    checked = col in new_selection
                    if st.checkbox(label, value=checked, key=f"cb_{col}"):
                        if col not in new_selection:
                            new_selection.append(col)
                    else:
                        if col in new_selection:
                            new_selection.remove(col)

        st.session_state.selected_cols = new_selection

        if st.button("Clear All"):
            st.session_state.selected_cols = []
            st.rerun()

        st.markdown(f"*{len(new_selection)} sensor(s) selected*")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart
    # ──────────────────────────────────────────────────────────────────────────
    with col_right:
        sel = [c for c in st.session_state.selected_cols if c in df.columns]

        # Diagnosis & limits buttons
        diag_col, lim_col = st.columns(2)
        run_diag = diag_col.button("🔬 Diagnose Hardware Signatures", use_container_width=True)
        show_lim = lim_col.button("⚙ Edit Detection Limits", use_container_width=True)

        if run_diag:
            with st.spinner("Analyzing signatures…"):
                results = run_signatures(df)
            if not results:
                st.success("✅ No hardware failure signatures detected. Log looks clean.")
            else:
                for r in results:
                    sev = r['severity']
                    icon = "🔴" if sev == "CRITICAL" else ("🟡" if sev == "WARNING" else "🔵")
                    with st.expander(f"{icon} **{sev}** — {r['name']}", expanded=(sev == "CRITICAL")):
                        st.write(r['description'])
                        for ev in r.get('evidence', []):
                            st.markdown(f"- {ev}")

        if show_lim:
            with st.expander("Detection Limits Editor", expanded=True):
                render_limits_editor()

        # Stats table for selected sensors
        if sel:
            stats_rows = []
            for col in sel:
                s = df[col].dropna()
                alert = "⚠️" if is_critical(df, col) else ""
                stats_rows.append({
                    "Sensor": f"{alert} {col}",
                    "Min": f"{s.min():.2f}",
                    "Avg": f"{s.mean():.2f}",
                    "Max": f"{s.max():.2f}",
                    "P95": f"{s.quantile(0.95):.2f}",
                })
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

        # Main chart
        fig = build_figure(sel, mode_multi, mode_delta, mode_time,
                           mode_heatmap, mode_compare, is_dark)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Raw data table (optional)
        with st.expander("📋 Raw Data Table"):
            cols_to_show = ([st.session_state.time_col] if st.session_state.time_col else []) + sel
            cols_to_show = [c for c in cols_to_show if c in df.columns]
            if cols_to_show:
                st.dataframe(df[cols_to_show].head(500), use_container_width=True)
            else:
                st.info("Select sensors to view their raw data.")


if __name__ == "__main__":
    main()