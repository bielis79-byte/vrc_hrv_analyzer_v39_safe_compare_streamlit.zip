
import io
import re
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import pdist, squareform

try:
    import networkx as nx
except Exception:
    nx = None


st.set_page_config(page_title="VRC / HRV RRi Analyzer Pro v3.9 SAFE", layout="wide")

# ============================================================
# CONFIGURACIÓN
# ============================================================

PHASES = ["Basal"] + [f"E{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 4)]

PHASE_GROUP = {
    "Basal": "Basal",
    **{f"E{i}": "Ejercicio" for i in range(1, 7)},
    **{f"R{i}": "Recuperación" for i in range(1, 4)},
}

PHASE_COLORS = {
    "Basal": "rgba(0,150,255,0.20)",
    "Ejercicio": "rgba(255,140,0,0.18)",
    "Recuperación": "rgba(0,200,100,0.18)",
}

DOMAIN_GROUPS = {
    "Amplitud": ["SDNN", "SD2", "TOTAL"],
    "Vagal": ["RMSSD", "SD1", "HF", "pNN50"],
    "Complejidad": ["DFA_alpha1", "DFA_alpha2", "ApEn", "SampEn"],
    "Recurrencia": ["REC", "DET", "Lmean", "Lmax", "ShanEn"],
}

FS_INTERP = 4.0
LAMBDA_DEFAULT = 500


# ============================================================
# UTILIDADES
# ============================================================

def sanitize_name(name):
    name = Path(str(name)).stem
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")
    return name or "registro"


def read_rri_file(uploaded_file):
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    vals = []
    for line in text.replace(";", "\n").replace("\t", "\n").splitlines():
        line = line.strip().replace(",", ".")
        if not line:
            continue
        for p in line.split():
            try:
                vals.append(float(p))
            except Exception:
                pass

    rr = np.asarray(vals, dtype=float)
    rr = rr[np.isfinite(rr)]
    if len(rr) == 0:
        raise ValueError("No se han detectado RRi numéricos.")

    if np.nanmedian(rr) > 10:
        rr = rr / 1000.0

    rr = rr[(rr >= 0.3) & (rr <= 2.0)]
    if len(rr) == 0:
        raise ValueError("Tras el filtrado fisiológico no quedan RRi válidos.")

    return rr


def cumulative_time(rr):
    return np.cumsum(rr)


def sec_to_hms(seconds):
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def hms_to_sec(s):
    parts = [float(p) for p in str(s).strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def cut_segment(rr, start_s, end_s):
    t = cumulative_time(rr)
    return rr[(t >= start_s) & (t <= end_s)]


def default_windows_from_duration(t_max):
    """
    Ventanas iniciales seguras.
    No limitan la duración: sólo son una propuesta inicial.
    El usuario puede escribir cualquier inicio/fin.
    """
    t_max = float(t_max)
    if t_max < 600:
        step = max(t_max / 10.0, 30.0)
        return {ph: [i * step, min((i + 1) * step, t_max)] for i, ph in enumerate(PHASES)}

    # Propuesta: basal primeros 5 min, ejercicio dividido en 6 bloques, recuperación en 3 bloques.
    b0 = min(300, t_max * 0.10)
    basal = [0.0, min(300.0, t_max)]
    remaining_start = basal[1]
    remaining = max(0.0, t_max - remaining_start)
    step = remaining / 9.0 if remaining > 0 else 60.0

    windows = {"Basal": basal}
    for i in range(1, 7):
        s = remaining_start + (i - 1) * step
        e = remaining_start + i * step
        windows[f"E{i}"] = [min(s, t_max), min(e, t_max)]
    for i in range(1, 4):
        idx = 6 + i
        s = remaining_start + (idx - 1) * step
        e = remaining_start + idx * step
        windows[f"R{i}"] = [min(s, t_max), min(e, t_max)]
    return windows


def smoothness_priors_detrend(y, lam=500):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y - np.mean(y) if n else y

    I = sparse.eye(n, format="csc")
    e = np.ones(n)
    D2 = sparse.diags([e[:-2], -2 * e[:-2], e[:-2]], [0, 1, 2], shape=(n - 2, n), format="csc")
    trend = spsolve(I + (lam ** 2) * (D2.T @ D2), y)
    return y - trend


def interpolate_rr(rr, fs=FS_INTERP, apply_lambda=False, lam=500):
    t = cumulative_time(rr)
    if len(t) < 5:
        return np.array([]), np.array([])

    t = t - t[0]
    x = rr.copy()
    keep = np.r_[True, np.diff(t) > 0]
    t = t[keep]
    x = x[keep]
    if len(t) < 5:
        return np.array([]), np.array([])

    ti = np.arange(0, t[-1], 1 / fs)
    if len(ti) < 5:
        return np.array([]), np.array([])

    xi = CubicSpline(t, x, bc_type="natural")(ti)
    if apply_lambda:
        xi = smoothness_priors_detrend(xi, lam)
    return ti, xi


# ============================================================
# MÉTRICAS
# ============================================================

def time_metrics(rr):
    rr_ms = rr * 1000.0
    diff = np.diff(rr_ms)
    mean_rr = np.mean(rr_ms)
    mean_hr = 60000.0 / mean_rr if mean_rr > 0 else np.nan
    sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
    rmssd = np.sqrt(np.mean(diff ** 2)) if len(diff) else np.nan
    nn50 = int(np.sum(np.abs(diff) > 50)) if len(diff) else 0
    pnn50 = 100.0 * nn50 / len(diff) if len(diff) else np.nan
    sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
    sd2 = np.sqrt(max(0.0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan

    return {
        "N_RRi": len(rr),
        "Duration_s": float(np.sum(rr)),
        "MeanRR": mean_rr,
        "MeanHR": mean_hr,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "NN50": nn50,
        "pNN50": pnn50,
        "SD1": sd1,
        "SD2": sd2,
    }


def psd_metrics(rr):
    ti, xi = interpolate_rr(rr, fs=FS_INTERP, apply_lambda=True, lam=LAMBDA_DEFAULT)
    if len(xi) < 32:
        return {"VLF": np.nan, "LF": np.nan, "HF": np.nan, "TOTAL": np.nan, "LF_HF": np.nan}

    xi_ms = xi * 1000.0
    xi_ms = xi_ms - np.mean(xi_ms)
    nperseg = min(int(256 * FS_INTERP), len(xi_ms))
    noverlap = int(0.5 * nperseg)

    f, pxx = signal.welch(
        xi_ms,
        fs=FS_INTERP,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )

    def bp(lo, hi):
        mask = (f >= lo) & (f < hi)
        return np.trapezoid(pxx[mask], f[mask]) if np.any(mask) else np.nan

    vlf = bp(0.0033, 0.04)
    lf = bp(0.04, 0.15)
    hf = bp(0.15, 0.40)
    total = np.nansum([vlf, lf, hf])
    return {"VLF": vlf, "LF": lf, "HF": hf, "TOTAL": total, "LF_HF": lf / hf if hf and hf > 0 else np.nan}


def _phi_apen(x, m, r):
    n = len(x)
    if n <= m + 1:
        return np.nan
    pats = np.array([x[i:i + m] for i in range(n - m + 1)])
    vals = []
    for p in pats:
        dist = np.max(np.abs(pats - p), axis=1)
        c = np.mean(dist <= r)
        if c > 0:
            vals.append(np.log(c))
    return np.mean(vals) if vals else np.nan


def apen_calc(x, m=2, r_ratio=0.2):
    x = smoothness_priors_detrend(np.asarray(x, dtype=float), LAMBDA_DEFAULT)
    r = r_ratio * np.std(x, ddof=1)
    if not np.isfinite(r) or r == 0:
        return np.nan
    return _phi_apen(x, m, r) - _phi_apen(x, m + 1, r)


def sampen_calc(x, m=2, r_ratio=0.2):
    x = smoothness_priors_detrend(np.asarray(x, dtype=float), LAMBDA_DEFAULT)
    n = len(x)
    if n <= m + 2:
        return np.nan
    r = r_ratio * np.std(x, ddof=1)
    if not np.isfinite(r) or r == 0:
        return np.nan

    def count(mm):
        pats = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        c = 0
        for i in range(len(pats) - 1):
            dist = np.max(np.abs(pats[i + 1:] - pats[i]), axis=1)
            c += np.sum(dist <= r)
        return c

    b = count(m)
    a = count(m + 1)
    if a == 0 or b == 0:
        return np.nan
    return -np.log(a / b)


def dfa_calc(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 50:
        return np.nan, np.nan
    y = np.cumsum(x - np.mean(x))
    scales = np.unique(np.floor(np.logspace(np.log10(4), np.log10(max(5, n // 4)), 18)).astype(int))
    ss, ff = [], []
    for s in scales:
        if s < 4 or n // s < 2:
            continue
        rms = []
        for i in range(n // s):
            seg = y[i * s:(i + 1) * s]
            t = np.arange(s)
            co = np.polyfit(t, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(co, t)) ** 2)))
        val = np.sqrt(np.mean(np.asarray(rms) ** 2))
        if val > 0:
            ss.append(s)
            ff.append(val)
    ss, ff = np.asarray(ss), np.asarray(ff)
    if len(ss) < 4:
        return np.nan, np.nan
    m1 = (ss >= 4) & (ss <= 16)
    m2 = ss > 16
    a1 = np.polyfit(np.log(ss[m1]), np.log(ff[m1]), 1)[0] if np.sum(m1) >= 2 else np.nan
    a2 = np.polyfit(np.log(ss[m2]), np.log(ff[m2]), 1)[0] if np.sum(m2) >= 2 else np.nan
    return a1, a2


def rqa_calc(x, emb_dim=10, tau=1, l_min=2, max_n=600):
    x = np.asarray(x, dtype=float)
    if len(x) > max_n:
        idx = np.linspace(0, len(x) - 1, max_n).astype(int)
        x = x[idx]

    n = len(x) - (emb_dim - 1) * tau
    if n < 20:
        return {"REC": np.nan, "DET": np.nan, "Lmean": np.nan, "Lmax": np.nan, "ShanEn": np.nan}

    X = np.array([x[i:i + emb_dim * tau:tau] for i in range(n)])
    D = squareform(pdist(X))
    radius = np.sqrt(emb_dim) * np.std(x, ddof=1)
    R = (D <= radius).astype(int)
    np.fill_diagonal(R, 0)
    rec = 100 * R.sum() / (n * n - n)

    lens = []
    for k in range(-n + 1, n):
        diag = np.diag(R, k=k)
        c = 0
        for val in diag:
            if val:
                c += 1
            else:
                if c >= l_min:
                    lens.append(c)
                c = 0
        if c >= l_min:
            lens.append(c)

    if not lens:
        return {"REC": rec, "DET": 0, "Lmean": 0, "Lmax": 0, "ShanEn": 0}

    lens = np.asarray(lens)
    det = 100 * lens.sum() / R.sum() if R.sum() > 0 else 0
    vals, counts = np.unique(lens, return_counts=True)
    p = counts / counts.sum()
    return {"REC": rec, "DET": det, "Lmean": np.mean(lens), "Lmax": np.max(lens), "ShanEn": -np.sum(p * np.log(p))}


def hvg_metrics(rr, max_nodes=700):
    if nx is None:
        return {}
    x = np.asarray(rr, dtype=float)
    if len(x) > max_nodes:
        idx = np.linspace(0, len(x) - 1, max_nodes).astype(int)
        x = x[idx]

    n = len(x)
    if n < 20:
        return {
            "HVG_nodes": n,
            "HVG_edges": np.nan,
            "HVG_degree_mean": np.nan,
            "HVG_degree_max": np.nan,
            "HVG_hubs_p90": np.nan,
            "HVG_clustering": np.nan,
            "HVG_lambda": np.nan,
        }

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
        for j in range(i + 2, n):
            if np.max(x[i + 1:j]) < min(x[i], x[j]):
                G.add_edge(i, j)

    deg = np.array([d for _, d in G.degree()])
    vals, counts = np.unique(deg, return_counts=True)
    p = counts / counts.sum()
    mask = (vals > 1) & (p > 0)
    lam = -np.polyfit(vals[mask], np.log(p[mask]), 1)[0] if np.sum(mask) >= 2 else np.nan

    out = {
        "HVG_nodes": n,
        "HVG_edges": G.number_of_edges(),
        "HVG_degree_mean": 2 * G.number_of_edges() / n,
        "HVG_degree_max": np.max(deg),
        "HVG_hubs_p90": int(np.sum(deg >= np.percentile(deg, 90))),
        "HVG_clustering": nx.average_clustering(G),
        "HVG_lambda": lam,
    }

    if nx.is_connected(G):
        out["HVG_path_length"] = nx.average_shortest_path_length(G)
        out["HVG_diameter"] = nx.diameter(G)
    else:
        out["HVG_path_length"] = np.nan
        out["HVG_diameter"] = np.nan

    return out


def calculate_all(rr, include_rqa=True, include_hvg=False):
    rr_ms = rr * 1000.0
    out = {}
    out.update(time_metrics(rr))
    out.update(psd_metrics(rr))
    a1, a2 = dfa_calc(rr_ms)
    out["DFA_alpha1"] = a1
    out["DFA_alpha2"] = a2
    out["ApEn"] = apen_calc(rr_ms)
    out["SampEn"] = sampen_calc(rr_ms)
    if include_rqa:
        out.update(rqa_calc(rr_ms))
    if include_hvg:
        out.update(hvg_metrics(rr))
    return out


def calculate_record(rr, windows, min_rr, include_rqa=True, include_hvg=False):
    rows = []
    segments = {}
    valid = {}
    for ph in PHASES:
        s, e = windows[ph]
        seg = cut_segment(rr, s, e)
        segments[ph] = seg
        valid[ph] = len(seg) >= min_rr
        if valid[ph]:
            res = calculate_all(seg, include_rqa=include_rqa, include_hvg=include_hvg)
            res["Fase"] = ph
            rows.append(res)
    df = pd.DataFrame(rows)
    return pd.DataFrame(rows).set_index("Fase") if rows else pd.DataFrame(), segments, valid


def domain_values(metrics_df, method="median"):
    if metrics_df.empty or "Basal" not in metrics_df.index:
        return pd.DataFrame()
    out = {}
    base = metrics_df.loc["Basal"]
    for dom, vars_ in DOMAIN_GROUPS.items():
        vals_phase = []
        for ph in metrics_df.index:
            vals = []
            for v in vars_:
                if v not in metrics_df.columns:
                    continue
                b = base[v]
                x = metrics_df.loc[ph, v]
                if pd.notna(b) and pd.notna(x) and b != 0:
                    vals.append(100 * x / b)
            vals_phase.append(np.nanmedian(vals) if vals and method == "median" else (np.nanmean(vals) if vals else np.nan))
        out[dom] = vals_phase
    return pd.DataFrame(out, index=metrics_df.index)


def plot_rr(record_data, windows, mode, selected_record):
    fig = go.Figure()
    names = [selected_record] if mode == "Registro principal" else list(record_data.keys())

    for name in names:
        rr = record_data[name]["rr"]
        t = cumulative_time(rr) / 60.0
        fig.add_trace(go.Scatter(x=t, y=rr * 1000, mode="lines", name=name))

    if mode == "Registro principal":
        for ph, (s, e) in windows.items():
            group = PHASE_GROUP.get(ph, ph)
            fig.add_vrect(
                x0=s / 60.0,
                x1=e / 60.0,
                fillcolor=PHASE_COLORS.get(group, "rgba(180,180,180,0.15)"),
                line_width=0,
                annotation_text=ph,
                annotation_position="top left",
            )

    fig.update_layout(height=480, xaxis_title="Tiempo acumulado (min)", yaxis_title="RRi (ms)", hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def plot_compare(pivot, variable):
    fig = go.Figure()
    for rec in pivot.columns:
        fig.add_trace(go.Scatter(x=pivot.index, y=pivot[rec], mode="lines+markers", name=str(rec)))
    fig.update_layout(height=500, title=f"Comparativa: {variable}", xaxis_title="Fase", yaxis_title=variable)
    return fig


def phase_overlay(record_data, windows, phase):
    fig = go.Figure()
    s, e = windows[phase]
    for name, data in record_data.items():
        seg = cut_segment(data["rr"], s, e)
        if len(seg) < 3:
            continue
        t = cumulative_time(seg)
        t = t - t[0]
        fig.add_trace(go.Scatter(x=t / 60, y=seg * 1000, mode="lines", name=name))
    fig.update_layout(height=450, title=f"RRi superpuesto en {phase}", xaxis_title="Tiempo dentro de la fase (min)", yaxis_title="RRi (ms)")
    return fig


# ============================================================
# APP
# ============================================================

st.title("VRC / HRV RRi Analyzer Pro v3.9 SAFE")
st.caption("Versión estable: varios registros, ventanas libres, comparación por fases, cálculo seguro y exportación.")

with st.sidebar:
    uploaded_files = st.file_uploader("Sube uno o varios CSV/TXT con RRi", type=["csv", "txt"], accept_multiple_files=True)
    min_rr = st.number_input("Mínimo de RRi por ventana", min_value=10, max_value=300, value=30, step=5)
    include_rqa = st.checkbox("Calcular RQA", value=True)
    include_hvg = st.checkbox("Calcular HVG/grafos", value=False, help="Más lento. Actívalo sólo cuando ya tengas las ventanas ajustadas.")
    domain_method = st.selectbox("Dominios", ["median", "mean"], index=0)

if not uploaded_files:
    st.info("Sube uno o varios archivos RRi para empezar.")
    st.stop()

record_data = {}
errors = []
for uf in uploaded_files:
    try:
        rr = read_rri_file(uf)
        name = sanitize_name(uf.name)
        base = name
        k = 2
        while name in record_data:
            name = f"{base}_{k}"
            k += 1
        record_data[name] = {"filename": uf.name, "rr": rr, "duration": float(np.sum(rr))}
    except Exception as e:
        errors.append(f"{uf.name}: {e}")

if errors:
    st.error("Errores leyendo archivos:\n" + "\n".join(errors))
if not record_data:
    st.stop()

record_names = list(record_data.keys())
selected_record = st.sidebar.selectbox("Registro principal", record_names)
t_max = record_data[selected_record]["duration"]

# Reiniciar ventanas cuando cambia el registro principal o al pulsar botón.
if "selected_record_prev" not in st.session_state or st.session_state.selected_record_prev != selected_record:
    st.session_state.selected_record_prev = selected_record
    st.session_state.windows = default_windows_from_duration(t_max)

if "windows" not in st.session_state:
    st.session_state.windows = default_windows_from_duration(t_max)

if st.sidebar.button("Reiniciar ventanas para este registro"):
    st.session_state.windows = default_windows_from_duration(t_max)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1) Segmentación",
    "2) Cálculo HRV",
    "3) Comparar registros",
    "4) Gráficas",
    "5) Exportar",
])

# Cálculo central seguro
records_results = {}
records_segments = {}
records_valid = {}
with st.spinner("Calculando ventanas válidas..."):
    for name, data in record_data.items():
        df, segs, valid = calculate_record(
            data["rr"],
            st.session_state.windows,
            min_rr=min_rr,
            include_rqa=include_rqa,
            include_hvg=include_hvg,
        )
        records_results[name] = df
        records_segments[name] = segs
        records_valid[name] = valid

metrics_df = records_results[selected_record]
segments = records_segments[selected_record]
valid_segments = records_valid[selected_record]

with tab1:
    st.subheader("Segmentación")
    st.write("Las ventanas son libres: puedes escribir cualquier inicio y fin en HH:MM:SS. No están limitadas a 5 minutos.")

    view_mode = st.radio("Visualización RRi", ["Registro principal", "Todos superpuestos"], horizontal=True)
    st.plotly_chart(plot_rr(record_data, st.session_state.windows, view_mode, selected_record), use_container_width=True)

    st.markdown("### Editar ventanas manualmente")
    cols = st.columns(5)
    edited = {}
    for idx, ph in enumerate(PHASES):
        with cols[idx % 5]:
            st.markdown(f"**{ph}**")
            s0, e0 = st.session_state.windows[ph]
            ini = st.text_input(f"{ph} inicio", value=sec_to_hms(s0), key=f"{ph}_ini_v39")
            fin = st.text_input(f"{ph} fin", value=sec_to_hms(e0), key=f"{ph}_fin_v39")
            edited[ph] = (ini, fin)

    if st.button("Aplicar ventanas"):
        ok = True
        new_w = {}
        for ph, (ini, fin) in edited.items():
            try:
                s = hms_to_sec(ini)
                e = hms_to_sec(fin)
                if e <= s:
                    st.warning(f"{ph}: el final debe ser mayor que el inicio.")
                    ok = False
                new_w[ph] = [s, e]
            except Exception:
                st.warning(f"{ph}: formato no válido.")
                ok = False
        if ok:
            st.session_state.windows = new_w
            st.success("Ventanas actualizadas. La app recalculará automáticamente.")
            st.rerun()

    win_table = pd.DataFrame([
        {
            "Fase": ph,
            "Inicio": sec_to_hms(st.session_state.windows[ph][0]),
            "Fin": sec_to_hms(st.session_state.windows[ph][1]),
            "Inicio_min": round(st.session_state.windows[ph][0] / 60, 2),
            "Fin_min": round(st.session_state.windows[ph][1] / 60, 2),
            "Duración_min": round((st.session_state.windows[ph][1] - st.session_state.windows[ph][0]) / 60, 2),
            "Válida_principal": valid_segments.get(ph, False),
            "N_RRi_principal": len(segments.get(ph, [])),
        }
        for ph in PHASES
    ])
    st.dataframe(win_table, use_container_width=True)

with tab2:
    st.subheader(f"Cálculo HRV: {selected_record}")
    if metrics_df.empty:
        st.info("No hay ventanas válidas para el registro principal.")
    else:
        st.dataframe(metrics_df, use_container_width=True)

with tab3:
    st.subheader("Comparar registros")
    if len(record_data) < 2:
        st.info("Sube dos o más registros para comparar.")
    else:
        valid_summary = pd.DataFrame(records_valid).T.reindex(columns=PHASES)
        st.markdown("### Ventanas válidas por registro")
        st.dataframe(valid_summary, use_container_width=True)

        long_rows = []
        for rec, df in records_results.items():
            if df.empty:
                continue
            tmp = df.copy()
            tmp.insert(0, "Registro", rec)
            tmp.insert(1, "Fase", tmp.index)
            long_rows.append(tmp.reset_index(drop=True))

        if not long_rows:
            st.info("No hay datos comparables.")
        else:
            long_df = pd.concat(long_rows, ignore_index=True)
            numeric_vars = [
                c for c in long_df.columns
                if c not in ["Registro", "Fase"] and pd.api.types.is_numeric_dtype(long_df[c])
            ]

            selected_phases = st.multiselect("Fases a comparar", PHASES, default=[p for p in PHASES if p in long_df["Fase"].unique()])
            variable = st.selectbox("Variable", numeric_vars, index=numeric_vars.index("RMSSD") if "RMSSD" in numeric_vars else 0)

            df_sel = long_df[long_df["Fase"].isin(selected_phases)] if selected_phases else long_df
            pivot = df_sel.pivot_table(index="Fase", columns="Registro", values=variable, aggfunc="first").reindex(selected_phases)

            st.markdown(f"### Tabla comparativa: {variable}")
            st.dataframe(pivot, use_container_width=True)
            st.plotly_chart(plot_compare(pivot, variable), use_container_width=True)

            phase_overlay_name = st.selectbox("RRi superpuesto en fase", selected_phases or PHASES)
            st.plotly_chart(phase_overlay(record_data, st.session_state.windows, phase_overlay_name), use_container_width=True)

            st.markdown("### Tabla larga")
            st.dataframe(df_sel, use_container_width=True)

with tab4:
    st.subheader("Gráficas")
    if metrics_df.empty:
        st.info("No hay datos para graficar.")
    else:
        variables = [c for c in metrics_df.columns if pd.api.types.is_numeric_dtype(metrics_df[c])]
        var = st.selectbox("Variable del registro principal", variables, index=variables.index("RMSSD") if "RMSSD" in variables else 0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics_df.index, y=metrics_df[var], mode="lines+markers", name=selected_record))
        fig.update_layout(height=450, title=f"{selected_record}: {var}", xaxis_title="Fase", yaxis_title=var)
        st.plotly_chart(fig, use_container_width=True)

        dom_df = domain_values(metrics_df, method=domain_method)
        if not dom_df.empty:
            st.markdown("### Dominios normalizados. Basal = 100%")
            st.dataframe(dom_df, use_container_width=True)
            fig2 = go.Figure()
            for col in dom_df.columns:
                fig2.add_trace(go.Scatter(x=dom_df.index, y=dom_df[col], mode="lines+markers", name=col))
            fig2.add_hline(y=100, line_dash="dash")
            fig2.update_layout(height=450, title="Dominios normalizados", xaxis_title="Fase", yaxis_title="% basal")
            st.plotly_chart(fig2, use_container_width=True)

with tab5:
    st.subheader("Exportar")
    long_rows = []
    for rec, df in records_results.items():
        if df.empty:
            continue
        tmp = df.copy()
        tmp.insert(0, "Registro", rec)
        tmp.insert(1, "Fase", tmp.index)
        long_rows.append(tmp.reset_index(drop=True))

    if not long_rows:
        st.info("No hay datos para exportar.")
    else:
        long_df = pd.concat(long_rows, ignore_index=True)
        valid_summary = pd.DataFrame(records_valid).T.reindex(columns=PHASES)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            p_xlsx = tmpdir / "resultados_hrv_comparativa.xlsx"
            p_csv = tmpdir / "resultados_hrv_comparativa.csv"
            p_zip = tmpdir / "resultados_hrv_comparativa.zip"

            long_df.to_csv(p_csv, index=False)
            with pd.ExcelWriter(p_xlsx) as writer:
                long_df.to_excel(writer, sheet_name="metricas", index=False)
                valid_summary.to_excel(writer, sheet_name="ventanas_validas")
                pd.DataFrame([
                    {
                        "Fase": ph,
                        "Inicio": sec_to_hms(st.session_state.windows[ph][0]),
                        "Fin": sec_to_hms(st.session_state.windows[ph][1]),
                        "Duracion_min": (st.session_state.windows[ph][1] - st.session_state.windows[ph][0]) / 60,
                    }
                    for ph in PHASES
                ]).to_excel(writer, sheet_name="ventanas", index=False)

            with zipfile.ZipFile(p_zip, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(p_xlsx, arcname=p_xlsx.name)
                z.write(p_csv, arcname=p_csv.name)

            st.download_button("Descargar ZIP", data=p_zip.read_bytes(), file_name="resultados_hrv_comparativa.zip", mime="application/zip")
            st.download_button("Descargar Excel", data=p_xlsx.read_bytes(), file_name="resultados_hrv_comparativa.xlsx")
