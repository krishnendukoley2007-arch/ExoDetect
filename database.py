"""
════════════════════════════════════════════════════════════════
 ExoDetect v9 — DATABASE MODULE (database.py)
════════════════════════════════════════════════════════════════
 Persistent SQLite storage for every analysis run + an
 interactive Plotly graph engine for the Database Explorer page.

 • Analyses survive app restarts (exodetect.db)
 • Catalog explorer over the ~2,000-star NASA v9 dataset
   (17 physics-informed features + stellar parameters)
 • Dark-theme animated Plotly figures: 3-D scatters, radar,
   parallel coordinates, HR diagram, play-button animations
 • One-click CSV / JSON exports

 Team OrbitX2026 | BAH2026 PS7 | Jadavpur University
════════════════════════════════════════════════════════════════
"""

import os
import json
import sqlite3
import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ════════════════════════════════════════════════════════════
# THEME — matches dashboard.py matplotlib palette exactly
# ════════════════════════════════════════════════════════════
ACCENT  = "#4a9eff"
ORANGE  = "#ff7a45"
GREEN   = "#22cc77"
RED     = "#ff4466"
PURPLE  = "#aa55ff"
YELLOW  = "#ffcc22"
CYAN    = "#00ccee"
BG      = "#0a1220"
CARD_BG = "#0c1422"
GRID_C  = "#1e3050"
TEXT_C  = "#ddeeff"
LABEL_C = "#7090b8"

LABEL_COLORS = {"planet": GREEN, "false_positive": RED}
LABEL_NAMES  = {"planet": "🪐 Planet", "false_positive": "⭐ False Positive"}
COLOR_MAP    = {LABEL_NAMES["planet"]: GREEN, LABEL_NAMES["false_positive"]: RED}

DB_PATH = "exodetect.db"

# The 11 light-curve features (v9) + friendly names for selectors
FEATURE_LABELS = {
    "depth":                    "Transit Depth",
    "snr":                      "Signal-to-Noise Ratio",
    "sec_ratio":                "Secondary Eclipse Ratio",
    "duration_hours":           "Transit Duration (h)",
    "bls_power":                "BLS Power",
    "odd_even_diff":            "Odd-Even Depth Diff",
    "transit_shape":            "Transit Shape (V vs U)",
    "dur_period_ratio":         "Duration/Period Ratio",
    "depth_consistency":        "Depth Consistency",
    "ingress_egress_asymmetry": "Ingress/Egress Asymmetry",
    "odd_even_ratio":           "Odd-Even Ratio",
}
STELLAR_LABELS = {
    "Teff": "Effective Temperature (K)", "rad": "Stellar Radius (R☉)",
    "mass": "Stellar Mass (M☉)", "logg": "Surface Gravity (log g)",
    "Tmag": "TESS Magnitude", "lum": "Luminosity (L☉)",
}


def style_fig(fig, title=None, height=450):
    """Apply the ExoDetect dark theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG,
        font=dict(family="Inter, Space Grotesk, sans-serif", color=LABEL_C, size=12),
        title=dict(text=title, font=dict(color=TEXT_C, size=16), x=0.5, xanchor="center") if title else None,
        height=height,
        margin=dict(l=50, r=30, t=60 if title else 30, b=50),
        hoverlabel=dict(bgcolor=CARD_BG, bordercolor=GRID_C, font=dict(color=TEXT_C, size=12)),
        legend=dict(bgcolor="rgba(12,20,34,0.7)", bordercolor=GRID_C, borderwidth=1,
                    font=dict(color=LABEL_C)),
    )
    fig.update_xaxes(gridcolor=GRID_C, zerolinecolor=GRID_C, linecolor=GRID_C)
    fig.update_yaxes(gridcolor=GRID_C, zerolinecolor=GRID_C, linecolor=GRID_C)
    return fig


# ════════════════════════════════════════════════════════════
# SQLITE — PERSISTENT ANALYSIS DATABASE
# ════════════════════════════════════════════════════════════
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    """Create the analyses table if it doesn't exist. Safe to call every run."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                tic_id         TEXT    NOT NULL,
                analyzed_at    TEXT    NOT NULL,
                period_days    REAL,
                depth_pct      REAL,
                snr            REAL,
                radius_earth   REAL,
                duration_hours REAL,
                bls_power      REAL,
                sec_ratio      REAL,
                odd_even_diff  REAL,
                sectors        INTEGER,
                ml_class       TEXT,
                ml_confidence  REAL,
                rule_verdict   TEXT,
                ai_insight     TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analyses_tic ON analyses(tic_id)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS frontier_results (
                tic_id         TEXT PRIMARY KEY,
                toi            TEXT,
                disposition    TEXT,
                surveyed_at    TEXT,
                planet_proba   REAL,
                verdict        TEXT,
                period_days    REAL,
                depth_pct      REAL,
                snr            REAL,
                radius_earth   REAL,
                tmag           REAL
            )
        """)


def save_analysis(result, insight=""):
    """Persist one successful pipeline result. Returns the new row id."""
    init_db()
    row = (
        str(result.get("tic_id", "")),
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        float(result.get("period_days", 0.0)),
        float(result.get("depth", 0.0)) * 100,
        float(result.get("snr", 0.0)),
        float(result.get("R_planet_earth", 0.0)),
        float(result.get("duration_hours", 0.0)),
        float(result.get("best_power", 0.0)),
        float(result.get("sec_ratio", 0.0)),
        float(result.get("odd_even_diff", 0.0)),
        int(result.get("n_sectors", 0)),
        str(result.get("ml_class", "")),
        float(result.get("ml_confidence", 0.0)),
        str(result.get("rule_verdict", "")),
        str(insight).replace("**", ""),
    )
    with _connect() as conn:
        cur = conn.execute("""
            INSERT INTO analyses
            (tic_id, analyzed_at, period_days, depth_pct, snr, radius_earth,
             duration_hours, bls_power, sec_ratio, odd_even_diff, sectors,
             ml_class, ml_confidence, rule_verdict, ai_insight)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, row)
        return cur.lastrowid


def fetch_analyses():
    """All saved analyses, newest first, as a DataFrame."""
    init_db()
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM analyses ORDER BY id DESC", conn
        )


def delete_analysis(row_id):
    with _connect() as conn:
        conn.execute("DELETE FROM analyses WHERE id = ?", (int(row_id),))


def clear_analyses():
    with _connect() as conn:
        conn.execute("DELETE FROM analyses")


def db_stats():
    """Quick aggregate stats for the animated overview cards."""
    df = fetch_analyses()
    if df.empty:
        return {"total": 0, "planets": 0, "fps": 0, "unique": 0,
                "avg_snr": 0.0, "best_snr": 0.0, "last": "—"}
    return {
        "total":   len(df),
        "planets": int((df["ml_class"] == "Exoplanet Candidate").sum()),
        "fps":     int(df["ml_class"].str.contains("Binary|False", na=False).sum()),
        "unique":  int(df["tic_id"].nunique()),
        "avg_snr": float(df["snr"].mean()),
        "best_snr": float(df["snr"].max()),
        "last":    str(df["analyzed_at"].iloc[0]),
    }


# ════════════════════════════════════════════════════════════
# CATALOG — NASA v9 training dataset explorer
# ════════════════════════════════════════════════════════════
def load_catalog():
    """v9 feature dataset (~2,000 NASA TOI stars) merged with stellar params."""
    fname = ("features_dataset_clean.csv"
             if os.path.exists("features_dataset_clean.csv")
             else "features_dataset.csv")
    if not os.path.exists(fname):
        return pd.DataFrame()
    df = pd.read_csv(fname)
    # Sanitize: ±inf / negatives break Plotly sizes and log axes
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df["depth_pct"]    = df["depth"] * 100
    df["radius_earth"] = np.sqrt(np.abs(df["depth"])) * 1.1 * 109.076
    df["Type"]         = df["label"].map(LABEL_NAMES).fillna(df["label"])
    df["size_pw"]      = df["bls_power"].clip(lower=0).fillna(0) + 1

    # Merge stellar parameters where available (Teff, radius, mass, …)
    # tic_id must be str on both sides: the catalog mixes TIC (numeric) and
    # KIC-prefixed Kepler ids since v10.2, while stellar_params.csv is numeric.
    if os.path.exists("stellar_params.csv"):
        df["tic_id"] = df["tic_id"].astype(str)
        sp = pd.read_csv("stellar_params.csv")
        if os.path.exists("stellar_params_kepler.csv"):
            sp = pd.concat([sp, pd.read_csv("stellar_params_kepler.csv")],
                           ignore_index=True)
        sp["tic_id"] = sp["tic_id"].astype(str)
        keep = [c for c in ["tic_id", "Teff", "rad", "mass", "logg", "Tmag", "lum"]
                if c in sp.columns]
        df = df.merge(sp[keep], on="tic_id", how="left", suffixes=("", "_sp"))
    return df


def available_features(df):
    """Feature columns actually present in the loaded catalog."""
    return {c: n for c, n in {**FEATURE_LABELS, **STELLAR_LABELS}.items()
            if c in df.columns}


def star_label(t):
    """Display label for a star id — 'KIC…' shown as-is, else 'TIC <id>'."""
    t = str(t)
    return t if t.startswith("KIC") else f"TIC {t}"


# ════════════════════════════════════════════════════════════
# PLOTLY FIGURES — interactive, animated, dark-themed
# ════════════════════════════════════════════════════════════
def fig_class_donut(df):
    """Animated donut — dataset composition."""
    counts = df["Type"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.62,
        marker=dict(colors=[GREEN if "Planet" in l else RED for l in counts.index],
                    line=dict(color=BG, width=3)),
        textinfo="label+percent", textfont=dict(color=TEXT_C, size=13),
        pull=[0.04] * len(counts), rotation=90,
    ))
    fig.add_annotation(text=f"<b>{len(df)}</b><br>stars", showarrow=False,
                       font=dict(size=22, color=TEXT_C))
    return style_fig(fig, "Dataset Composition", height=380)


def fig_snr_depth(df):
    """Interactive log-log SNR vs depth scatter — the classifier's view."""
    fig = px.scatter(
        df, x="snr", y="depth_pct", color="Type", color_discrete_map=COLOR_MAP,
        size="size_pw", size_max=26, opacity=0.75,
        log_x=True, log_y=True,
        hover_name=df["tic_id"].apply(star_label),
        hover_data={"snr": ":.1f", "depth_pct": ":.4f",
                    "period_days": ":.3f", "bls_power": ":.0f",
                    "size_pw": False, "Type": False},
        labels={"snr": "SNR (log)", "depth_pct": "Transit Depth % (log)"},
    )
    fig.update_traces(marker=dict(line=dict(width=1, color=GRID_C)))
    return style_fig(fig, "SNR vs Transit Depth — bubble size = BLS power", height=500)


def fig_period_radius(df):
    """Period–radius diagram: the classic exoplanet population plot."""
    fig = px.scatter(
        df, x="period_days", y="radius_earth", color="Type",
        color_discrete_map=COLOR_MAP,
        log_x=True, log_y=True, opacity=0.8,
        hover_name=df["tic_id"].apply(star_label),
        hover_data={"period_days": ":.3f", "radius_earth": ":.2f",
                    "snr": ":.1f", "Type": False},
        labels={"period_days": "Orbital Period (days, log)",
                "radius_earth": "Radius (R⊕, log)"},
    )
    for y0, y1, name, c in [(0.5, 1.6, "Rocky", "#7090b8"),
                            (1.6, 4, "Sub-Neptune", ACCENT),
                            (4, 10, "Neptune-size", PURPLE),
                            (10, 25, "Jupiter-size", ORANGE)]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=c, opacity=0.05, line_width=0,
                      annotation_text=name, annotation_position="right",
                      annotation_font=dict(size=10, color=c))
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color=GRID_C)))
    return style_fig(fig, "Period–Radius Diagram (NASA population view)", height=520)


def fig_animated_discovery(df):
    """▶️ Animated build-up: stars revealed in order of detection strength."""
    d = df[(df["period_days"] > 0) & (df["depth_pct"] > 0)] \
        .dropna(subset=["period_days", "depth_pct", "snr"]) \
        .sort_values("snr").reset_index(drop=True)
    n_frames = 30
    step = max(1, int(np.ceil(len(d) / n_frames)))
    frames_data = []
    for i, upto in enumerate(range(step, len(d) + step, step)):
        chunk = d.iloc[:upto].copy()
        chunk["frame"] = i
        frames_data.append(chunk)
    anim = pd.concat(frames_data, ignore_index=True)
    fig = px.scatter(
        anim, x="period_days", y="depth_pct", color="Type",
        color_discrete_map=COLOR_MAP,
        animation_frame="frame", log_x=True, log_y=True,
        hover_name=anim["tic_id"].apply(star_label),
        range_x=[max(d["period_days"].min() * 0.7, 1e-3),
                 d["period_days"].max() * 1.4],
        range_y=[max(d["depth_pct"].min() * 0.7, 1e-5),
                 d["depth_pct"].max() * 1.4],
        labels={"period_days": "Orbital Period (days, log)",
                "depth_pct": "Transit Depth % (log)"},
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8,
                                  line=dict(width=1, color=GRID_C)))
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 100
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 60
    fig.layout.updatemenus[0].bgcolor = CARD_BG
    fig.layout.updatemenus[0].font = dict(color=TEXT_C)
    if fig.layout.sliders:
        fig.layout.sliders[0].currentvalue.prefix = "reveal step "
        fig.layout.sliders[0].font = dict(color=LABEL_C)
    return style_fig(fig, "▶️ Press Play — Discovery Animation (weakest → strongest signal)",
                     height=560)


def _scatter3d(df, x, y, z, xlab, ylab, zlab, title, logs=(True, True, True)):
    fig = go.Figure()
    for label, color in [("planet", GREEN), ("false_positive", RED)]:
        sub = df[df["label"] == label].dropna(subset=[x, y, z])
        if sub.empty:
            continue
        def tf(s, do_log):
            return np.log10(s.clip(lower=1e-6)) if do_log else s
        fig.add_trace(go.Scatter3d(
            x=tf(sub[x], logs[0]), y=tf(sub[y], logs[1]), z=tf(sub[z], logs[2]),
            mode="markers", name=LABEL_NAMES[label],
            text=[star_label(t) for t in sub["tic_id"]],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=4, color=color, opacity=0.75,
                        line=dict(width=0.5, color=GRID_C)),
        ))
    def axt(lab, do_log):
        return ("log₁₀ " if do_log else "") + lab
    fig.update_layout(scene=dict(
        xaxis=dict(title=axt(xlab, logs[0]), backgroundcolor=BG, gridcolor=GRID_C, color=LABEL_C),
        yaxis=dict(title=axt(ylab, logs[1]), backgroundcolor=BG, gridcolor=GRID_C, color=LABEL_C),
        zaxis=dict(title=axt(zlab, logs[2]), backgroundcolor=BG, gridcolor=GRID_C, color=LABEL_C),
        bgcolor=BG,
    ))
    return style_fig(fig, title, height=600)


def fig_3d_features(df):
    """3-D classic feature space: period × depth × SNR."""
    return _scatter3d(df, "period_days", "depth_pct", "snr",
                      "Period (d)", "Depth (%)", "SNR",
                      "🧊 3-D Feature Space — Period × Depth × SNR (drag to rotate)")


def fig_3d_shape(df):
    """3-D v9 physics space: transit shape × depth consistency × asymmetry."""
    cols = ["transit_shape", "depth_consistency", "ingress_egress_asymmetry"]
    if not all(c in df.columns for c in cols):
        return None
    return _scatter3d(df, *cols,
                      "Transit Shape", "Depth Consistency", "Ingress/Egress Asym.",
                      "🧊 3-D Physics Space — v9 shape features (drag to rotate)",
                      logs=(False, False, False))


def fig_hr_diagram(df):
    """Hertzsprung–Russell-style diagram from merged stellar params."""
    if "Teff" not in df.columns or "rad" not in df.columns:
        return None
    d = df.dropna(subset=["Teff", "rad"])
    d = d[(d["Teff"] > 2000) & (d["rad"] > 0)]
    if len(d) < 10:
        return None
    fig = px.scatter(
        d, x="Teff", y="rad", color="Type", color_discrete_map=COLOR_MAP,
        log_y=True, opacity=0.85,
        hover_name=d["tic_id"].apply(star_label),
        hover_data={"Teff": ":.0f", "rad": ":.2f",
                    "mass": ":.2f" if "mass" in d.columns else False, "Type": False},
        labels={"Teff": "Effective Temperature (K)", "rad": "Stellar Radius (R☉, log)"},
    )
    fig.update_xaxes(autorange="reversed")  # HR convention: hot stars on the left
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color=GRID_C)))
    return style_fig(fig, f"⭐ Host-Star HR Diagram — {len(d)} stars with TIC stellar params",
                     height=500)


def fig_radar_profile(df):
    """Radar: mean normalized feature profile — planet vs false positive."""
    cols = [c for c in FEATURE_LABELS if c in df.columns]
    d = df[cols + ["label"]].copy()
    # Normalize each feature 0-1 by rank so wildly different scales compare fairly
    for c in cols:
        d[c] = d[c].rank(pct=True)
    fig = go.Figure()
    for label, color in [("planet", GREEN), ("false_positive", RED)]:
        means = d[d["label"] == label][cols].mean()
        fig.add_trace(go.Scatterpolar(
            r=list(means.values) + [means.values[0]],
            theta=[FEATURE_LABELS[c] for c in cols] + [FEATURE_LABELS[cols[0]]],
            fill="toself", name=LABEL_NAMES[label],
            line=dict(color=color, width=2),
            fillcolor=color.replace(")", ""), opacity=0.55,
        ))
    fig.update_layout(polar=dict(
        bgcolor=BG,
        radialaxis=dict(range=[0, 1], gridcolor=GRID_C, color=LABEL_C,
                        tickfont=dict(size=9)),
        angularaxis=dict(gridcolor=GRID_C, color=LABEL_C, tickfont=dict(size=10)),
    ))
    return style_fig(fig, "🕸️ Feature Fingerprint — planets vs false positives "
                          "(percentile-normalized means)", height=520)


def fig_parallel_coords(df, max_rows=600):
    """Parallel coordinates across the strongest separating v9 features."""
    cols = [c for c in ["snr", "depth_pct", "sec_ratio", "transit_shape",
                        "depth_consistency", "odd_even_ratio", "bls_power"]
            if c in df.columns]
    d = df.dropna(subset=cols).copy()
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=42)
    d["is_planet"] = (d["label"] == "planet").astype(int)
    # log-compress heavy-tailed columns for readable axes
    dims = []
    for c in cols:
        vals = d[c]
        if c in ("snr", "depth_pct", "bls_power"):
            vals = np.log10(vals.clip(lower=1e-6))
            name = "log " + FEATURE_LABELS.get(c, c) if c != "depth_pct" else "log Depth (%)"
        else:
            name = FEATURE_LABELS.get(c, c)
        dims.append(dict(label=name, values=vals))
    fig = go.Figure(go.Parcoords(
        line=dict(color=d["is_planet"],
                  colorscale=[[0, RED], [1, GREEN]], cmin=0, cmax=1),
        dimensions=dims,
        labelfont=dict(color=LABEL_C, size=11),
        tickfont=dict(color=LABEL_C, size=9),
        rangefont=dict(color=LABEL_C, size=8),
    ))
    return style_fig(fig, "🧵 Parallel Coordinates — drag along any axis to filter "
                          "(green = planet, red = false positive)", height=480)


def fig_sunburst(df):
    """Sunburst: label → planet-size class breakdown."""
    d = df.copy()
    d["Size class"] = pd.cut(
        d["radius_earth"], bins=[0, 1.6, 4, 10, 30, np.inf],
        labels=["Rocky", "Sub-Neptune", "Neptune-size", "Jupiter-size", "Stellar-size"])
    d["Size class"] = d["Size class"].cat.add_categories("Unknown").fillna("Unknown")
    fig = px.sunburst(
        d, path=["Type", "Size class"],
        color="Type", color_discrete_map={**COLOR_MAP, "(?)": GRID_C},
    )
    fig.update_traces(textfont=dict(color=TEXT_C, size=12),
                      insidetextorientation="radial",
                      marker=dict(line=dict(color=BG, width=2)))
    return style_fig(fig, "☀️ Sunburst — click a ring to zoom in", height=480)


def fig_feature_histogram(df, col):
    """Overlaid distribution of any selected feature, planets vs FPs."""
    fig = go.Figure()
    heavy_tail = df[col].dropna().abs().max() > 50 * max(df[col].dropna().abs().median(), 1e-9)
    for label, color in [("planet", GREEN), ("false_positive", RED)]:
        vals = df[df["label"] == label][col].dropna()
        if heavy_tail:
            vals = np.log10(vals.clip(lower=1e-9))
        fig.add_trace(go.Histogram(
            x=vals, name=LABEL_NAMES[label],
            marker_color=color, opacity=0.6, nbinsx=40,
        ))
    name = {**FEATURE_LABELS, **STELLAR_LABELS}.get(col, col)
    xlabel = f"log₁₀ {name}" if heavy_tail else name
    fig.update_layout(barmode="overlay", xaxis_title=xlabel,
                      yaxis_title="Number of stars")
    return style_fig(fig, f"Distribution — {name}", height=400)


def fig_corr_heatmap(df):
    """Correlation matrix of all v9 ML features."""
    cols = [c for c in FEATURE_LABELS if c in df.columns]
    corr = df[cols].corr()
    short = {c: c.replace("_", " ")[:16] for c in cols}
    corr = corr.rename(index=short, columns=short)
    fig = px.imshow(
        corr, text_auto=".2f", zmin=-1, zmax=1,
        color_continuous_scale=[[0, RED], [0.5, BG], [1, ACCENT]],
        aspect="auto",
    )
    fig.update_traces(textfont=dict(color=TEXT_C, size=9))
    fig.update_coloraxes(colorbar=dict(tickfont=dict(color=LABEL_C)))
    return style_fig(fig, "Feature Correlation Matrix (11 v9 features)", height=560)


def fig_confidence_gauge(confidence, ml_class):
    """Animated gauge for a single saved analysis."""
    color = GREEN if ml_class == "Exoplanet Candidate" else \
            RED if ("Binary" in str(ml_class) or "False" in str(ml_class)) else "#555577"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(confidence),
        number=dict(suffix="%", font=dict(color=TEXT_C, size=36)),
        title=dict(text=str(ml_class), font=dict(color=LABEL_C, size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=LABEL_C, tickfont=dict(color=LABEL_C)),
            bar=dict(color=color, thickness=0.28),
            bgcolor=CARD_BG, borderwidth=1, bordercolor=GRID_C,
            steps=[dict(range=[0, 50], color="#101c30"),
                   dict(range=[50, 80], color="#132440"),
                   dict(range=[80, 100], color="#173052")],
            threshold=dict(line=dict(color=YELLOW, width=3), thickness=0.8, value=97.6),
        ),
    ))
    return style_fig(fig, height=300)


def fig_analyses_timeline(df):
    """Timeline of saved analyses — SNR over time, colored by verdict."""
    if df.empty:
        return None
    d = df.copy()
    d["analyzed_at"] = pd.to_datetime(d["analyzed_at"])
    color_map = {}
    for v in d["ml_class"].unique():
        color_map[v] = GREEN if v == "Exoplanet Candidate" else \
                       RED if ("Binary" in str(v) or "False" in str(v)) else "#8a8ab0"
    fig = px.scatter(
        d, x="analyzed_at", y="snr", color="ml_class",
        color_discrete_map=color_map,
        size=d["ml_confidence"].clip(lower=1), size_max=22,
        hover_name=d["tic_id"].apply(star_label),
        hover_data={"ml_confidence": ":.1f", "period_days": ":.3f", "ml_class": False},
        labels={"analyzed_at": "Analyzed at", "snr": "SNR", "ml_class": "Verdict"},
    )
    fig.update_traces(marker=dict(line=dict(width=1, color=GRID_C)))
    return style_fig(fig, "📈 Your Analysis Timeline — bubble size = ML confidence", height=420)


# ════════════════════════════════════════════════════════════
# ORBITAL PHYSICS — derived data from stellar parameters
# ════════════════════════════════════════════════════════════
R_SUN_AU = 0.00465  # solar radius in AU


def planet_physics(period_days, st_mass=1.0, st_rad=1.0, st_teff=5778.0):
    """Derive orbit + climate quantities from Kepler's 3rd law and stellar params.

    Returns: semi-major axis (AU), stellar luminosity (L☉), equilibrium
    temperature (K, albedo 0.3), habitable-zone bounds (AU), and in_hz flag.
    """
    st_mass = float(st_mass) if st_mass and st_mass > 0 else 1.0
    st_rad  = float(st_rad)  if st_rad  and st_rad  > 0 else 1.0
    st_teff = float(st_teff) if st_teff and st_teff > 1000 else 5778.0
    a_au = (st_mass * (float(period_days) / 365.25) ** 2) ** (1 / 3)
    lum  = st_rad ** 2 * (st_teff / 5778.0) ** 4
    teq  = st_teff * np.sqrt(st_rad * R_SUN_AU / (2 * a_au)) * (1 - 0.3) ** 0.25
    hz_in, hz_out = np.sqrt(lum / 1.1), np.sqrt(lum / 0.53)  # Kasting-style bounds
    return {
        "a_au": a_au, "lum": lum, "teq": teq,
        "hz_in": hz_in, "hz_out": hz_out,
        "in_hz": bool(hz_in <= a_au <= hz_out),
    }


def _circle3d(radius, tilt_deg=12.0, n=120):
    th = np.linspace(0, 2 * np.pi, n)
    x, y = radius * np.cos(th), radius * np.sin(th)
    z = y * np.tan(np.radians(tilt_deg))
    return x, y, z


def fig_orbit_3d(period_days, st_mass=1.0, st_rad=1.0, st_teff=5778.0,
                 planet_radius_earth=2.0, tic_id="", ml_class=""):
    """▶️ Animated 3-D orbit simulator: star, habitable zone, orbiting planet."""
    phys = planet_physics(period_days, st_mass, st_rad, st_teff)
    a = phys["a_au"]
    star_color = ("#aaccff" if st_teff > 6500 else
                  "#ffdd66" if st_teff > 5000 else "#ff8844")
    planet_color = GREEN if "Exoplanet" in str(ml_class) else ACCENT

    ox, oy, oz = _circle3d(a)
    hix, hiy, hiz = _circle3d(phys["hz_in"])
    hox, hoy, hoz = _circle3d(phys["hz_out"])

    data = [
        # 0: star
        go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                     marker=dict(size=22, color=star_color, opacity=1,
                                 line=dict(width=2, color="#fff8e0")),
                     name=f"Star (Teff {st_teff:.0f} K)", hovertext="Host star",
                     hoverinfo="text"),
        # 1-2: habitable zone bounds
        go.Scatter3d(x=hix, y=hiy, z=hiz, mode="lines",
                     line=dict(color=GREEN, width=3, dash="dash"),
                     name=f"HZ inner ({phys['hz_in']:.3f} AU)", hoverinfo="name"),
        go.Scatter3d(x=hox, y=hoy, z=hoz, mode="lines",
                     line=dict(color=GREEN, width=3, dash="dot"),
                     name=f"HZ outer ({phys['hz_out']:.3f} AU)", hoverinfo="name"),
        # 3: orbit path
        go.Scatter3d(x=ox, y=oy, z=oz, mode="lines",
                     line=dict(color=ORANGE, width=4),
                     name=f"Orbit ({a:.4f} AU / {period_days:.2f} d)",
                     hoverinfo="name"),
        # 4: planet (animated)
        go.Scatter3d(x=[ox[0]], y=[oy[0]], z=[oz[0]], mode="markers",
                     marker=dict(size=max(6, min(16, 4 + planet_radius_earth)),
                                 color=planet_color,
                                 line=dict(width=1, color="#ffffff")),
                     name=f"Planet ({planet_radius_earth:.1f} R⊕)",
                     hovertext=f"Teq ≈ {phys['teq']:.0f} K", hoverinfo="text"),
    ]
    n_frames = 60
    idx = np.linspace(0, len(ox) - 1, n_frames).astype(int)
    frames = [go.Frame(
        data=[go.Scatter3d(x=[ox[i]], y=[oy[i]], z=[oz[i]], mode="markers")],
        traces=[4], name=str(k),
    ) for k, i in enumerate(idx)]

    fig = go.Figure(data=data, frames=frames)
    lim = max(phys["hz_out"], a) * 1.25
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-lim, lim], title="AU", backgroundcolor=BG,
                       gridcolor=GRID_C, color=LABEL_C),
            yaxis=dict(range=[-lim, lim], title="AU", backgroundcolor=BG,
                       gridcolor=GRID_C, color=LABEL_C),
            zaxis=dict(range=[-lim, lim], title="", backgroundcolor=BG,
                       gridcolor=GRID_C, color=LABEL_C, showticklabels=False),
            bgcolor=BG, aspectmode="cube",
            camera=dict(eye=dict(x=1.3, y=1.3, z=0.5)),
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, bgcolor=CARD_BG,
            font=dict(color=TEXT_C), x=0.05, y=0.02,
            buttons=[
                dict(label="▶️ Orbit", method="animate",
                     args=[None, dict(frame=dict(duration=50, redraw=True),
                                      transition=dict(duration=0),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )],
    )
    hz_note = "🌿 inside the habitable zone!" if phys["in_hz"] else \
              "🔥 inside the HZ inner edge" if a < phys["hz_in"] else "🧊 beyond the HZ"
    return style_fig(
        fig, f"🌍 3-D Orbit Simulator — TIC {tic_id} | Teq ≈ {phys['teq']:.0f} K | {hz_note}",
        height=620), phys


def fig_stellar_3d(df):
    """3-D host-star space: temperature × gravity × radius."""
    cols = ["Teff", "logg", "rad"]
    if not all(c in df.columns for c in cols):
        return None
    d = df.dropna(subset=cols)
    d = d[(d["Teff"] > 2000) & (d["rad"] > 0)]
    if len(d) < 10:
        return None
    fig = go.Figure()
    for label, color in [("planet", GREEN), ("false_positive", RED)]:
        sub = d[d["label"] == label]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=sub["Teff"], y=sub["logg"], z=np.log10(sub["rad"].clip(lower=1e-3)),
            mode="markers", name=LABEL_NAMES[label],
            text=[star_label(t) for t in sub["tic_id"]],
            hovertemplate="%{text}<br>Teff %{x:.0f} K | log g %{y:.2f}<extra></extra>",
            marker=dict(size=5, color=color, opacity=0.8,
                        line=dict(width=0.5, color=GRID_C)),
        ))
    fig.update_layout(scene=dict(
        xaxis=dict(title="Teff (K)", backgroundcolor=BG, gridcolor=GRID_C,
                   color=LABEL_C, autorange="reversed"),
        yaxis=dict(title="log g", backgroundcolor=BG, gridcolor=GRID_C, color=LABEL_C),
        zaxis=dict(title="log₁₀ R (R☉)", backgroundcolor=BG, gridcolor=GRID_C, color=LABEL_C),
        bgcolor=BG,
    ))
    return style_fig(fig, f"⭐ 3-D Host-Star Space — {len(d)} stars with stellar params",
                     height=560)


# ════════════════════════════════════════════════════════════
# SHAP EXPLANATION — per-prediction feature contributions
# ════════════════════════════════════════════════════════════
def fig_shap(shap_dict, ml_class=""):
    """Horizontal bar chart of per-feature SHAP contributions for ONE star.

    Positive (green) pushes toward planet, negative (red) toward false positive.
    """
    items = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
    names = [{**FEATURE_LABELS, **STELLAR_LABELS}.get(k, k) for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    colors = [GREEN if v > 0 else RED for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in vals], textposition="outside",
        textfont=dict(color=TEXT_C, size=10),
    ))
    fig.add_vline(x=0, line_color=LABEL_C, line_width=1)
    fig.update_layout(xaxis_title="SHAP contribution (log-odds → planet)",
                      yaxis=dict(tickfont=dict(size=10)))
    return style_fig(fig, "🧠 Why the model decided this — SHAP feature contributions "
                          "(green → planet, red → false positive)", height=460)


# ════════════════════════════════════════════════════════════
# TOI CROSS-MATCH — is this star already a known object?
# ════════════════════════════════════════════════════════════
_TOI_CACHE = None

def lookup_toi(tic_id):
    """Cross-match a TIC ID against the live-pulled TOI catalog.

    Returns dict with disposition info, or None if not in the catalog.
    Dispositions: CP/KP = confirmed/known planet, FP/FA = false positive,
    PC/APC = unconfirmed candidate (the interesting case!).
    """
    global _TOI_CACHE
    if _TOI_CACHE is None:
        fname = "toi_raw_full.csv" if os.path.exists("toi_raw_full.csv") else "toi_raw.csv"
        if not os.path.exists(fname):
            return None
        _TOI_CACHE = pd.read_csv(fname)
        _TOI_CACHE["tid"] = _TOI_CACHE["tid"].astype(str)
    rows = _TOI_CACHE[_TOI_CACHE["tid"] == str(tic_id)]
    if rows.empty:
        return None
    r = rows.iloc[0]
    disp = str(r.get("tfopwg_disp", ""))
    return {
        "toi": str(r.get("toi", "?")),
        "disposition": disp,
        "known_planet": disp in ("CP", "KP"),
        "known_fp": disp in ("FP", "FA"),
        "candidate": disp in ("PC", "APC"),
        "catalog_period": float(r["pl_orbper"]) if pd.notna(r.get("pl_orbper")) else None,
        "catalog_radius": float(r["pl_rade"]) if pd.notna(r.get("pl_rade")) else None,
    }


def save_frontier_result(result):
    """Upsert one Batch Survey verdict on an unconfirmed (PC/APC) candidate.

    Called for every successful survey run; silently ignores stars that are
    already confirmed / known false positives so the leaderboard stays a list
    of genuinely open questions. Returns True if the row was stored.
    """
    xm = lookup_toi(result.get("tic_id", ""))
    if xm is None or not xm.get("candidate"):
        return False
    init_db()
    tmag = None
    try:
        rows = _TOI_CACHE[_TOI_CACHE["tid"] == str(result["tic_id"])]
        if not rows.empty and pd.notna(rows.iloc[0].get("st_tmag")):
            tmag = float(rows.iloc[0]["st_tmag"])
    except Exception:
        pass
    with _connect() as conn:
        conn.execute("""
            INSERT INTO frontier_results
            (tic_id, toi, disposition, surveyed_at, planet_proba, verdict,
             period_days, depth_pct, snr, radius_earth, tmag)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(tic_id) DO UPDATE SET
                surveyed_at=excluded.surveyed_at,
                planet_proba=excluded.planet_proba,
                verdict=excluded.verdict,
                period_days=excluded.period_days,
                depth_pct=excluded.depth_pct,
                snr=excluded.snr,
                radius_earth=excluded.radius_earth
        """, (
            str(result["tic_id"]),
            xm.get("toi", "?"),
            xm.get("disposition", ""),
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            float(result.get("planet_proba", 0.0)),
            str(result.get("ml_class", "")),
            float(result.get("period_days", 0.0)),
            float(result.get("depth", 0.0)) * 100,
            float(result.get("snr", 0.0)),
            float(result.get("R_planet_earth", 0.0)),
            tmag,
        ))
    return True


def load_frontier_results():
    """All accumulated frontier verdicts, highest planet probability first."""
    init_db()
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM frontier_results ORDER BY planet_proba DESC", conn
        )


def fig_frontier_leaderboard(df, top_n=20):
    """Horizontal bar chart of the top-N most-likely-real unconfirmed candidates."""
    d = df.head(top_n).iloc[::-1]
    labels = d.apply(lambda r: f"TOI {r['toi']} (TIC {r['tic_id']})", axis=1)
    colors = ["#22cc77" if p >= 0.65 else "#4a9eff" if p >= 0.5 else "#8090a8"
              for p in d["planet_proba"]]
    fig = go.Figure(go.Bar(
        x=d["planet_proba"] * 100, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in d["planet_proba"]],
        textposition="outside",
        customdata=np.stack([d["period_days"], d["radius_earth"], d["snr"]], axis=-1),
        hovertemplate="%{y}<br>Planet prob: %{x:.1f}%<br>"
                      "Period: %{customdata[0]:.3f} d<br>"
                      "Radius: %{customdata[1]:.2f} R⊕<br>"
                      "SNR: %{customdata[2]:.1f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 108], title="Calibrated planet probability (%)")
    return style_fig(fig, f"🏆 Top {len(d)} most-likely-real unconfirmed candidates",
                     height=max(360, 30 * len(d) + 120))


def fig_frontier_scatter(df):
    """Period vs radius of surveyed frontier candidates, colored by probability."""
    d = df[(df["period_days"] > 0) & (df["radius_earth"] > 0)].copy()
    d["Prob (%)"] = d["planet_proba"] * 100
    fig = px.scatter(
        d, x="period_days", y="radius_earth", color="Prob (%)",
        color_continuous_scale=["#8090a8", "#4a9eff", "#22cc77"],
        size=np.clip(d["snr"], 1, 60),
        hover_name=d.apply(lambda r: f"TOI {r['toi']}", axis=1),
        hover_data={"tic_id": True, "snr": ":.1f"},
        log_x=True,
        labels={"period_days": "Orbital period (days)",
                "radius_earth": "Planet radius (R⊕)"},
    )
    return style_fig(fig, "🌌 Surveyed frontier — period vs radius", height=480)


def load_frontier(n=None):
    """Unlabeled PC/APC candidates for the Batch Survey page."""
    if not os.path.exists("frontier_targets.csv"):
        return pd.DataFrame()
    df = pd.read_csv("frontier_targets.csv")
    return df.head(n) if n else df


# ════════════════════════════════════════════════════════════
# SKY MAP — where the dataset lives on the celestial sphere
# ════════════════════════════════════════════════════════════
def load_sky_targets():
    """TOI targets with RA/Dec, labeled by disposition group."""
    if not os.path.exists("toi_raw_full.csv"):
        return pd.DataFrame()
    df = pd.read_csv("toi_raw_full.csv").dropna(subset=["ra", "dec"])
    disp_group = {
        "CP": "🪐 Confirmed planet", "KP": "🪐 Confirmed planet",
        "FP": "⭐ False positive", "FA": "⭐ False positive",
        "PC": "🔭 Unconfirmed candidate", "APC": "🔭 Unconfirmed candidate",
    }
    df["Group"] = df["tfopwg_disp"].map(disp_group).fillna("Other")
    return df[df["Group"] != "Other"]


SKY_COLORS = {"🪐 Confirmed planet": GREEN,
              "⭐ False positive": RED,
              "🔭 Unconfirmed candidate": ACCENT}


def fig_sky_map(df):
    """2-D celestial map (RA/Dec) of the full TOI population."""
    fig = px.scatter(
        df, x="ra", y="dec", color="Group", color_discrete_map=SKY_COLORS,
        opacity=0.65,
        hover_name=df["tid"].apply(star_label),
        hover_data={"toi": True, "ra": ":.2f", "dec": ":.2f",
                    "st_tmag": ":.1f", "Group": False},
        labels={"ra": "Right Ascension (deg)", "dec": "Declination (deg)"},
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_xaxes(autorange="reversed")  # sky convention: RA increases leftward
    # TESS continuous viewing zones sit near the ecliptic poles
    for dec0, name in [(84, "N continuous viewing zone"), (-84, "S continuous viewing zone")]:
        fig.add_hline(y=dec0, line_dash="dot", line_color=LABEL_C, opacity=0.4,
                      annotation_text=name, annotation_font=dict(size=9, color=LABEL_C))
    return style_fig(fig, f"🗺️ Sky Map — {len(df)} TOI targets (RA reversed, sky convention)",
                     height=540)


def fig_sky_3d(df, max_pts=4000):
    """Rotatable 3-D celestial sphere with every TOI pinned on it."""
    d = df.sample(max_pts, random_state=42) if len(df) > max_pts else df
    ra = np.radians(d["ra"].values)
    dec = np.radians(d["dec"].values)
    x, y, z = np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)

    fig = go.Figure()
    # faint wireframe sphere
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(-np.pi / 2, np.pi / 2, 18)
    for vv in v[::3]:
        fig.add_trace(go.Scatter3d(
            x=np.cos(vv) * np.cos(u), y=np.cos(vv) * np.sin(u),
            z=np.full_like(u, np.sin(vv)), mode="lines",
            line=dict(color=GRID_C, width=1), showlegend=False, hoverinfo="skip"))
    for uu in u[::3]:
        fig.add_trace(go.Scatter3d(
            x=np.cos(v) * np.cos(uu), y=np.cos(v) * np.sin(uu), z=np.sin(v),
            mode="lines", line=dict(color=GRID_C, width=1),
            showlegend=False, hoverinfo="skip"))
    for grp, color in SKY_COLORS.items():
        m = (d["Group"] == grp).values
        if not m.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=x[m], y=y[m], z=z[m], mode="markers", name=grp,
            text=[f"TOI {t}" for t in d["toi"].values[m]],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=2.5, color=color, opacity=0.85),
        ))
    fig.update_layout(scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        bgcolor=BG, aspectmode="cube",
        camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
    ))
    return style_fig(fig, "🌐 Celestial Sphere — every TOI pinned in 3-D (drag to spin)",
                     height=640)


# ════════════════════════════════════════════════════════════
# MODEL HONESTY — ROC + calibration curves from held-out stars
# ════════════════════════════════════════════════════════════
def load_holdout():
    if not os.path.exists("holdout_predictions.csv"):
        return pd.DataFrame()
    return pd.read_csv("holdout_predictions.csv")


def fig_roc_curve(hold):
    """ROC on the untouched held-out set, with threshold hover.
    Overlays per-mission (TESS vs Kepler) curves when both are present."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thr = roc_curve(hold["y_true"], hold["planet_proba"])
    a = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", line=dict(color=ACCENT, width=3),
        name=f"All missions (AUC {a:.3f})",
        customdata=thr, hovertemplate=("FPR %{x:.2f} | TPR %{y:.2f}<br>"
                                       "threshold %{customdata:.2f}<extra></extra>"),
    ))
    is_kep = hold["tic_id"].astype(str).str.startswith("KIC")
    if is_kep.any() and (~is_kep).any():
        for mask, mname, col in [(~is_kep, "TESS", ORANGE), (is_kep, "Kepler", PURPLE)]:
            sub = hold[mask]
            if sub["y_true"].nunique() < 2:
                continue
            f_m, t_m, _ = roc_curve(sub["y_true"], sub["planet_proba"])
            a_m = auc(f_m, t_m)
            fig.add_trace(go.Scatter(
                x=f_m, y=t_m, mode="lines",
                line=dict(color=col, width=1.6, dash="dot"),
                name=f"{mname} only, {len(sub)} stars (AUC {a_m:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=LABEL_C, dash="dash", width=1),
                             name="Random guess"))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return style_fig(fig, f"ROC Curve — {len(hold)} never-seen stars (AUC {a:.3f})",
                     height=440)


def fig_reliability(hold, n_bins=8):
    """Calibration curve: does '80% planet' really mean 80%?"""
    d = hold.copy()
    d["bin"] = pd.cut(d["planet_proba"], bins=np.linspace(0, 1, n_bins + 1))
    g = d.groupby("bin", observed=True).agg(
        mean_pred=("planet_proba", "mean"),
        frac_true=("y_true", "mean"),
        n=("y_true", "size")).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=LABEL_C, dash="dash", width=1),
                             name="Perfect calibration"))
    fig.add_trace(go.Scatter(
        x=g["mean_pred"], y=g["frac_true"], mode="lines+markers",
        line=dict(color=GREEN, width=3),
        marker=dict(size=(4 + 14 * g["n"] / g["n"].max()).tolist(), color=GREEN),
        customdata=g["n"], name="ExoDetect v10 (isotonic)",
        hovertemplate=("predicted %{x:.2f} → actual %{y:.2f}<br>"
                       "%{customdata} stars in bin<extra></extra>"),
    ))
    fig.update_layout(xaxis_title="Predicted planet probability",
                      yaxis_title="Actual fraction that are planets",
                      xaxis_range=[0, 1], yaxis_range=[0, 1.02])
    return style_fig(fig, "Reliability Curve — calibrated probabilities on held-out stars",
                     height=440)


def fig_proba_split(hold):
    """Distribution of predicted probability, split by ground truth."""
    fig = go.Figure()
    for val, name, color in [(1, "True planets", GREEN), (0, "True false positives", RED)]:
        fig.add_trace(go.Histogram(
            x=hold[hold["y_true"] == val]["planet_proba"],
            name=name, marker_color=color, opacity=0.6, nbinsx=25,
        ))
    fig.update_layout(barmode="overlay",
                      xaxis_title="Predicted planet probability",
                      yaxis_title="Held-out stars")
    return style_fig(fig, "Probability Separation — ideal is red left, green right",
                     height=400)


# ════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════
def export_csv(df):
    return df.to_csv(index=False).encode()


def export_json(df):
    return json.dumps(df.to_dict(orient="records"), indent=2, default=str).encode()
