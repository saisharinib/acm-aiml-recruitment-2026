"""Interactive anomaly detection dashboard (Streamlit)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["temperature", "vibration", "power"]
DATA_PATH = Path(__file__).parent / "sensor_data.csv"

st.set_page_config(page_title="Sensor Anomaly Dashboard", layout="wide")


@st.cache_data
def load_default():
    return pd.read_csv(DATA_PATH)


def iqr_flags(df, k):
    """Flag a reading if ANY sensor lies outside [Q1 - k*IQR, Q3 + k*IQR]."""
    flags = pd.DataFrame(index=df.index)
    for f in FEATURES:
        q1, q3 = df[f].quantile([0.25, 0.75])
        iqr = q3 - q1
        flags[f] = (df[f] < q1 - k * iqr) | (df[f] > q3 + k * iqr)
    return flags.any(axis=1).astype(int)


@st.cache_data
def run_isolation_forest(values, n_trees, contamination):
    X = StandardScaler().fit_transform(values)
    model = IsolationForest(n_estimators=n_trees, contamination=contamination,
                            random_state=42)
    pred = (model.fit_predict(X) == -1).astype(int)
    return pred, -model.score_samples(X)  # higher score = more anomalous


def scores(y_true, y_pred):
    return (precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0))


def plot_series(sub, flag_col):
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, f in zip(axes, FEATURES):
        ax.plot(sub["time"], sub[f], lw=1)
        hit = sub[sub[flag_col] == 1]
        ax.scatter(hit["time"], hit[f], color="red", s=22, zorder=3)
        ax.set_ylabel(f)
    axes[-1].set_xlabel("time (hours)")
    fig.tight_layout()
    return fig


st.title("Sensor Anomaly Detection Dashboard")
st.caption("Change the settings in the sidebar and every chart and metric updates instantly.")

# ---------- Sidebar: user input ----------
st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")
method = st.sidebar.radio("Method to display", ["IQR", "Isolation Forest"])
k = st.sidebar.slider("IQR multiplier", 0.5, 4.0, 1.5, 0.1)
contamination = st.sidebar.slider("Isolation Forest contamination", 0.01, 0.20, 0.05, 0.01)
n_trees = st.sidebar.slider("Isolation Forest trees", 50, 500, 200, 50)
window = st.sidebar.slider("Hours shown in time plot", 100, 2000, 500, 100)
st.sidebar.markdown(
    "**Own CSV format:** columns `temperature`, `vibration`, `power`. "
    "Add `is_anomaly` (0/1) to see precision, recall and F1."
)

# ---------- Data ----------
if uploaded is not None:
    df = pd.read_csv(uploaded)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"Your CSV must contain these columns: {missing}")
        st.stop()
elif DATA_PATH.exists():
    df = load_default().copy()
else:
    st.error("sensor_data.csv not found. Copy it from task_01_anomaly_detection into this folder.")
    st.stop()

df = df.reset_index(drop=True)
if "time" not in df.columns:
    df["time"] = np.arange(len(df))
has_labels = "is_anomaly" in df.columns

# ---------- Detection ----------
df["iqr_anomaly"] = iqr_flags(df, k)
df["iso_anomaly"], df["iso_score"] = run_isolation_forest(
    df[FEATURES].values, n_trees, contamination)
flag_col = "iqr_anomaly" if method == "IQR" else "iso_anomaly"

# ---------- Metrics row ----------
flagged = int(df[flag_col].sum())
cols = st.columns(6 if has_labels else 3)
cols[0].metric("Readings", len(df))
cols[1].metric("Flagged", flagged)
cols[2].metric("Share flagged", f"{flagged / len(df):.1%}")
if has_labels:
    p, r, f1 = scores(df["is_anomaly"], df[flag_col])
    cols[3].metric("Precision", f"{p:.3f}")
    cols[4].metric("Recall", f"{r:.3f}")
    cols[5].metric("F1", f"{f1:.3f}")

# ---------- Tabs ----------
tab_series, tab_scatter, tab_dist, tab_compare, tab_data = st.tabs(
    ["Time series", "Scatter", "Distributions", "Method comparison", "Flagged data"])

with tab_series:
    st.subheader(f"{method}: flagged readings in red")
    fig = plot_series(df.iloc[:window], flag_col)
    st.pyplot(fig)
    plt.close(fig)

with tab_scatter:
    c1, c2 = st.columns(2)
    x = c1.selectbox("X axis", FEATURES, index=0)
    y = c2.selectbox("Y axis", FEATURES, index=1)
    show_truth = has_labels and st.checkbox("Circle the true anomalies")
    fig, ax = plt.subplots(figsize=(8, 5))
    normal, hit = df[df[flag_col] == 0], df[df[flag_col] == 1]
    ax.scatter(normal[x], normal[y], s=12, color="lightgrey", label="normal")
    ax.scatter(hit[x], hit[y], s=22, color="red", label="flagged")
    if show_truth:
        truth = df[df["is_anomaly"] == 1]
        ax.scatter(truth[x], truth[y], s=70, facecolors="none",
                   edgecolors="black", label="true anomaly")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with tab_dist:
    left, right = st.columns(2)
    with left:
        st.subheader("Boxplots (whiskers follow the IQR multiplier)")
        fig, axes = plt.subplots(1, 3, figsize=(8, 4))
        for ax, f in zip(axes, FEATURES):
            ax.boxplot(df[f], whis=k)
            ax.set_title(f)
            ax.set_xticks([])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Isolation Forest scores")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["iso_score"], bins=50, color="steelblue")
        ax.set_xlabel("anomaly score (higher = more anomalous)")
        ax.set_ylabel("readings")
        st.pyplot(fig)
        plt.close(fig)

with tab_compare:
    if not has_labels:
        st.info("Comparison needs an is_anomaly column. The built-in sensor data has one.")
    else:
        rows = []
        for name, col in [("IQR", "iqr_anomaly"), ("Isolation Forest", "iso_anomaly")]:
            p, r, f1 = scores(df["is_anomaly"], df[col])
            rows.append({"Method": name, "Flagged": int(df[col].sum()),
                         "Precision": round(p, 3), "Recall": round(r, 3),
                         "F1": round(f1, 3)})
        st.dataframe(pd.DataFrame(rows))
        if "anomaly_type" in df.columns:
            st.subheader("Share of each anomaly type caught")
            by_type = (df[df["is_anomaly"] == 1]
                       .groupby("anomaly_type")[["iqr_anomaly", "iso_anomaly"]]
                       .mean().round(2))
            st.dataframe(by_type)
            st.bar_chart(by_type)

with tab_data:
    flagged_df = df[df[flag_col] == 1]
    st.write(f"{len(flagged_df)} flagged readings")
    st.dataframe(flagged_df)
    st.download_button("Download flagged readings as CSV",
                       flagged_df.to_csv(index=False), "flagged.csv", "text/csv")