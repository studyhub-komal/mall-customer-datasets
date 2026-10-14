import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️", layout="wide")

# ---------- Settings ----------
DATA_FILE = "store_customers.csv"
INC, SPD = "Annual Income (k$)", "Spending Score (1-100)"
FEATURES = [INC, SPD]                      # model sirf in 2 par train hota hai (CustomerID/Age nahi)
PROFILE = ["Age", INC, SPD]                # sirf insights dikhane ke liye
N_CLUSTERS = 4
PALETTE = ["#6C63FF", "#FF6584", "#00C9A7", "#FFB703", "#3A86FF", "#FB5607", "#8338EC", "#06D6A0"]

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(120deg, #6C63FF 0%, #FF6584 100%);
        padding: 2rem 2.2rem; border-radius: 20px; color: white;
        margin-bottom: 1.4rem; box-shadow: 0 10px 30px rgba(108,99,255,.25);
    }
    .hero h1 { margin: 0; font-size: 2.2rem; color: white; }
    .hero p  { margin: .4rem 0 0 0; opacity: .92; font-size: 1.05rem; }
    .kpi {
        background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.18);
        border-radius: 16px; padding: 1rem 1.2rem; border-left: 6px solid var(--c);
    }
    .kpi .label { font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .06em; }
    .kpi .value { font-size: 1.8rem; font-weight: 700; margin-top: .1rem; }
    .result {
        border-radius: 20px; padding: 1.6rem; color: white; text-align: center;
        background: linear-gradient(135deg, var(--c1), var(--c2));
        box-shadow: 0 10px 28px rgba(0,0,0,.18);
    }
    .result .small { opacity: .85; letter-spacing: .08em; text-transform: uppercase; font-size: .8rem; }
    .result .big { font-size: 2rem; font-weight: 800; margin: .3rem 0; }
    .stTabs [data-baseweb="tab"] { font-size: 1.05rem; padding: .6rem 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Data helpers ----------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values ko bhar do taaki har customer ka segment nikal sake."""
    df = df.copy()
    for c in df.columns:
        if c == "CustomerID":
            continue
        if df[c].dtype == object or str(df[c].dtype).startswith(("str", "category")):
            if df[c].isna().any() and not df[c].mode().empty:
                df[c] = df[c].fillna(df[c].mode()[0])
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
    return df


# ---------- Model: app ke andar hi train hota hai ----------
@st.cache_resource
def train_model():
    df = clean(pd.read_csv(DATA_FILE))
    X = df[FEATURES].astype(float)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # clusters ke naam (centre dekhkar)
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES)
    names = {}
    rich = centers[INC].idxmax()
    spender = centers[SPD].idxmax()
    names[rich] = "Wealthy Savers"          # sabse zyada income, bahut kam kharch
    names[spender] = "Trendy Spenders"      # kam income, zyada kharch
    rest = sorted([i for i in centers.index if i not in names], key=lambda i: centers.loc[i, INC], reverse=True)
    if len(rest) == 2:
        names[rest[0]] = "Affluent Regulars"   # achhi income, medium kharch
        names[rest[1]] = "Average Shoppers"    # medium income, medium kharch
    else:
        for i in rest:
            names[i] = f"Segment {i}"
    names = {int(k): v for k, v in names.items()}

    # .pkl files bhi save kar do (kahin aur use karne ke liye)
    try:
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(kmeans, "kmeans_model.pkl")
        joblib.dump(names, "cluster_names.pkl")
    except Exception:
        pass

    return scaler, kmeans, names


try:
    scaler, kmeans, cluster_names = train_model()
except FileNotFoundError:
    st.error(f"`{DATA_FILE}` nahi mili. Isko app.py ke same folder mein rakho.")
    st.stop()

SEG_NAMES = [cluster_names[i] for i in range(N_CLUSTERS)]
COLOR_MAP = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(SEG_NAMES)}
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES)
centers["Segment"] = SEG_NAMES


def seg_name(label) -> str:
    return cluster_names[int(label)]


def predict(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].astype(float)
    Xs = pd.DataFrame(scaler.transform(X), columns=FEATURES)
    return kmeans.predict(Xs)


def kpi(label, value, color):
    return (f'<div class="kpi" style="--c:{color}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>')


def slider_range(name):
    n = name.lower()
    if "age" in n:
        return 18, 80, 35
    if "income" in n:
        return 10, 150, 60
    return 1, 100, 50


# ---------- Hero ----------
st.markdown(
    '<div class="hero"><h1>🛍️ Mall Customer Segmentation</h1>'
    "<p>Customers ko unke behaviour ke hisaab se 4 groups mein baantein — K-Means powered insights.</p></div>",
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Data")
    up = st.file_uploader("Upload customer CSV (optional)", type=["csv"])
    st.caption(f"Upload nahi karoge to `{DATA_FILE}` use hoga.")
    st.divider()
    st.subheader("Model info")
    st.write(f"**Clusters:** {N_CLUSTERS}")
    st.write("**Model features:** " + ", ".join(FEATURES))
    st.caption("Age sirf insights/charts ke liye dikhai jaati hai, model mein use nahi hoti.")


@st.cache_data
def load_default():
    return pd.read_csv(DATA_FILE)


raw = pd.read_csv(up) if up is not None else load_default()

data = None
missing = [c for c in FEATURES if c not in raw.columns]
if missing:
    st.warning(f"Dataset mein ye columns nahi mile: {missing}")
else:
    data = clean(raw)
    data["Cluster"] = predict(data)
    data["Segment"] = data["Cluster"].apply(seg_name)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Predict", "📁 Batch", "🧩 Segments"])

# ================= Dashboard =================
with tab1:
    if data is None:
        st.info("Dashboard ke liye sahi columns wali CSV upload karein.")
    else:
        k = st.columns(5)
        k[0].markdown(kpi("Customers", f"{len(data):,}", PALETTE[0]), unsafe_allow_html=True)
        if "Age" in data:
            k[1].markdown(kpi("Avg Age", f"{data['Age'].mean():.1f}", PALETTE[1]), unsafe_allow_html=True)
        k[2].markdown(kpi("Avg Income (k$)", f"{data[INC].mean():.1f}", PALETTE[2]), unsafe_allow_html=True)
        k[3].markdown(kpi("Avg Spending", f"{data[SPD].mean():.1f}", PALETTE[3]), unsafe_allow_html=True)
        k[4].markdown(kpi("Segments", N_CLUSTERS, PALETTE[4]), unsafe_allow_html=True)
        st.write("")

        left, right = st.columns([3, 2])
        with left:
            hover = [h for h in ["CustomerID", "Age", "Gender"] if h in data]
            fig = px.scatter(data, x=INC, y=SPD, color="Segment", color_discrete_map=COLOR_MAP,
                             opacity=0.75, hover_data=hover, title="Customer Segments")
            fig.add_trace(go.Scatter(
                x=centers[INC], y=centers[SPD], mode="markers+text", text=centers["Segment"],
                textposition="top center", name="Centroids",
                marker=dict(symbol="star", size=18, color="black", line=dict(width=2, color="white"))))
            fig.update_layout(height=470, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            share = data["Segment"].value_counts().reset_index()
            share.columns = ["Segment", "Customers"]
            fig = px.pie(share, names="Segment", values="Customers", hole=0.55,
                         color="Segment", color_discrete_map=COLOR_MAP, title="Segment Share")
            fig.update_layout(height=470, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

        l2, r2 = st.columns(2)
        with l2:
            prof_cols = [c for c in PROFILE if c in data]
            avg = data.groupby("Segment")[prof_cols].mean().reset_index().melt("Segment")
            fig = px.bar(avg, x="Segment", y="value", color="variable", barmode="group",
                         title="Average Profile per Segment", color_discrete_sequence=PALETTE)
            fig.update_layout(legend_title_text="", yaxis_title=None, xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            if "Gender" in data:
                g = data.groupby(["Segment", "Gender"]).size().reset_index(name="Customers")
                fig = px.bar(g, x="Segment", y="Customers", color="Gender", barmode="stack",
                             title="Gender Mix per Segment",
                             color_discrete_sequence=[PALETTE[0], PALETTE[1], PALETTE[2]])
                fig.update_layout(legend_title_text="", xaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)
            elif "Age" in data:
                st.plotly_chart(px.histogram(data, x="Age", color="Segment", color_discrete_map=COLOR_MAP,
                                             title="Age Distribution"), use_container_width=True)

        with st.expander("🔎 Raw data dekhein"):
            st.dataframe(data, use_container_width=True)

# ================= Predict =================
def apply_preset():
    p = st.session_state["preset"]
    if p != "Manual":
        row = centers[centers["Segment"] == p].iloc[0]
        st.session_state[f"s_{INC}"] = int(round(row[INC]))
        st.session_state[f"s_{SPD}"] = int(round(row[SPD]))


with tab2:
    st.subheader("Ek customer ka segment pata karein")
    col_in, col_out = st.columns([1, 1.2])
    values = {}
    with col_in:
        st.selectbox("Example se shuru karein", ["Manual"] + SEG_NAMES, key="preset", on_change=apply_preset)
        for feat in FEATURES:
            lo, hi, dflt = slider_range(feat)
            st.session_state.setdefault(f"s_{feat}", dflt)
            values[feat] = st.slider(feat, lo, hi, key=f"s_{feat}")

    with col_out:
        row = pd.DataFrame([values])
        label = int(predict(row)[0])
        name = seg_name(label)
        col = COLOR_MAP[name]
        st.markdown(
            f'<div class="result" style="--c1:{col};--c2:#1f1f3a">'
            f'<div class="small">Predicted segment</div><div class="big">{name}</div>'
            f'<div class="small">Cluster {label}</div></div>',
            unsafe_allow_html=True,
        )

        # har segment se kitna match
        dist = kmeans.transform(pd.DataFrame(scaler.transform(row[FEATURES]), columns=FEATURES))[0]
        w = np.exp(-dist)
        match = pd.DataFrame({"Segment": SEG_NAMES, "Match %": (w / w.sum() * 100).round(1)})
        fig = px.bar(match, x="Match %", y="Segment", orientation="h", color="Segment",
                     color_discrete_map=COLOR_MAP, title="Kis segment se kitna match", text="Match %")
        fig.update_layout(height=250, showlegend=False, margin=dict(t=40, b=10), yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    if data is not None:
        fig = px.scatter(data, x=INC, y=SPD, color="Segment", color_discrete_map=COLOR_MAP, opacity=0.35)
        fig.add_trace(go.Scatter(
            x=[values[INC]], y=[values[SPD]], mode="markers", name="Ye customer",
            marker=dict(symbol="star", size=24, color="gold", line=dict(width=2, color="black"))))
        fig.update_layout(height=380, legend_title_text="", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

# ================= Batch =================
with tab3:
    st.subheader("Bulk segmentation")
    st.write(f"CSV mein ye columns hone chahiye: `{'`, `'.join(FEATURES)}`")
    f = st.file_uploader("CSV upload karein", type=["csv"], key="batch")
    if f is not None:
        b = pd.read_csv(f)
        miss = [c for c in FEATURES if c not in b.columns]
        if miss:
            st.error(f"Missing columns: {miss}")
        else:
            b = clean(b)
            b["Cluster"] = predict(b)
            b["Segment"] = b["Cluster"].apply(seg_name)
            st.dataframe(b, use_container_width=True)
            st.plotly_chart(px.scatter(b, x=INC, y=SPD, color="Segment", color_discrete_map=COLOR_MAP),
                            use_container_width=True)
            st.download_button("⬇️ Results download karein", b.to_csv(index=False).encode(),
                               "segmented_customers.csv", "text/csv")

# ================= Segments =================
with tab4:
    st.subheader("Har segment ki pehchaan")
    if data is not None:
        prof_cols = [c for c in PROFILE if c in data]
        prof = data.groupby("Segment")[prof_cols].mean().reindex(SEG_NAMES)
        counts = data["Segment"].value_counts().reindex(SEG_NAMES)
    else:
        prof_cols = FEATURES
        prof = centers.set_index("Segment")[FEATURES]
        counts = None

    cols = st.columns(N_CLUSTERS)
    for i, sname in enumerate(SEG_NAMES):
        with cols[i % len(cols)]:
            body = "".join(f"<div>{ft}: <b>{prof.loc[sname, ft]:.1f}</b></div>" for ft in prof_cols)
            if counts is not None:
                body += f"<div>Customers: <b>{int(counts[sname])}</b></div>"
            st.markdown(
                f'<div class="kpi" style="--c:{COLOR_MAP[sname]};margin-bottom:1rem"><div class="label">Cluster {i}</div>'
                f'<div class="value" style="font-size:1.3rem">{sname}</div>{body}</div>',
                unsafe_allow_html=True)

    norm = prof.copy()
    rng = (norm.max() - norm.min()).replace(0, 1)
    norm = (norm - norm.min()) / rng
    fig = go.Figure()
    for sname, r in norm.iterrows():
        fig.add_trace(go.Scatterpolar(r=list(r) + [r.iloc[0]], theta=prof_cols + [prof_cols[0]],
                                      fill="toself", name=sname, line_color=COLOR_MAP[sname]))
    fig.update_layout(title="Segment comparison (radar)", height=450, polar=dict(radialaxis=dict(visible=False)))
    st.plotly_chart(fig, use_container_width=True)