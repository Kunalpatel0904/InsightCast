"""
InsightCast — Smart Business Analytics Dashboard
================================================
Run:  streamlit run app.py
"""
import warnings, os
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from preprocessing import load_and_clean, build_customer_features, prepare_ml_features
from churn_model    import train_churn_model, get_shap_values, predict_churn
from clustering     import find_optimal_k, run_clustering, label_segments, SEGMENT_NAMES
from forecasting    import build_monthly_revenue, forecast_revenue

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightCast Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* General */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2d2f3e; }
h1,h2,h3,h4 { color: #e2e8f0 !important; }
p, li, div { color: #cbd5e1; }

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg,#1e2235,#252840);
    border: 1px solid #3d4166;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    margin-bottom: 6px;
}
.kpi-title { font-size: 13px; color: #94a3b8; letter-spacing: .6px; text-transform: uppercase; margin-bottom: 4px; }
.kpi-value { font-size: 32px; font-weight: 700; color: #e2e8f0; }
.kpi-delta { font-size: 12px; margin-top: 4px; }

/* Risk badges */
.badge-high { background:#7f1d1d; color:#fca5a5; padding:2px 8px; border-radius:20px; font-size:11px; }
.badge-med  { background:#78350f; color:#fcd34d; padding:2px 8px; border-radius:20px; font-size:11px; }
.badge-low  { background:#14532d; color:#86efac; padding:2px 8px; border-radius:20px; font-size:11px; }

/* Metric cards in performance */
.metric-box {
    background:#1e2235; border:1px solid #2d3561;
    border-radius:10px; padding:14px; text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=72)
    st.title("InsightCast")
    st.caption("Smart Business Analytics")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload CSV dataset", type=["csv"])
    default_path = os.path.join("data", "ecommerce.csv")

    if uploaded:
        import tempfile, shutil
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.read()); tmp.close()
        DATA_PATH = tmp.name
    elif os.path.exists(default_path):
        DATA_PATH = default_path
    else:
        st.warning("No dataset found. Run `python generate_data.py` first.")
        st.stop()

    st.markdown("---")
    st.subheader("⚙️ Settings")
    churn_days = st.slider("Churn threshold (days inactive)", 30, 180, 90, 10)
    churn_thresh = st.slider("Risk flag threshold", 0.3, 0.8, 0.5, 0.05)
    forecast_periods = st.slider("Forecast months", 3, 12, 6)
    st.markdown("---")
    st.caption("NMIMS · MPSTME Shirpur\nKunal Patel — 70012200155")


# ── Load & process data (cached) ───────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning data…")
def get_data(path, churn_days):
    df   = load_and_clean(path)
    cust = build_customer_features(df, churn_days)
    return df, cust

@st.cache_resource(show_spinner="Training ML models…")
def get_models(path, churn_days):
    df, cust = get_data(path, churn_days)
    X, y, feat_names, scaler = prepare_ml_features(cust)
    model, metrics = train_churn_model(X, y, feat_names)
    explainer, shap_vals = get_shap_values(model, X, feat_names)
    optimal_k, _ = find_optimal_k(X)
    km, cluster_labels = run_clustering(X, optimal_k)
    seg_names = label_segments(cluster_labels)
    churn_proba, churn_flags = predict_churn(model, X)
    return (X, y, feat_names, scaler, model, metrics,
            explainer, shap_vals, cluster_labels, seg_names,
            churn_proba, churn_flags)

df, cust = get_data(DATA_PATH, churn_days)
(X, y, feat_names, scaler, model, metrics,
 explainer, shap_vals, cluster_labels, seg_names,
 churn_proba, churn_flags) = get_models(DATA_PATH, churn_days)

cust["churn_probability"] = churn_proba
cust["churn_flag"]        = churn_flags
cust["segment"]           = seg_names

monthly_rev = build_monthly_revenue(df)
fut_dates, fut_vals, lower, upper, mape = forecast_revenue(monthly_rev, forecast_periods)

# ── KPI computation ────────────────────────────────────────────────────────────
total_customers  = len(cust)
active_customers = int((cust["churn_flag"] == 0).sum())
churn_rate       = round(cust["churn"].mean() * 100, 1)
total_revenue    = df["total_order_value"].sum() if "total_order_value" in df.columns else 0
revenue_at_risk  = cust[cust["churn_flag"] == 1]["total_revenue"].sum()
avg_order_val    = round(df["total_order_value"].mean(), 2) if "total_order_value" in df.columns else 0

# ════════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 KPI Overview",
    "🚨 Churn Predictor",
    "📈 Revenue Forecast",
    "👥 Customer Segments",
    "🔍 Model Explainability",
])

# ── TAB 1 — KPI OVERVIEW ──────────────────────────────────────────────────────
with tab1:
    st.markdown("## 📊 Dataset Overview")
    st.caption("High-level view of customer base health and trends.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-title">Total Customers</div>
            <div class="kpi-value">{total_customers:,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        color = "#ef4444" if churn_rate > 30 else "#f59e0b"
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-title">Churn Rate</div>
            <div class="kpi-value" style="color:{color}">{churn_rate}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-title">Active Customers</div>
            <div class="kpi-value" style="color:#34d399">{active_customers:,}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-title">Revenue at Risk</div>
            <div class="kpi-value" style="color:#f87171">₹{revenue_at_risk:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Base & churn trend chart
    st.subheader("📉 Customer Base & Churn Trend")
    if "order_month" in df.columns:
        monthly_orders = df.groupby("order_month")["customer_id"].nunique().reset_index()
        monthly_orders.columns = ["month","active"]
        monthly_orders["month"] = monthly_orders["month"].astype(str)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_orders["month"], y=monthly_orders["active"],
                                 name="Active Customers", line=dict(color="#60a5fa", width=2),
                                 fill="tozeroy", fillcolor="rgba(96,165,250,0.08)"))
        fig.update_layout(template="plotly_dark", height=320,
                          margin=dict(l=10,r=10,t=10,b=10),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("🛍️ Revenue by Product Category")
        if "product_category_name" in df.columns:
            cat_rev = df.groupby("product_category_name")["total_order_value"].sum().sort_values(ascending=False).head(10)
            fig2 = go.Figure(go.Bar(x=cat_rev.values, y=cat_rev.index,
                                    orientation="h", marker_color="#818cf8"))
            fig2.update_layout(template="plotly_dark", height=350,
                               margin=dict(l=10,r=10,t=10,b=10), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.subheader("💳 Payment Method Distribution")
        if "payment_type" in df.columns:
            pay_dist = df["payment_type"].value_counts()
            fig3 = go.Figure(go.Pie(labels=pay_dist.index, values=pay_dist.values,
                                    hole=0.45,
                                    marker_colors=["#60a5fa","#34d399","#f59e0b","#f87171","#c084fc"]))
            fig3.update_layout(template="plotly_dark", height=350,
                               margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig3, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    with col_b:
        st.metric("Avg Order Value", f"₹{avg_order_val:,.2f}")
    with col_c:
        st.metric("Total Orders", f"{df['order_id'].nunique():,}")


# ── TAB 2 — CHURN PREDICTOR ───────────────────────────────────────────────────
with tab2:
    st.markdown("## 🚨 Churn Risk Dashboard")
    st.caption(f"Customers flagged as high-risk (churn probability ≥ {churn_thresh})")

    at_risk = cust[cust["churn_probability"] >= churn_thresh].copy()
    at_risk["risk_level"] = pd.cut(at_risk["churn_probability"],
                                   bins=[0,.55,.75,1.0],
                                   labels=["🟡 Medium","🟠 High","🔴 Critical"])
    at_risk = at_risk.sort_values("churn_probability", ascending=False)

    r1, r2, r3 = st.columns(3)
    r1.metric("At-Risk Customers", f"{len(at_risk):,}", delta=f"{len(at_risk)/total_customers*100:.1f}% of base", delta_color="inverse")
    r2.metric("Revenue at Risk", f"₹{at_risk['total_revenue'].sum():,.0f}", delta_color="inverse")
    r3.metric("Avg Churn Prob", f"{at_risk['churn_probability'].mean()*100:.1f}%", delta_color="inverse")

    st.markdown("---")
    st.subheader("📋 At-Risk Customer Table")

    display_cols = ["customer_id","churn_probability","risk_level","recency_days",
                    "order_count","avg_order_value","total_revenue","segment"]
    display_cols = [c for c in display_cols if c in at_risk.columns]

    show_df = at_risk[display_cols].head(200).copy()
    show_df["churn_probability"] = (show_df["churn_probability"]*100).round(1).astype(str) + "%"
    if "avg_order_value" in show_df.columns:
        show_df["avg_order_value"] = show_df["avg_order_value"].round(2)
    if "total_revenue" in show_df.columns:
        show_df["total_revenue"] = show_df["total_revenue"].round(2)

    st.dataframe(show_df, use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("📊 Churn Probability Distribution")
    fig_hist = px.histogram(cust, x="churn_probability", nbins=40,
                            color_discrete_sequence=["#818cf8"])
    fig_hist.add_vline(x=churn_thresh, line_dash="dash", line_color="#f87171",
                       annotation_text=f"Threshold: {churn_thresh}")
    fig_hist.update_layout(template="plotly_dark", height=300,
                           xaxis_title="Churn Probability", yaxis_title="# Customers",
                           margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("📦 Recency vs Revenue (Risk Map)")
    fig_scatter = px.scatter(cust, x="recency_days", y="total_revenue",
                             color="churn_probability", size="order_count",
                             color_continuous_scale="RdYlGn_r",
                             hover_data=["customer_id","segment"],
                             labels={"recency_days":"Days Since Last Purchase",
                                     "total_revenue":"Total Revenue (₹)"})
    fig_scatter.update_layout(template="plotly_dark", height=380,
                              margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)


# ── TAB 3 — REVENUE FORECAST ──────────────────────────────────────────────────
with tab3:
    st.markdown("## 📈 Revenue Forecast & Projections")

    f1, f2, f3 = st.columns(3)
    f1.metric("Forecast MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error — lower is better")
    f2.metric("Next Month Forecast", f"₹{fut_vals[0]:,.0f}")
    f3.metric(f"Total {forecast_periods}-Month Forecast", f"₹{fut_vals.sum():,.0f}")

    st.markdown("---")
    fig_fc = go.Figure()

    # Historical
    fig_fc.add_trace(go.Scatter(
        x=monthly_rev.index, y=monthly_rev.values,
        name="Historical Revenue", line=dict(color="#60a5fa", width=2.5)))

    # Confidence band
    fig_fc.add_trace(go.Scatter(
        x=list(fut_dates) + list(fut_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor="rgba(129,140,248,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Band"))

    # Forecast line
    fig_fc.add_trace(go.Scatter(
        x=fut_dates, y=fut_vals,
        name="Forecast", line=dict(color="#818cf8", width=2.5, dash="dot"),
        mode="lines+markers", marker=dict(size=7, color="#c084fc")))

    # Connect historical to forecast
    if len(monthly_rev) > 0:
        fig_fc.add_trace(go.Scatter(
            x=[monthly_rev.index[-1], fut_dates[0]],
            y=[monthly_rev.values[-1], fut_vals[0]],
            line=dict(color="#818cf8", width=1.5, dash="dot"),
            showlegend=False))

    fig_fc.update_layout(template="plotly_dark", height=420,
                         xaxis_title="Month", yaxis_title="Revenue (₹)",
                         legend=dict(orientation="h", y=1.08),
                         margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Monthly Forecast Table")
    fc_df = pd.DataFrame({
        "Month":         [d.strftime("%b %Y") for d in fut_dates],
        "Forecast (₹)":  [f"₹{v:,.0f}" for v in fut_vals],
        "Lower (₹)":     [f"₹{v:,.0f}" for v in lower],
        "Upper (₹)":     [f"₹{v:,.0f}" for v in upper],
    })
    st.dataframe(fc_df, use_container_width=True)

    if "product_category_name" in df.columns:
        st.markdown("---")
        st.subheader("📦 Revenue Trend by Category (top 5)")
        df["ym"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
        top5 = df.groupby("product_category_name")["total_order_value"].sum().nlargest(5).index
        cat_trend = df[df["product_category_name"].isin(top5)].groupby(
            ["ym","product_category_name"])["total_order_value"].sum().reset_index()
        fig_cat = px.line(cat_trend, x="ym", y="total_order_value",
                          color="product_category_name",
                          labels={"ym":"Month","total_order_value":"Revenue (₹)",
                                  "product_category_name":"Category"})
        fig_cat.update_layout(template="plotly_dark", height=350,
                              margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_cat, use_container_width=True)


# ── TAB 4 — CUSTOMER SEGMENTS ─────────────────────────────────────────────────
with tab4:
    st.markdown("## 👥 Customer Segmentation (K-Means)")

    seg_counts = pd.Series(seg_names).value_counts()
    sc1, sc2, sc3, sc4 = st.columns(4)
    colors_ = ["#818cf8","#34d399","#f59e0b","#f87171"]
    for idx, (col, (seg, cnt)) in enumerate(zip([sc1,sc2,sc3,sc4], seg_counts.items())):
        pct = cnt / len(seg_names) * 100
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-title" style="font-size:11px">{seg}</div>
            <div class="kpi-value" style="font-size:26px;color:{colors_[idx%4]}">{cnt:,}</div>
            <div class="kpi-delta">{pct:.1f}% of customers</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    cust2 = cust.copy()
    cust2["segment_label"] = seg_names

    col_pie, col_bar = st.columns(2)
    with col_pie:
        st.subheader("Segment Distribution")
        fig_pie = px.pie(values=seg_counts.values, names=seg_counts.index,
                         color_discrete_sequence=["#818cf8","#34d399","#f59e0b","#f87171"],
                         hole=0.4)
        fig_pie.update_layout(template="plotly_dark", height=340,
                               margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.subheader("Avg Order Value by Segment")
        seg_stats = cust2.groupby("segment_label")["avg_order_value"].mean().sort_values(ascending=False)
        fig_bar = px.bar(x=seg_stats.values, y=seg_stats.index,
                         orientation="h",
                         color=seg_stats.values,
                         color_continuous_scale="Viridis",
                         labels={"x":"Avg Order Value (₹)", "y":""})
        fig_bar.update_layout(template="plotly_dark", height=340,
                               margin=dict(l=10,r=10,t=10,b=10),
                               yaxis=dict(autorange="reversed"),
                               coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("📡 Radar Chart — Segment Profiles")

    radar_features = ["recency_score","order_count","avg_order_value",
                      "avg_installments","avg_review"]
    radar_features = [f for f in radar_features if f in cust2.columns]

    # Normalize for radar
    seg_means = cust2.groupby("segment_label")[radar_features].mean()
    seg_norm  = (seg_means - seg_means.min()) / (seg_means.max() - seg_means.min() + 1e-9)

    radar_fig = go.Figure()
    radar_colors = ["#818cf8","#34d399","#f59e0b","#f87171"]
    for i, seg in enumerate(seg_norm.index):
        vals = seg_norm.loc[seg].tolist()
        vals += [vals[0]]
        cats = radar_features + [radar_features[0]]
        radar_fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself", name=seg,
            line=dict(color=radar_colors[i % 4])))
    radar_fig.update_layout(template="plotly_dark", height=420,
                             polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                             margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🗺️ Segment Scatter Map")
    fig_seg = px.scatter(cust2, x="recency_days", y="avg_order_value",
                         color="segment_label", size="order_count",
                         hover_data=["customer_id","total_revenue"],
                         color_discrete_sequence=["#818cf8","#34d399","#f59e0b","#f87171"],
                         labels={"recency_days":"Days Since Last Purchase",
                                 "avg_order_value":"Avg Order Value (₹)",
                                 "segment_label":"Segment"})
    fig_seg.update_layout(template="plotly_dark", height=400,
                           margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_seg, use_container_width=True)


# ── TAB 5 — MODEL EXPLAINABILITY ──────────────────────────────────────────────
with tab5:
    st.markdown("## 🔍 Model Explainability (SHAP)")

    m1, m2, m3, m4, m5 = st.columns(5)
    def metric_card(col, title, val, color="#818cf8"):
        col.markdown(f"""<div class="metric-box">
            <div style="font-size:11px;color:#94a3b8;text-transform:uppercase">{title}</div>
            <div style="font-size:24px;font-weight:700;color:{color}">{val}%</div>
        </div>""", unsafe_allow_html=True)

    metric_card(m1, "Accuracy",  metrics["accuracy"],  "#60a5fa")
    metric_card(m2, "Precision", metrics["precision"], "#34d399")
    metric_card(m3, "Recall",    metrics["recall"],    "#f59e0b")
    metric_card(m4, "F1 Score",  metrics["f1"],        "#c084fc")
    metric_card(m5, "AUC-ROC",   metrics["auc_roc"],   "#f87171")

    st.markdown(f"**5-Fold CV AUC:** {metrics['cv_auc_mean']}% ± {metrics['cv_auc_std']}%")

    st.markdown("---")
    st.subheader("📊 Global Feature Importance (SHAP)")

    # Bar chart of mean |SHAP|
    mean_shap = np.abs(shap_vals).mean(axis=0)
    fi_df = pd.DataFrame({"feature": feat_names, "importance": mean_shap})
    fi_df = fi_df.sort_values("importance", ascending=True)

    fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h",
                    color="importance", color_continuous_scale="Purples",
                    labels={"importance":"Mean |SHAP value|", "feature":"Feature"})
    fig_fi.update_layout(template="plotly_dark", height=360,
                          margin=dict(l=10,r=10,t=10,b=10),
                          coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.subheader("🐝 SHAP Beeswarm Plot (Global)")
    fig_bsw, ax = plt.subplots(figsize=(10, 5))
    fig_bsw.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    shap.summary_plot(shap_vals, features=X, feature_names=feat_names,
                      show=False, plot_size=None, color_bar=True)
    plt.tight_layout()
    st.pyplot(fig_bsw)
    plt.close()

    st.markdown("---")
    st.subheader("🔎 Local Explanation — Individual Customer")

    cid_list = cust["customer_id"].tolist()
    sel_cid  = st.selectbox("Select a customer ID:", cid_list[:300])
    idx      = cust.index[cust["customer_id"] == sel_cid].tolist()

    if idx:
        i = idx[0]
        prob = cust.loc[i, "churn_probability"]
        flag = cust.loc[i, "churn_flag"]
        risk_color = "#ef4444" if prob >= 0.75 else "#f59e0b" if prob >= 0.5 else "#22c55e"
        badge = "🔴 CRITICAL" if prob >= 0.75 else "🟠 HIGH" if prob >= 0.5 else "🟢 LOW"

        st.markdown(f"""
        **Customer:** `{sel_cid}` &nbsp;|&nbsp;
        **Churn Probability:** <span style="color:{risk_color};font-weight:700">{prob*100:.1f}%</span> &nbsp;|&nbsp;
        **Risk Level:** {badge} &nbsp;|&nbsp;
        **Segment:** {cust.loc[i,'segment']}
        """, unsafe_allow_html=True)

        fig_wp, ax2 = plt.subplots(figsize=(10, 4))
        fig_wp.patch.set_facecolor("#0f1117")
        ax2.set_facecolor("#0f1117")
        exp = shap.Explanation(
            values=shap_vals[i],
            base_values=explainer.expected_value,
            data=X[i],
            feature_names=feat_names)
        shap.waterfall_plot(exp, show=False, max_display=10)
        plt.tight_layout()
        st.pyplot(fig_wp)
        plt.close()

    st.markdown("---")
    st.subheader("📋 Confusion Matrix")
    cm = np.array(metrics["conf_matrix"])
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=["Predicted: Retained","Predicted: Churned"],
        y=["Actual: Retained","Actual: Churned"],
        colorscale="Blues", text=cm.astype(str),
        texttemplate="%{text}", textfont={"size":18}))
    fig_cm.update_layout(template="plotly_dark", height=320,
                          margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#475569;font-size:13px;padding:8px">
    InsightCast — Smart Business Analytics Dashboard &nbsp;|&nbsp;
    Kunal Patel (70012200155) &nbsp;|&nbsp;
    NMIMS · MPSTME Shirpur · 2025-26
</div>
""", unsafe_allow_html=True)
