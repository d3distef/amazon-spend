# app.py — Streamlit dashboard v2
# ───────────────────────────────────────────────────────────────────────
# Requirements (add to requirements.txt):
# streamlit
# pandas
# plotly
# prophet       # for forecast tab

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# Prophet is optional for local dev
try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False

st.set_page_config(page_title="Amazon Spend", layout="wide")

# ── 1 · LOAD & PREP ────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("amazon_complete_categories.csv", parse_dates=["date"])
    df["Real Total"] = df["Real Total"].astype(float).round(2)
    df["year"] = df["date"].dt.year
    df["month_name"] = df["date"].dt.month_name()
    df["month_num"] = df["date"].dt.month
    return df

df = load_data()

# ── 2 · SIDEBAR FILTERS ───────────────────────────────────────────────
st.sidebar.header("Filters")
cats = sorted(df["category"].unique())
incs = sorted(df["insurance_category"].unique())
pick_cat = st.sidebar.multiselect("Shopping categories", cats, default=cats)
pick_inc = st.sidebar.multiselect("Insurance categories", incs, default=incs)

df = df[df["category"].isin(pick_cat) & df["insurance_category"].isin(pick_inc)]

# ── 3 · LAYOUT TABS ───────────────────────────────────────────────────
tab_over, tab_deep, tab_heat, tab_pie, tab_fcst, tab_data = st.tabs(
    ["Overview 📊", "Deep Dive 🔍", "Heat maps 🔥", "Pie charts 🥧",
     "Forecast 📈", "Data / Download ⬇️"]
)

# ── 3·1 OVERVIEW TAB ──────────────────────────────────────────────────
with tab_over:
    st.subheader("Treemap – Lifetime Spend")
    agg_tree = df.groupby(["category", "items"])["Real Total"].sum().reset_index()
    st.plotly_chart(
        px.treemap(agg_tree, path=["category", "items"], values="Real Total"),
        use_container_width=True)

    st.divider()
    st.subheader("Monthly Spend by Category")
    monthly = (df.groupby([pd.Grouper(key="date", freq="M"), "category"])
                 ["Real Total"].sum().reset_index())
    st.plotly_chart(
        px.area(monthly, x="date", y="Real Total", color="category"),
        use_container_width=True)

# ── 3·2 DEEP-DIVE TAB ────────────────────────────────────────────────
with tab_deep:
    st.subheader("Sunburst – Shopping → Insurance → Item")
    st.plotly_chart(
        px.sunburst(df, path=["category", "insurance_category", "items"],
                    values="Real Total"),
        use_container_width=True)

    st.divider()
    st.subheader("Top-10 Items by Spend")
    top10 = (df.groupby("items")["Real Total"].sum()
               .nlargest(10).reset_index())
    fig_top = px.bar(top10, x="Real Total", y="items",
                     orientation="h", title="Top-10 Items")
    fig_top.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_top, use_container_width=True)

    st.divider()
    st.subheader("Annual % Mix")
    annual = (df.groupby(["year", "category"])["Real Total"].sum()
                .groupby(level=0, group_keys=False)
                .apply(lambda s: 100*s/s.sum())
                .reset_index(name="pct"))
    st.plotly_chart(
        px.area(annual, x="year", y="pct", color="category",
                groupnorm="fraction", labels={"pct": "% of year"}),
        use_container_width=True)

# ── 3·3 HEAT-MAP TAB ─────────────────────────────────────────────────
with tab_heat:
    sub1, sub2 = st.tabs(["Insurance vs Month", "Shopping vs Month"])
    order_months = ["January","February","March","April","May","June",
                    "July","August","September","October","November","December"]

    with sub1:
        st.subheader("Insurance Category vs Month")
        if len(df):
            heat = (df.groupby(["insurance_category","month_name"])
                      ["Real Total"].sum().reset_index())
            heat["month_name"] = pd.Categorical(heat["month_name"],
                                                categories=order_months,
                                                ordered=True)
            st.plotly_chart(
                px.density_heatmap(
                    heat, x="month_name", y="insurance_category",
                    z="Real Total", color_continuous_scale="Viridis"),
                use_container_width=True)
        else:
            st.info("No data for current filter selection.")

    with sub2:
        st.subheader("Shopping Category vs Month")
        if len(df):
            heat2 = (df.groupby(["category","month_name"])
                       ["Real Total"].sum().reset_index())
            heat2["month_name"] = pd.Categorical(heat2["month_name"],
                                                 categories=order_months,
                                                 ordered=True)
            st.plotly_chart(
                px.density_heatmap(
                    heat2, x="month_name", y="category", z="Real Total",
                    color_continuous_scale="Viridis"),
                use_container_width=True)
        else:
            st.info("No data for current filter selection.")

# ── 3·4 PIE-CHART TAB ────────────────────────────────────────────────
with tab_pie:
    st.subheader("Pie Charts – Spend Breakdown per Shopping Category")
    col1, col2 = st.columns(2)
    for i, cat in enumerate(sorted(df["category"].unique())):
        slice_df = df[df["category"] == cat]
        spend = slice_df.groupby("insurance_category")["Real Total"].sum()
        if spend.empty:
            continue
        pie = px.pie(spend.reset_index(), names="insurance_category",
                     values="Real Total", title=f"{cat} → Insurance split")
        (col1 if i % 2 == 0 else col2).plotly_chart(pie, use_container_width=True)

# ── 3·5 FORECAST TAB ─────────────────────────────────────────────────
with tab_fcst:
    st.subheader("12-month Spend Forecast (Prophet)")
    if not PROPHET_OK:
        st.warning("`prophet` package not installed – add it to requirements.txt")
    else:
        sel_cat = st.selectbox("Choose shopping category", cats, index=0)
        ts = (df[df["category"] == sel_cat]
                .groupby(pd.Grouper(key="date", freq="M"))["Real Total"]
                .sum().reset_index())
        ts = ts.rename(columns={"date": "ds", "Real Total": "y"})
        if len(ts) < 2 or ts["y"].sum() == 0:
            st.info("Not enough data to forecast.")
        else:
            m = Prophet(yearly_seasonality=True)
            m.fit(ts)
            future = m.make_future_dataframe(periods=12, freq="M")
            fcst = m.predict(future)
            fig_fcst = px.line(fcst, x="ds", y="yhat",
                               title=f"Forecast – {sel_cat}")
            fig_fcst.add_scatter(x=ts["ds"], y=ts["y"],
                                 mode="markers", name="Actual")
            st.plotly_chart(fig_fcst, use_container_width=True)

# ── 3·6 DATA / DOWNLOAD TAB ─────────────────────────────────────────
with tab_data:
    st.subheader("Data preview (first 1 000 rows)")
    st.dataframe(df.head(1000), use_container_width=True)

    st.divider()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered CSV",
                       csv_bytes, "amazon_complete_categories.csv",
                       "text/csv")
