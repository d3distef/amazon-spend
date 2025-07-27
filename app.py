# app.py – Amazon spend dashboard
# ───────────────────────────────────────────────────────────────────────
# Author: you 😊   Run:  streamlit run app.py
# CSV expected at C:\amazon-spend\Official_categorized_purchase_history.csv
# Columns: see README or sample screenshot

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pathlib import Path

# Prophet is heavy – import lazily for fast reloads
from functools import lru_cache

st.set_page_config(page_title="Amazon Spend Dashboard", layout="wide")

# ───────────────────────────────────────────────────────────────────────
# 1 · LOAD  +  PREP
# ───────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading CSV …")
def load_data() -> pd.DataFrame:
    df = pd.read_csv("Official_categorized_purchase_history.csv")

    # ── clean‑up columns ──────────────────
    df["Order Date"] = pd.to_datetime(df["Order Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Order Date"])

    df["Total"] = pd.to_numeric(df["Total"], errors="coerce").fillna(0).round(2)

    df["Year"]       = df["Order Date"].dt.year
    df["MonthNum"]   = df["Order Date"].dt.month
    df["Month name"] = df["Order Date"].dt.month_name()

    df["Insurance_Cats"] = df["Insurance_Cats"].astype(str).str.strip()
    df["Gen_Cats"]       = df["Gen_Cats"].astype(str).str.strip()
    df["Room"]           = df["Room"].astype(str).str.strip()
    return df

df = load_data()          # ← no argument

# ───────────────────────────────────────────────────────────────────────
# 2 · SIDEBAR FILTERS
# ───────────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

years = sorted(df["Year"].unique())
gen_all  = sorted(df["Gen_Cats"].unique())
room_all = sorted(df["Room"].unique())

pick_year = st.sidebar.slider("Year range", int(min(years)), int(max(years)),
                              (int(min(years)), int(max(years))))
pick_gen  = st.sidebar.multiselect("Gen_Cats (shopping)", gen_all, default=gen_all)
pick_room = st.sidebar.multiselect("Room", room_all, default=room_all)

mask = (
    df["Year"].between(pick_year[0], pick_year[1]) &
    df["Gen_Cats"].isin(pick_gen) &
    df["Room"].isin(pick_room)
)
df_filt = df.loc[mask].copy()

# ───────────────────────────────────────────────────────────────────────
# 3 · EXECUTIVE METRICS
# ───────────────────────────────────────────────────────────────────────
colA, colB, colC = st.columns(3)
colA.metric("Rows (after filter)", f"{len(df_filt):,}")
colB.metric("Total spend $", f"${df_filt['Total'].sum():,.2f}")
colC.metric("Avg. per item $", f"${df_filt['Total'].mean():,.2f}")

st.divider()

# ───────────────────────────────────────────────────────────────────────
# 4 · TABS
# ───────────────────────────────────────────────────────────────────────
tab_ins, tab_shop, tab_year, tab_heat, tab_pie, tab_fcst, tab_data = st.tabs(
    ["Insurance Treemap 🏠",
     "Shopping Sunburst 🛒",
     "Year‑on‑Year 📊",
     "Heat maps 🔥",
     "Pie charts 🥧",
     "Forecast (Prophet) 📈",
     "Data / Download ⬇️"]
)

# ── 4·1 INSURANCE TREEMAP ────────────────────────────────────────────
with tab_ins:
    st.subheader("Room → Insurance_Cats → Product treemap")
    if df_filt.empty:
        st.info("No data for current filters.")
    else:
        tree = (df_filt.groupby(["Room","Insurance_Cats","Product Name"])
                       ["Total"].sum().reset_index())
        fig_tree = px.treemap(tree,
                              path=["Room","Insurance_Cats","Product Name"],
                              values="Total",
                              color="Room",
                              hover_data={"Total":":.2f"})
        st.plotly_chart(fig_tree, use_container_width=True)
    st.divider()
    st.subheader("Gen_Cats → Product treemap")
    if not df_filt.empty:
        shop_tree = (df_filt.groupby(["Gen_Cats", "Product Name"])["Total"]
                               .sum().reset_index())
        fig_shop = px.treemap(shop_tree,
                              path=["Gen_Cats", "Product Name"],
                              values="Total",
                              color="Gen_Cats",
                              hover_data={"Total":":.2f"})
        st.plotly_chart(fig_shop, use_container_width=True)

# ── 4·2 SHOPPING SUNBURST ────────────────────────────────────────────
with tab_shop:
    st.subheader("Gen_Cats → Insurance_Cats → Product sunburst")
    if df_filt.empty:
        st.info("No data for current filters.")
    else:
        sun = (df_filt.groupby(["Gen_Cats","Insurance_Cats","Product Name"])
                      ["Total"].sum().reset_index())
        fig_sun = px.sunburst(sun,
                              path=["Gen_Cats","Insurance_Cats","Product Name"],
                              values="Total")
        st.plotly_chart(fig_sun, use_container_width=True)

    st.divider()
    st.subheader("Top‑15 items by lifetime spend")
    if not df_filt.empty:
        top15 = (df_filt.groupby("Product Name")["Total"].sum()
                           .nlargest(15).reset_index())
        fig_top = px.bar(top15, x="Total", y="Product Name",
                         orientation="h", text_auto=".2f")
        fig_top.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top, use_container_width=True)

# ── 4·3 YEAR‑ON‑YEAR COMPARISON ──────────────────────────────────────
with tab_year:
    if df_filt.empty:
        st.info("No data for current filters.")
    else:
        yo, y_pct = st.tabs(["$ spend by year", "% mix by year"])

        with yo:
            st.subheader("Stacked‑bar — annual spend by Gen_Cats")
            g = (df_filt.groupby(["Year","Gen_Cats"])["Total"].sum()
                       .reset_index())
            fig_bar = px.bar(g, x="Year", y="Total", color="Gen_Cats",
                             text_auto=".0f", title="Annual spend ($)")
            st.plotly_chart(fig_bar, use_container_width=True)

        with y_pct:
            st.subheader("Area — annual % mix (Gen_Cats)")
            g2 = (df_filt.groupby(["Year","Gen_Cats"])["Total"].sum()
                          .reset_index())
            g2["pct"] = g2.groupby("Year")["Total"].transform(
                            lambda s: 100*s/s.sum())
            fig_pct = px.area(g2, x="Year", y="pct", color="Gen_Cats",
                              groupnorm="fraction", labels={"pct":"%"})
            st.plotly_chart(fig_pct, use_container_width=True)

# ── 4·4 HEAT MAP TAB ────────────────────────────────────────────────
with tab_heat:
    order_mo = ["January","February","March","April","May","June",
                "July","August","September","October","November","December"]

    st.subheader("Insurance_Cats vs Month heat‑map")
    if df_filt.empty:
        st.info("No data for current filters.")
    else:
        h = (df_filt.groupby(["Insurance_Cats","Month name"])["Total"].sum()
                      .reset_index())
        h["Month name"] = pd.Categorical(h["Month name"], categories=order_mo,
                                         ordered=True)
        fig_h = px.density_heatmap(
            h, x="Month name", y="Insurance_Cats", z="Total",
            color_continuous_scale="Viridis")
        st.plotly_chart(fig_h, use_container_width=True)

# ── 4·5 PIE CHARTS ──────────────────────────────────────────────────
with tab_pie:
    st.subheader("Per‑Gen_Cats spend split by Insurance_Cats")
    col1, col2 = st.columns(2)
    for i, g in enumerate(sorted(df_filt["Gen_Cats"].unique())):
        slice_ = df_filt[df_filt["Gen_Cats"] == g]
        pies = slice_.groupby("Insurance_Cats")["Total"].sum()
        if pies.empty: continue
        fig_p = px.pie(pies.reset_index(), names="Insurance_Cats",
                       values="Total", title=g)
        (col1 if i%2==0 else col2).plotly_chart(fig_p, use_container_width=True)

# ── 4·6 FORECAST (PROPHET) ──────────────────────────────────────────
with tab_fcst:
    st.subheader("12‑month spend forecast (Prophet)")

    if df_filt.empty:
        st.info("No data for current filters.")

    else:
        # -------- choose category & build monthly TS --------
        sel_gen = st.selectbox(
            "Choose Gen_Cats for forecast",
            sorted(df_filt["Gen_Cats"].unique())
        )

        ts = (df_filt[df_filt["Gen_Cats"] == sel_gen]
                .groupby(pd.Grouper(key="Order Date", freq="M"))["Total"]
                .sum().reset_index()
              )
        ts = ts.rename(columns={"Order Date": "ds", "Total": "y"})
        ts["ds"] = ts["ds"].dt.tz_localize(None)     #  ← remove UTC tz‑info

        # -------- guardrail: need enough history --------
        if len(ts) < 6 or ts["y"].sum() == 0:
            st.info("Need ≥ 6 months of non‑zero data for a stable forecast.")

        else:
            # -------- Prophet fit (cached by category) --------
            @st.cache_data(show_spinner="Fitting Prophet …")
            def do_prophet(gen_name: str, df_in: pd.DataFrame):
                from prophet import Prophet
                m = Prophet(yearly_seasonality=True,
                            weekly_seasonality=False,
                            daily_seasonality=False)
                m.fit(df_in)
                future = m.make_future_dataframe(periods=12, freq="M")
                return m.predict(future)

            fc = do_prophet(sel_gen, ts)

            # -------- plot --------
            fig_f = px.line(fc, x="ds", y="yhat",
                            title=f"{sel_gen} | 12‑month forecast")
            fig_f.add_scatter(x=ts["ds"], y=ts["y"],
                              mode="markers+lines", name="Actual")
            st.plotly_chart(fig_f, use_container_width=True)
# ── 4·7 DATA / DOWNLOAD ─────────────────────────────────────────────
with tab_data:
    st.subheader("Preview (first 1 000 rows)")
    st.dataframe(df_filt.head(1000), use_container_width=True)

    st.divider()
    st.download_button(
        "⬇️ Download filtered CSV",
        df_filt.to_csv(index=False).encode("utf-8"),
        file_name="filtered_purchase_history.csv",
        mime="text/csv"
    )
