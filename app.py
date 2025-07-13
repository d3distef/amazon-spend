import pandas as pd, plotly.express as px, streamlit as st

st.set_page_config(page_title="Amazon Spend", layout="wide")

@st.cache_data
def load():
    return pd.read_csv("amazon_complete_categories.csv", parse_dates=["date"])

df = load()

# --- sidebar filters ----------------------------------------------------
cats = sorted(df["category"].unique())
ins  = sorted(df["insurance_category"].unique())
pick_cat = st.sidebar.multiselect("Shopping categories", cats, default=cats)
pick_ins = st.sidebar.multiselect("Insurance categories", ins, default=ins)
df = df[df["category"].isin(pick_cat) & df["insurance_category"].isin(pick_ins)]

# --- treemap ------------------------------------------------------------
agg = df.groupby(["category","items"])["Real Total"].sum().reset_index()
fig_tree = px.treemap(agg, path=["category","items"], values="Real Total",
                      title="Lifetime Spend (Category → Item)")
st.plotly_chart(fig_tree, use_container_width=True)

# --- monthly stacked area ----------------------------------------------
monthly = (df.groupby([pd.Grouper(key="date", freq="M"), "category"])
             ["Real Total"].sum().reset_index())
fig_area = px.area(monthly, x="date", y="Real Total", color="category",
                   title="Monthly Spend by Category")
st.plotly_chart(fig_area, use_container_width=True)
