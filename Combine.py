import pandas as pd, numpy as np, textwrap

cat_csv = "amazon_categorized.csv"                  # shopping cats
ins_csv = "amazon_categorized_with_insurance.csv"   # insurance cats

cat_df = pd.read_csv(cat_csv)
ins_df = pd.read_csv(ins_csv)

def clean(df):
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")
    df = df.rename(columns={" Real Total ": "Real Total"})
    # numeric
    df["Real Total"] = (df["Real Total"].astype(str)
                        .str.replace(r"[\$,]", "", regex=True)
                        .str.strip()
                        .replace({"-": np.nan, "": np.nan})
                        .astype(float).round(2))
    # date to YYYY-MM-DD for consistency
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

cat_df, ins_df = clean(cat_df), clean(ins_df)

# keep unique keys in insurance file (ID+date+amount)
key_cols = ["ID", "date", "Real Total"]
ins_df = ins_df.drop_duplicates(key_cols, keep="first")

merged = pd.merge(
    cat_df,                                   # keeps ALL shopping rows
    ins_df[key_cols + ["insurance_category"]],
    on=key_cols,
    how="left",       # many-to-one: m:1
    validate="m:1"
)

# diagnostics
total_rows = len(cat_df)
matched    = merged["insurance_category"].notna().sum()
print(textwrap.dedent(f"""
    Shopping rows : {total_rows}
    Matched rows  : {matched}
    Still NaN     : {total_rows - matched}
"""))

merged.to_csv("amazon_complete_categories.csv", index=False)
print("✅ wrote amazon_complete_categories.csv")
