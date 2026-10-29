import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import streamlit as st
from src.retail_ai.data import default_data_path, load_transactions

st.set_page_config(page_title="Data Explorer", page_icon="🧾", layout="wide")
st.title("🧾 Data Explorer")
if default_data_path().exists():
    df = load_transactions(default_data_path())
    st.write(f"Rows: {len(df):,} · Columns: {len(df.columns)}")
    st.dataframe(df.head(500), use_container_width=True, hide_index=True)
    st.download_button("Download demo data", data=df.to_csv(index=False), file_name="retail_transactions.csv", mime="text/csv")
else:
    st.warning("Run `python generate_data.py` first.")
