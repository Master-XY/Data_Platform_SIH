# streamlit_app.py
import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="MVP Demo", layout="wide")

@st.cache_data
def load_data():
    # adjust path if needed
    return pd.read_parquet("data/processed/unified.parquet")

def image_exists(path):
    return isinstance(path, str) and path and os.path.exists(path)

st.title("SIH MVP — Samples")
df = load_data()
st.write("Total samples:", len(df))

# defensive: ensure sample_id column exists
if "sample_id" not in df.columns:
    st.error("Unified dataset missing `sample_id` column.")
    st.stop()

sel = st.selectbox("Choose sample", df["sample_id"].tolist())
row = df[df["sample_id"] == sel].iloc[0]

# Safe image handling
image_path = row.get("thumbnail_path")
if image_exists(image_path):
    try:
       st.image(image_path, caption=sel, width=300)

    except Exception as e:
        st.warning(f"Failed to load image for {sel}: {e}")
        st.write("Path tried:", image_path)
else:
    st.info("No thumbnail available for this sample.")
    # optionally show a placeholder (simple coloured box)
    st.markdown("**Placeholder image**")

# show metadata table
meta_cols = ["collection_date", "latitude", "longitude", "depth_m", "species"]
meta = {c: row[c] for c in meta_cols if c in row.index}
st.table(pd.DataFrame([meta]))

# placeholder for ASV chart
st.write("Top ASV (placeholder)")
