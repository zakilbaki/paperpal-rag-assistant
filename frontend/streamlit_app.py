import os
import pandas as pd
import streamlit as st
import requests
import time

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="PaperPal 2.0",
    page_icon="📄",
    layout="wide",
)

# -----------------------------------------------------
# STYLING
# -----------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(145deg, #0f2027, #203a43, #2c5364);
    color: #f5f5f5;
}

h1, h2, h3 {
    color: #f5f5f5 !important;
    font-weight: 700;
}

section[data-testid="stFileUploader"] > div {
    border: 2px dashed #00b4d8;
    border-radius: 10px;
    background: rgba(255,255,255,0.05);
}
section[data-testid="stFileUploader"] p {
    color: #e0e0e0 !important;
}

.stButton>button {
    background: linear-gradient(90deg, #00b4d8, #0077b6);
    color: white;
    border: none;
    font-weight: 600;
    border-radius: 8px;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #0096c7, #023e8a);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05);
    border-radius: 6px 6px 0 0;
    color: #cbd5e1;
    padding: 10px 20px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #00b4d8;
    color: #ffffff !important;
}

[data-testid="stProgressBar"] div div {
    background-color: #00b4d8 !important;
}

div.block-container {
    padding-top: 2rem;
    max-width: 900px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# BACKEND ENDPOINTS
# -----------------------------------------------------
UPLOAD_URL = f"{BACKEND_BASE_URL}/api/v1/papers/upload"
SUMMARIZE_URL = f"{BACKEND_BASE_URL}/api/v1/papers/summarize"
KEYWORDS_URL = f"{BACKEND_BASE_URL}/api/v1/papers/keywords"
HEALTH_URL = f"{BACKEND_BASE_URL}/api/v1/health/"

# -----------------------------------------------------
# HEADER
# -----------------------------------------------------
st.markdown("""
<div style='text-align:center; margin-bottom: 2rem;'>
    <h1 style='font-size:2.6rem;'>PaperPal 2.0</h1>
    <p style='color:#cbd5e1; font-size:1.1rem;'>AI-powered summarization and keyword extraction for research papers</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# UPLOAD SECTION
# -----------------------------------------------------
st.markdown("### Upload Your Paper")

uploaded_file = st.file_uploader("Choose a PDF (max 1 MB)", type=["pdf"])

if uploaded_file:
    if uploaded_file.size > 1 * 1024 * 1024:
        st.error("File too large (limit 1 MB).")
        st.stop()

@st.cache_data(show_spinner=False)
def upload_pdf_cached(file_obj):
    files = {"file": (file_obj.name, file_obj.getvalue(), file_obj.type)}
    r = requests.post(UPLOAD_URL, files=files, timeout=(10, 120))
    r.raise_for_status()
    return r.json()

if uploaded_file and st.button("Upload Paper", use_container_width=True):
    with st.spinner("Uploading..."):
        try:
            res = upload_pdf_cached(uploaded_file)
            paper_id = res.get("paper_id")
            st.session_state["paper_id"] = paper_id
            st.session_state["paper_name"] = uploaded_file.name
            st.success(f"Uploaded successfully: {uploaded_file.name}")
            st.caption(f"Paper ID: `{paper_id}`")
        except Exception as e:
            st.error(f"Upload failed: {e}")

if "paper_id" not in st.session_state:
    st.stop()

# -----------------------------------------------------
# TABS
# -----------------------------------------------------
tab1, tab2 = st.tabs(["Summarize", "Keywords"])

# -----------------------------------------------------
# SUMMARIZATION TAB
# -----------------------------------------------------
with tab1:
    st.subheader("Generate Summary")
    summary_type = st.radio("Summary level", ["short", "medium", "detailed"], horizontal=True)
    use_cache = st.toggle("Use cached summary", value=True)
    progress = st.progress(0)
    status = st.empty()

    def summarize_paper(pid, stype, cache=True):
        payload = {"paper_id": pid, "summary_type": stype, "use_cache": cache}
        r = requests.post(SUMMARIZE_URL, json=payload, timeout=(10, 400))
        r.raise_for_status()
        return r.json()

    if st.button("Generate Summary", use_container_width=True):
        try:
            for i in range(0, 95, 5):
                progress.progress(i)
                status.text(f"Summarizing... {i}%")
                time.sleep(0.15)
            res = summarize_paper(st.session_state["paper_id"], summary_type, use_cache)
            progress.progress(100)
            status.text("Done.")
            with st.container():
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.05);padding:20px;border-radius:10px;'>"
                    f"{res.get('summary','No summary returned.')}</div>",
                    unsafe_allow_html=True,
                )
            st.caption(
                f"Time: {res.get('duration_ms',0)/1000:.2f}s • "
                f"Chunks: {res.get('chunks',0)} • "
                f"{'Cached' if res.get('cached') else 'Fresh generation'}"
            )
        except Exception as e:
            st.error(f"Summarization failed: {e}")

# -----------------------------------------------------
# KEYWORDS TAB
# -----------------------------------------------------
with tab2:
    st.subheader("Keyword Extraction")
    top_k = st.slider("Number of top keywords", 5, 40, 15)
    if st.button("Extract Keywords", use_container_width=True):
        with st.spinner("Extracting..."):
            try:
                response = requests.post(
                    KEYWORDS_URL,
                    json={"paper_id": st.session_state["paper_id"], "top_k": top_k},
                    timeout=(10, 120),
                )
                response.raise_for_status()
                res = response.json()
                kws = res.get("keywords", [])
                if kws:
                    df = pd.DataFrame(kws).sort_values("score", ascending=False)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No keywords found.")
            except Exception as e:
                st.error(f"Extraction failed: {e}")

st.markdown("<hr><center style='color:#94a3b8;'>PaperPal 2.0 — FastAPI × Streamlit • 2026</center>", unsafe_allow_html=True)
