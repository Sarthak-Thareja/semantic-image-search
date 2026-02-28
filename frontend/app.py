"""Streamlit UI for Multimodal Image-Text Retrieval."""

import base64
import os
from io import BytesIO
import requests
import streamlit as st

# Config: RETRIEVAL_API_URL (full URL) or RETRIEVAL_API_PORT (uses localhost)
def _resolve_api_url():
    if os.environ.get("RETRIEVAL_API_URL"):
        return os.environ["RETRIEVAL_API_URL"]
    port = os.environ.get("RETRIEVAL_API_PORT")
    if port:
        return f"http://localhost:{port}"
    return "http://localhost:8000"

API_URL = _resolve_api_url()
API_PREFIX = os.environ.get("RETRIEVAL_API_PREFIX", "/api/v1")
SEARCH_TEXT_URL = f"{API_URL}{API_PREFIX}/search-text"
SEARCH_IMAGE_URL = f"{API_URL}{API_PREFIX}/search-image"
HEALTH_URL = f"{API_URL}{API_PREFIX}/health"

st.set_page_config(
    page_title="Semantic Image Search",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Multimodal Semantic Image Search")
st.caption("Search images by natural language or upload an image to find similar ones")

# Check API health
def fetch_image_bytes(image_id: str) -> bytes | None:
    """Fetch image bytes from API; returns None on failure."""
    try:
        r = requests.get(f"{API_URL}{API_PREFIX}/images/{image_id}", timeout=10)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None


def check_health():
    try:
        r = requests.get(HEALTH_URL, timeout=60)  # Allow time for CLIP model load on cold start
        if r.status_code == 200:
            return r.json(), None
        return None, f"API returned {r.status_code}"
    except Exception as e:
        return None, str(e)


health, health_err = check_health()
if health_err:
    st.error(f"⚠️ API unavailable: {health_err}")
    st.info(f"Start the API: `./run.sh` or `uvicorn app.main:app --reload`\n\nIf API runs on port 8001: `RETRIEVAL_API_PORT=8001 streamlit run frontend/app.py`")
    st.stop()

status = health.get("status", "unknown")
vector_connected = health.get("vector_db_connected", False)
stats = health.get("index_stats", {})
total = stats.get("total_vectors", 0)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("API Status", status)
with col2:
    st.metric("Vector DB", "Connected" if vector_connected else "Disconnected")
with col3:
    st.metric("Indexed Images", total)

if total == 0:
    st.warning("No images indexed. Run the indexing script to populate the database.")

# Tabs for search modes
tab_text, tab_image = st.tabs(["📝 Text Search", "🖼️ Image Search"])

with tab_text:
    st.subheader("Search by text")
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., dog playing in snow",
        key="text_query",
    )
    top_k = st.slider("Number of results", 1, 50, 10, key="text_top_k")
    if st.button("Search", key="btn_text"):
        if not query or not query.strip():
            st.error("Please enter a search query")
        else:
            with st.spinner("Searching..."):
                try:
                    r = requests.post(
                        SEARCH_TEXT_URL,
                        json={"query": query.strip(), "top_k": top_k},
                        timeout=30,
                    )
                    r.raise_for_status()
                    data = r.json()
                    results = data.get("results", [])
                    latency = data.get("latency_ms", 0)
                    st.success(f"Found {len(results)} results in {latency:.0f}ms")
                    if results:
                        ncols = min(5, len(results))
                        cols = st.columns(ncols)
                        for i, res in enumerate(results):
                            c = cols[i % ncols]
                            meta = res.get("metadata", {})
                            caption = meta.get("caption", meta.get("path", res["id"])) or ""
                            score = res.get("score", 0)
                            img_bytes = None
                            if res.get("image_base64"):
                                try:
                                    img_bytes = base64.b64decode(res["image_base64"])
                                except Exception:
                                    pass
                            if not img_bytes:
                                img_bytes = fetch_image_bytes(res["id"])
                            if img_bytes:
                                try:
                                    # Use BytesIO + img tag for reliable display
                                    b64 = base64.b64encode(img_bytes).decode()
                                    c.markdown(f'<img src="data:image/jpeg;base64,{b64}" style="max-width:100%;border-radius:8px;">', unsafe_allow_html=True)
                                    c.caption(f"{score:.2f} | {(caption[:40] + '...') if len(caption) > 40 else caption}")
                                except Exception:
                                    c.write(f"**{res['id']}** — {score:.2f}")
                            else:
                                c.write(f"**{res['id']}** — {score:.2f}")
                                c.caption(caption[:60])
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab_image:
    st.subheader("Search by image")
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="img_upload",
    )
    top_k_img = st.slider("Number of results", 1, 50, 10, key="img_top_k")
    if uploaded and st.button("Find similar images", key="btn_img"):
        with st.spinner("Searching..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                data = {"top_k": top_k_img}
                r = requests.post(
                    SEARCH_IMAGE_URL,
                    files=files,
                    data=data,
                    timeout=30,
                )
                r.raise_for_status()
                resp = r.json()
                results = resp.get("results", [])
                latency = resp.get("latency_ms", 0)
                st.success(f"Found {len(results)} similar images in {latency:.0f}ms")
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.image(uploaded, caption="Your image", use_container_width=True)
                with col_b:
                    if results:
                        ncols = min(4, len(results))
                        cols = st.columns(ncols)
                        for i, res in enumerate(results):
                            c = cols[i % ncols]
                            meta = res.get("metadata", {})
                            caption = meta.get("caption", meta.get("path", res["id"])) or ""
                            score = res.get("score", 0)
                            img_bytes = None
                            if res.get("image_base64"):
                                try:
                                    img_bytes = base64.b64decode(res["image_base64"])
                                except Exception:
                                    pass
                            if not img_bytes:
                                img_bytes = fetch_image_bytes(res["id"])
                            if img_bytes:
                                try:
                                    b64 = base64.b64encode(img_bytes).decode()
                                    c.markdown(f'<img src="data:image/jpeg;base64,{b64}" style="max-width:100%;border-radius:8px;">', unsafe_allow_html=True)
                                    c.caption(f"{score:.2f}")
                                except Exception:
                                    c.write(f"**{res['id']}** — {score:.2f}")
                            else:
                                c.write(f"**{res['id']}** — {score:.2f}")
                                c.caption(caption[:50])
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
