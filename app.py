"""
app.py — Fashion Recommendation System
Run with: streamlit run app.py
"""

import os
import streamlit as st
from PIL import Image

from utils import (
    load_data,
    build_feature_matrix,
    build_knn_model,
    build_visual_feature_matrix,
    build_visual_knn,
    get_image_path,
    get_unique_values,
    recommend_three_outfits,
    find_similar_by_image,
    find_similar_items_by_id,
    OUTFIT_SLOTS,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FashionAI · Style Recommender",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --cream:   #FAF7F2;
    --sand:    #EDE5D8;
    --clay:    #C4A882;
    --umber:   #7A5C3A;
    --ink:     #1C1410;
    --rose:    #C97B5E;
    --sage:    #7A9E85;
    --card:    #FFFFFF;
    --r:       14px;
    --shadow:  0 2px 16px rgba(28,20,16,0.09);
    --shadow-h:0 8px 32px rgba(28,20,16,0.16);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--cream) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}
[data-testid="stSidebar"] {
    background: var(--sand) !important;
    border-right: 2px solid var(--clay);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stMainBlockContainer"] { padding-top: 1.5rem !important; }

/* ──── HERO ──── */
.hero {
    background: linear-gradient(130deg, #3d2010 0%, #7A5C3A 45%, #C4A882 100%);
    border-radius: 20px;
    padding: 44px 44px 38px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.hero h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem;
    font-weight: 300;
    color: #fff;
    margin: 0 0 8px;
    letter-spacing: -0.3px;
}
.hero p { color: rgba(255,255,255,0.78); font-size: 1rem; margin: 0; font-weight: 300; }

/* ──── OUTFIT CARD ──── */
.outfit-wrapper {
    background: var(--card);
    border-radius: 20px;
    box-shadow: var(--shadow);
    padding: 0 0 20px;
    margin-bottom: 32px;
    overflow: hidden;
}
.outfit-header {
    background: linear-gradient(90deg, var(--umber) 0%, var(--clay) 100%);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.outfit-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.45rem;
    font-weight: 400;
    color: #fff;
    margin: 0;
}
.outfit-desc {
    color: rgba(255,255,255,0.72);
    font-size: 0.82rem;
    margin: 0;
    font-style: italic;
}
.outfit-body { padding: 20px 20px 0; }

/* ──── SLOT LABEL ──── */
.slot-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--umber);
    margin: 16px 0 8px;
    padding-left: 2px;
}

/* ──── ITEM CARD ──── */
.item-card {
    background: var(--card);
    border-radius: var(--r);
    overflow: hidden;
    box-shadow: var(--shadow);
    transition: transform .2s ease, box-shadow .2s ease;
    border: 1px solid var(--sand);
}
.item-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-h);
}
.card-body { padding: 12px 14px 14px; }
.card-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--ink);
    margin: 0 0 5px;
    line-height: 1.3;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.card-meta { font-size: 0.74rem; color: #7a6050; margin: 2px 0 0; }
.chip {
    display: inline-block;
    background: var(--sand);
    color: var(--umber);
    border-radius: 30px;
    padding: 2px 9px;
    font-size: 0.69rem;
    font-weight: 500;
    margin-top: 8px;
}
.chip-score {
    float: right;
    background: var(--umber);
    color: #fff;
    border-radius: 30px;
    padding: 2px 9px;
    font-size: 0.69rem;
    font-weight: 600;
    margin-top: 8px;
}

/* ──── NO IMAGE ──── */
.no-img {
    width: 100%;
    height: 210px;
    background: linear-gradient(140deg, var(--sand), #d5c9bb);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.4rem;
}

/* ──── TABS ──── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid var(--sand) !important;
}
[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    color: var(--umber) !important;
    padding: 10px 26px !important;
}
[aria-selected="true"] {
    color: var(--rose) !important;
    border-bottom: 2px solid var(--rose) !important;
}

/* ──── BUTTONS ──── */
.stButton > button {
    background: linear-gradient(130deg, var(--umber), var(--clay)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 28px !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.3px !important;
    transition: opacity .18s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ──── SIDEBAR ──── */
.sidebar-brand {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 400;
    color: var(--umber);
    margin: 0 0 2px;
}
.sidebar-tagline { font-size: 0.8rem; color: #8a6d52; margin-bottom: 18px; }

/* ──── EMPTY STATE ──── */
.empty-state {
    text-align: center;
    padding: 60px 24px;
}
.empty-state .e-icon { font-size: 3.2rem; margin-bottom: 14px; }
.empty-state h3 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 400;
    color: var(--umber);
    margin: 0 0 8px;
}
.empty-state p { font-size: 0.86rem; color: #8a6d52; }

/* ──── IMAGE UPLOAD ZONE ──── */
.upload-zone {
    border: 2px dashed var(--clay);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    background: rgba(196,168,130,0.06);
    margin-bottom: 20px;
}
.upload-zone p { color: var(--umber); font-size: 0.9rem; margin: 0; }

/* ──── RESULTS HEADER ──── */
.results-header {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    font-weight: 400;
    color: var(--umber);
    border-left: 4px solid var(--clay);
    padding-left: 14px;
    margin: 24px 0 16px;
}

/* ──── MISC ──── */
hr { border-color: var(--sand) !important; margin: 20px 0 !important; }
.stSelectbox label, .stTextInput label, .stSlider label, .stFileUploader label {
    font-weight: 500 !important;
    color: var(--umber) !important;
    font-size: 0.84rem !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

CSV_PATH = r"C:\Users\osama\OneDrive\Documents\FAI\Python Projects\Level 3\Applied Machine Learning\Cluade project\styles.csv"
IMG_DIR  = r"C:\Users\osama\OneDrive\Documents\FAI\Python Projects\Level 3\Applied Machine Learning\Cluade project\images"

SLOT_ICONS = {
    "Topwear":     "👕",
    "Bottomwear":  "👖",
    "Footwear":    "👟",
    "Accessories": "⌚",
}

OUTFIT_GRADIENTS = [
    "linear-gradient(90deg, #3d2010 0%, #7A5C3A 100%)",
    "linear-gradient(90deg, #1a3a4a 0%, #4a7a8a 100%)",
    "linear-gradient(90deg, #2a3a1a 0%, #6a8a4a 100%)",
]

# ─────────────────────────────────────────────
# CACHED MODEL LOADERS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_load():
    return load_data(CSV_PATH)


@st.cache_data(show_spinner=False)
def cached_features(_df):
    return build_feature_matrix(_df)


@st.cache_resource(show_spinner=False)
def cached_knn(_feat):
    return build_knn_model(_feat, n_neighbors=30)


@st.cache_data(show_spinner=False)
def cached_visual(_df):
    return build_visual_feature_matrix(_df, IMG_DIR)


@st.cache_resource(show_spinner=False)
def cached_visual_knn(_vis_mat):
    return build_visual_knn(_vis_mat, n_neighbors=20)


# ─────────────────────────────────────────────
# CARD RENDERING
# ─────────────────────────────────────────────

def render_no_img():
    return '<div class="no-img">👗</div>'


def display_item_card(col, item: dict, show_score: bool = False, img_height: int = 210):
    img_p  = item.get("image_path")
    name   = item.get("productDisplayName", "")[:55]
    sub    = item.get("subCategory", "")
    colour = item.get("baseColour", "")
    art    = item.get("articleType", "")
    score  = item.get("similarity", None)
    usage  = item.get("usage", "")

    score_html = ""
    if show_score and score is not None:
        score_html = f'<span class="chip-score">{score}%</span>'

    with col:
        st.markdown('<div class="item-card">', unsafe_allow_html=True)

        if img_p and os.path.exists(img_p):
            try:
                img = Image.open(img_p).convert("RGB")
                st.image(img, use_container_width=True)
            except Exception:
                st.markdown(render_no_img(), unsafe_allow_html=True)
        else:
            st.markdown(render_no_img(), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card-body">
            <p class="card-name">{name}</p>
            <p class="card-meta">🎨 {colour} · {art}</p>
            <p class="card-meta">📌 {usage}</p>
            <span class="chip">{sub}</span>{score_html}
        </div>
        </div>
        """, unsafe_allow_html=True)


def render_item_grid(items: list, show_score: bool = False, n_cols: int = 4):
    for i in range(0, len(items), n_cols):
        chunk = items[i:i + n_cols]
        cols = st.columns(n_cols)
        for col, item in zip(cols, chunk):
            display_item_card(col, item, show_score=show_score)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar(df):
    with st.sidebar:
        st.markdown('<p class="sidebar-brand">👗 FashionAI</p>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-tagline">Intelligent Style Curator</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### ✦ Outfit Preferences")

        genders = get_unique_values(df, "gender")
        seasons = get_unique_values(df, "season")
        usages  = get_unique_values(df, "usage")
        colours = get_unique_values(df, "baseColour")

        gender = st.selectbox("Gender",  genders,
                              index=genders.index("Men") if "Men" in genders else 0)
        season = st.selectbox("Season",  seasons,
                              index=seasons.index("Summer") if "Summer" in seasons else 0)
        usage  = st.selectbox("Usage",   usages,
                              index=usages.index("Casual") if "Casual" in usages else 0)
        colour = st.selectbox("Preferred Color", colours,
                              index=colours.index("Blue") if "Blue" in colours else 0)

        st.markdown("---")
        st.markdown("#### ✦ Dataset Stats")
        c1, c2 = st.columns(2)
        c1.metric("Items",      f"{len(df):,}")
        c2.metric("Categories", df["subCategory"].nunique())
        c1.metric("Colours",    df["baseColour"].nunique())
        c2.metric("Types",      df["articleType"].nunique())

        st.markdown("---")
        st.caption("Built with Streamlit · scikit-learn · PIL")

    return gender, season, usage, colour


# ─────────────────────────────────────────────
# TAB 1 — OUTFIT GENERATOR (3 full looks)
# ─────────────────────────────────────────────

def render_outfit(outfit: dict, idx: int):
    """Render one complete outfit in a styled card."""
    gradient  = OUTFIT_GRADIENTS[idx % len(OUTFIT_GRADIENTS)]
    title     = outfit["theme_name"]
    desc      = outfit["description"]
    slots     = outfit["slots"]

    # Outfit wrapper header
    st.markdown(f"""
    <div style="background:{gradient}; border-radius:16px 16px 0 0; padding:16px 24px; margin-top:12px;">
        <p class="outfit-title" style="font-family:'Cormorant Garamond',serif;
           font-size:1.4rem;font-weight:400;color:#fff;margin:0 0 3px;">{title}</p>
        <p style="color:rgba(255,255,255,0.72);font-size:0.82rem;font-style:italic;margin:0;">{desc}</p>
    </div>
    <div style="background:#fff;border-radius:0 0 16px 16px;
         box-shadow:0 4px 20px rgba(28,20,16,0.10);padding:20px;margin-bottom:8px;">
    """, unsafe_allow_html=True)

    # Render each slot as a row
    for slot in ["Topwear", "Bottomwear", "Footwear", "Accessories"]:
        item = slots.get(slot)
        if not item:
            continue

        icon = SLOT_ICONS.get(slot, "✦")
        st.markdown(
            f'<p class="slot-label">{icon} {slot}</p>',
            unsafe_allow_html=True
        )

        col, _ = st.columns([1, 3])
        display_item_card(col, item, show_score=False)

    st.markdown("</div>", unsafe_allow_html=True)


def tab_outfit_generator(df, feat_matrix, encoders, gender, season, usage, colour):
    st.markdown("""
    <div class="hero">
        <h1>✦ Outfit Generator</h1>
        <p>Three complete, curated looks — each with a different colour strategy and style personality.</p>
    </div>
    """, unsafe_allow_html=True)

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown(
            f"**Profile:** {gender} · {season} · {usage} · {colour} &nbsp;"
            f"<span style='font-size:0.8rem;color:#8a6d52;'>(edit in sidebar ←)</span>",
            unsafe_allow_html=True
        )
    with col_btn:
        generate = st.button("🪄 Generate 3 Looks", use_container_width=True)

    # Session state
    if "outfits" not in st.session_state:
        st.session_state.outfits = None
    if "outfit_params" not in st.session_state:
        st.session_state.outfit_params = None

    params = (gender, season, usage, colour)

    if generate:
        with st.spinner("Curating 3 complete looks for you…"):
            outfits = recommend_three_outfits(
                df, feat_matrix, encoders,
                gender, season, usage, colour, IMG_DIR
            )
        st.session_state.outfits = outfits
        st.session_state.outfit_params = params

    outfits = st.session_state.outfits
    params_match = st.session_state.outfit_params == params

    if outfits and params_match:
        if not outfits:
            st.warning("No items matched your filters. Try a different colour or usage.")
            return

        # Display 3 outfits stacked
        for i, outfit in enumerate(outfits):
            render_outfit(outfit, i)

        st.markdown(
            '<p style="text-align:center;color:#8a6d52;font-size:0.8rem;margin-top:8px;">'
            '✦ Hit <b>Generate 3 Looks</b> again for fresh picks · '
            'Adjust preferences in the sidebar for a different vibe</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="e-icon">✨</div>
            <h3>Three looks, ready to curate</h3>
            <p>Set your preferences in the sidebar, then hit <strong>Generate 3 Looks</strong></p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 2 — SIMILAR ITEMS (image upload)
# ─────────────────────────────────────────────

def tab_similar_items(df, feat_matrix, knn, visual_matrix, valid_ids, visual_knn):
    st.markdown("""
    <div class="hero" style="background:linear-gradient(130deg,#1a3a2a 0%,#4a7a5a 50%,#8aa870 100%);">
        <h1>✦ Visual Search</h1>
        <p>Upload any clothing photo — from the dataset or anywhere — and find visually similar items.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Search mode ──
    mode = st.radio(
        "mode",
        ["📷 Upload Photo", "🔢 Search by Item ID"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("---")

    if mode == "📷 Upload Photo":
        _similar_by_upload(df, visual_matrix, valid_ids, visual_knn)
    else:
        _similar_by_id(df, feat_matrix, knn)


def _similar_by_upload(df, visual_matrix, valid_ids, visual_knn):
    if visual_matrix is None or len(valid_ids) == 0:
        st.warning(
            "No images found in the `images/` folder. "
            "Please add clothing images to use visual search."
        )
        return

    st.markdown("""
    <div class="upload-zone">
        <p>📸 Upload any clothing photo — product shots, outfit photos, screenshots, anything.</p>
        <p style="font-size:0.8rem;color:#8a6d52;margin-top:6px;">
            Works for images both <strong>inside and outside</strong> the dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a clothing image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    top_n = st.slider("Number of similar items", 4, 16, 8)

    if uploaded is not None:
        try:
            user_img = Image.open(uploaded).convert("RGB")
        except Exception as e:
            st.error(f"Could not open image: {e}")
            return

        # Show uploaded image
        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(user_img, caption="Your uploaded image", use_container_width=True)
        with col_info:
            st.markdown("**Searching for visually similar items…**")
            st.markdown(
                "The search uses colour histograms, HSV distributions, and dominant colour "
                "extraction to find the closest matches in the dataset — no deep learning required."
            )
            st.caption(f"Searching across {len(valid_ids):,} indexed images")

        with st.spinner("Analysing image and searching…"):
            results = find_similar_by_image(
                user_img, df, visual_matrix, valid_ids, visual_knn,
                IMG_DIR, top_n
            )

        if not results:
            st.info("No similar items found. Try a different image.")
            return

        st.markdown(
            f'<p class="results-header">✦ Top {len(results)} Visually Similar Items</p>',
            unsafe_allow_html=True
        )
        render_item_grid(results, show_score=True, n_cols=4)


def _similar_by_id(df, feat_matrix, knn):
    col1, col2 = st.columns([2, 1])
    with col1:
        item_id_input = st.text_input(
            "Enter Item ID",
            placeholder="e.g. 15970",
            help="The numeric id from your styles.csv dataset"
        )
    with col2:
        top_n = st.slider("Results", 4, 16, 8)

    if not item_id_input:
        st.caption("Enter an item ID to find similar products based on category, colour, and usage.")
        return

    item_id = item_id_input.strip()
    match = df[df["id"] == item_id]

    if match.empty:
        st.error(f"Item ID **{item_id}** not found. Check your `styles.csv`.")
        return

    row = match.iloc[0]

    # Reference item display
    st.markdown('<p class="results-header">✦ Reference Item</p>', unsafe_allow_html=True)
    ref_c1, ref_c2 = st.columns([1, 3])
    with ref_c1:
        img_p = get_image_path(item_id, IMG_DIR)
        if img_p and os.path.exists(img_p):
            st.image(Image.open(img_p).convert("RGB"), use_container_width=True)
        else:
            st.markdown('<div class="no-img">👗</div>', unsafe_allow_html=True)
    with ref_c2:
        st.markdown(f"**{row['productDisplayName']}**")
        st.markdown(f"🏷️ {row['articleType']}  ·  {row['subCategory']}")
        st.markdown(f"🎨 {row['baseColour']}  ·  📅 {row['season']}  ·  📌 {row['usage']}")
        st.markdown(f"👤 {row['gender']}")

    # Similar items
    st.markdown('<p class="results-header">✦ Similar Items</p>', unsafe_allow_html=True)
    with st.spinner("Finding similar items…"):
        similar = find_similar_items_by_id(df, feat_matrix, knn, item_id, IMG_DIR, top_n)

    if not similar:
        st.info("No similar items found.")
        return

    render_item_grid(similar, show_score=True, n_cols=4)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if not os.path.exists(CSV_PATH):
        st.error(
            f"**Dataset not found:** `{CSV_PATH}` is missing.\n\n"
            "Place `styles.csv` in the same directory as `app.py`, "
            "and clothing images in the `images/` folder.\n\n"
            "To generate demo data: `python generate_demo_data.py`"
        )
        st.stop()

    # ── Load data & build models ──
    with st.spinner("Loading dataset…"):
        df = cached_load()

    if df.empty:
        st.error("Dataset is empty. Check your `styles.csv`.")
        st.stop()

    with st.spinner("Building recommendation models…"):
        feat_matrix, encoders = cached_features(df)
        knn = cached_knn(feat_matrix)

    with st.spinner(f"Indexing images for visual search…"):
        visual_matrix, valid_ids = cached_visual(df)

    visual_knn = None
    if visual_matrix is not None and len(valid_ids) > 0:
        visual_knn = cached_visual_knn(visual_matrix)

    # ── Sidebar ──
    gender, season, usage, colour = render_sidebar(df)

    # ── Tabs ──
    tab1, tab2 = st.tabs(["✦ Outfit Generator", "✦ Visual Search"])

    with tab1:
        tab_outfit_generator(df, feat_matrix, encoders, gender, season, usage, colour)

    with tab2:
        tab_similar_items(df, feat_matrix, knn, visual_matrix, valid_ids, visual_knn)


if __name__ == "__main__":
    main()
