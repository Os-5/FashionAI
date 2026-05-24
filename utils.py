"""
utils.py — ML logic for Fashion Recommendation System
Features:
  - 3 distinct full outfits with internal colour/style coherence
  - Visual image similarity via colour histogram + HSV + dominant colour features
  - Works on ANY uploaded image (not just dataset images)
  - KNN + cosine similarity on encoded categorical features
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_data(csv_path: str = "styles.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    required = ["id", "gender", "masterCategory", "subCategory",
                "articleType", "baseColour", "season", "usage", "productDisplayName"]
    df = df.dropna(subset=required)
    df["id"] = df["id"].astype(str)
    for col in ["gender", "masterCategory", "subCategory", "articleType",
                "baseColour", "season", "usage"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    return df.reset_index(drop=True)


def get_image_path(item_id: str, images_dir: str = "images"):
    for ext in ["jpg", "jpeg", "png", "webp"]:
        p = os.path.join(images_dir, f"{item_id}.{ext}")
        if os.path.exists(p):
            return p
    return None


def get_unique_values(df: pd.DataFrame, col: str) -> list:
    return sorted(df[col].dropna().unique().tolist())


# ─────────────────────────────────────────────
# CATEGORICAL FEATURE ENCODING
# ─────────────────────────────────────────────

FEATURE_COLS = ["gender", "season", "usage", "baseColour",
                "articleType", "subCategory", "masterCategory"]


@st.cache_data
def build_feature_matrix(df: pd.DataFrame):
    encoders = {}
    encoded = np.zeros((len(df), len(FEATURE_COLS)), dtype=np.float32)
    for i, col in enumerate(FEATURE_COLS):
        le = LabelEncoder()
        encoded[:, i] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return encoded, encoders


@st.cache_resource
def build_knn_model(feature_matrix: np.ndarray, n_neighbors: int = 30):
    k = min(n_neighbors, len(feature_matrix))
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(feature_matrix)
    return knn


def encode_query(query: dict, encoders: dict) -> np.ndarray:
    vec = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
    for i, col in enumerate(FEATURE_COLS):
        le = encoders[col]
        val = str(query.get(col, "")).strip().title()
        vec[0, i] = le.transform([val])[0] if val in le.classes_ else 0.0
    return vec


# ─────────────────────────────────────────────
# OUTFIT SLOTS + COLOUR HARMONY
# ─────────────────────────────────────────────

OUTFIT_SLOTS = {
    "Topwear":     ["Topwear", "Shirts", "Tshirts", "Tops", "Blouses",
                    "Sweatshirts", "Jackets", "Sweaters", "Kurtas",
                    "Tunics", "Dresses", "Jumpsuits", "Nehru Jackets"],
    "Bottomwear":  ["Bottomwear", "Jeans", "Trousers", "Shorts", "Skirts",
                    "Leggings", "Track Pants", "Churidars", "Salwar", "Capris"],
    "Footwear":    ["Shoes", "Sandals", "Flip Flops", "Flats", "Heels",
                    "Sports Shoes", "Formal Shoes", "Casual Shoes",
                    "Sneakers", "Boots", "Loafers"],
    "Accessories": ["Watches", "Belts", "Bags", "Wallets", "Sunglasses",
                    "Jewellery", "Ties", "Scarves", "Caps", "Hats",
                    "Socks", "Backpacks", "Clutches", "Handbags"],
}

COLOUR_HARMONY_GROUPS = {
    "neutral": ["Black", "White", "Grey", "Beige", "Cream", "Off White", "Khaki"],
    "earth":   ["Brown", "Tan", "Camel", "Olive", "Coffee Brown"],
    "cool":    ["Blue", "Navy Blue", "Teal", "Turquoise", "Cyan", "Sea Green"],
    "warm":    ["Red", "Orange", "Yellow", "Gold", "Mustard", "Rust"],
    "pastel":  ["Pink", "Lavender", "Peach", "Baby Blue", "Mint", "Lilac"],
    "jewel":   ["Purple", "Maroon", "Burgundy", "Magenta", "Violet"],
    "green":   ["Green", "Olive", "Sea Green", "Bottle Green", "Lime Green"],
}

# 3 distinct style themes with different colour strategies
OUTFIT_THEMES = [
    {
        "name": "Look 1 — Classic",
        "description": "Timeless tones that pair effortlessly together",
        "strategy": "monochrome",      # same colour group across all pieces
        "usage_tolerance": True,       # exact usage match encouraged
    },
    {
        "name": "Look 2 — Contrast",
        "description": "A bold neutral base anchored by one standout colour",
        "strategy": "neutral_anchor",  # one neutral + one accent
        "usage_tolerance": False,
    },
    {
        "name": "Look 3 — Harmony",
        "description": "Analogous tones from the same colour family",
        "strategy": "harmony_group",   # similar hue family
        "usage_tolerance": False,
    },
]


def _colour_group(colour: str):
    c = colour.strip().title()
    for grp, cols in COLOUR_HARMONY_GROUPS.items():
        if c in cols:
            return grp
    return None


def _harmony_bonus(c_anchor: str, c_item: str, strategy: str) -> float:
    """Score [0..1] how well c_item matches c_anchor under the given strategy."""
    if c_anchor == c_item:
        return 1.0
    g_anchor = _colour_group(c_anchor)
    g_item   = _colour_group(c_item)

    if strategy == "monochrome":
        if g_anchor and g_anchor == g_item:
            return 0.8
        if g_item == "neutral":
            return 0.4   # neutrals always ok
        return 0.0

    if strategy == "neutral_anchor":
        if g_anchor == "neutral" or g_item == "neutral":
            return 0.7
        if g_anchor and g_anchor == g_item:
            return 0.3
        return 0.0

    if strategy == "harmony_group":
        if g_anchor and g_anchor == g_item:
            return 0.7
        if g_item == "neutral":
            return 0.5
        return 0.05

    return 0.0


def _slot_for_item(row: pd.Series):
    sub, art = row["subCategory"], row["articleType"]
    for slot, cats in OUTFIT_SLOTS.items():
        if sub in cats or art in cats:
            return slot
    return None


# ─────────────────────────────────────────────
# 3-OUTFIT GENERATION
# ─────────────────────────────────────────────

def _build_candidate_pool(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    encoders: dict,
    gender: str,
    season: str,
    usage: str,
    colour: str,
) -> pd.DataFrame:
    """Score every gender-matching item and return sorted pool with slot labels."""
    query = {
        "gender": gender, "season": season, "usage": usage,
        "baseColour": colour, "articleType": "", "subCategory": "", "masterCategory": "",
    }
    sims = cosine_similarity(encode_query(query, encoders), feature_matrix)[0]

    mask_gender = df["gender"].isin([gender.strip().title(), "Unisex"])
    mask_season = df["season"].isin([season.strip().title(), "All Season"])
    mask_usage  = df["usage"] == usage.strip().title()

    pool = df.copy()
    pool["_sim"]        = sims
    pool["_base_score"] = sims + mask_season.astype(float) * 0.15 + mask_usage.astype(float) * 0.15
    pool = pool[mask_gender].copy()
    pool["_slot"] = pool.apply(_slot_for_item, axis=1)
    pool = pool[pool["_slot"].notna()]
    return pool.sort_values("_base_score", ascending=False).reset_index(drop=True)


def _build_one_outfit(
    pool: pd.DataFrame,
    theme: dict,
    anchor_colour: str,
    used_ids: set,
    images_dir: str,
) -> dict:
    """
    Pick the single best item per slot such that all pieces harmonise.
    Returns {slot: item_dict}
    """
    outfit = {}
    strategy = theme["strategy"]

    for slot in ["Topwear", "Bottomwear", "Footwear", "Accessories"]:
        slot_pool = pool[pool["_slot"] == slot].copy()
        if slot_pool.empty:
            continue

        # Colour harmony vs anchor
        slot_pool = slot_pool.copy()
        slot_pool["_h1"] = slot_pool["baseColour"].apply(
            lambda c: _harmony_bonus(anchor_colour, c, strategy) * 0.30
        )
        # Also score against already-chosen topwear colour
        if "Topwear" in outfit:
            top_col = outfit["Topwear"]["baseColour"]
            slot_pool["_h2"] = slot_pool["baseColour"].apply(
                lambda c: _harmony_bonus(top_col, c, strategy) * 0.20
            )
        else:
            slot_pool["_h2"] = 0.0

        slot_pool["_final"] = slot_pool["_base_score"] + slot_pool["_h1"] + slot_pool["_h2"]

        # Prefer unused IDs; fall back to full pool for accessories
        available = slot_pool[~slot_pool["id"].isin(used_ids)]
        if available.empty:
            available = slot_pool

        best = available.sort_values("_final", ascending=False).iloc[0]
        used_ids.add(str(best["id"]))

        outfit[slot] = {
            "id":                 str(best["id"]),
            "productDisplayName": best["productDisplayName"],
            "subCategory":        best["subCategory"],
            "articleType":        best["articleType"],
            "baseColour":         best["baseColour"],
            "season":             best["season"],
            "usage":              best["usage"],
            "score":              round(float(best["_final"]), 3),
            "image_path":         get_image_path(str(best["id"]), images_dir),
        }
    return outfit


def recommend_three_outfits(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    encoders: dict,
    gender: str,
    season: str,
    usage: str,
    colour: str,
    images_dir: str = "images",
) -> list:
    """
    Returns a list of 3 outfit dicts, each with keys:
      theme_name, description, slots {Topwear, Bottomwear, Footwear, Accessories}
    Items across outfits are diversified (different IDs where possible).
    """
    pool = _build_candidate_pool(df, feature_matrix, encoders, gender, season, usage, colour)
    if pool.empty:
        return []

    outfits = []
    used_ids: set = set()

    for theme in OUTFIT_THEMES:
        slots = _build_one_outfit(pool, theme, colour, used_ids, images_dir)
        if slots:
            outfits.append({
                "theme_name":  theme["name"],
                "description": theme["description"],
                "slots":       slots,
            })

    return outfits


# ─────────────────────────────────────────────
# VISUAL IMAGE SIMILARITY
# Works on ANY image — from dataset or outside
# Features: RGB histogram + HSV histogram + dominant colours + stats
# ─────────────────────────────────────────────

def _extract_visual_features(img: Image.Image) -> np.ndarray:
    """
    Extract a 71-dimensional visual fingerprint from a PIL image.
    No deep learning required — fast CPU inference.
    """
    img_rgb = img.convert("RGB").resize((128, 128), Image.LANCZOS)
    arr = np.array(img_rgb, dtype=np.float32)

    # ── RGB histograms (8 bins × 3 channels = 24 dims) ──
    rgb_feats = []
    for ch in range(3):
        h, _ = np.histogram(arr[:, :, ch], bins=8, range=(0, 256))
        rgb_feats.extend(h / (h.sum() + 1e-9))

    # ── HSV histograms (16+8+4 = 28 dims) ──
    img_hsv = img_rgb.convert("HSV")
    hsv_arr = np.array(img_hsv, dtype=np.float32)
    hh, _ = np.histogram(hsv_arr[:, :, 0], bins=16, range=(0, 256))
    sh, _ = np.histogram(hsv_arr[:, :, 1], bins=8,  range=(0, 256))
    vh, _ = np.histogram(hsv_arr[:, :, 2], bins=4,  range=(0, 256))
    hsv_feats = np.concatenate([
        hh / (hh.sum() + 1e-9),
        sh / (sh.sum() + 1e-9),
        vh / (vh.sum() + 1e-9),
    ])

    # ── Dominant colours via lightweight k-means (5 clusters, 15 dims) ──
    pixels = arr.reshape(-1, 3)[::16]  # subsample for speed
    np.random.seed(0)
    centres = pixels[np.random.choice(len(pixels), 5, replace=False)].copy()
    for _ in range(6):
        dists = np.linalg.norm(pixels[:, None, :] - centres[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        for k in range(5):
            m = labels == k
            if m.sum() > 0:
                centres[k] = pixels[m].mean(axis=0)
    dom_feats = (centres / 255.0).flatten()

    # ── Global stats (4 dims) ──
    brightness = arr.mean(axis=2)
    sat = hsv_arr[:, :, 1]
    stats = np.array([
        brightness.mean() / 255.0,
        brightness.std()  / 255.0,
        sat.mean()        / 255.0,
        sat.std()         / 255.0,
    ])

    return np.concatenate([rgb_feats, hsv_feats, dom_feats, stats]).astype(np.float32)


@st.cache_data(show_spinner=False)
def build_visual_feature_matrix(df: pd.DataFrame, images_dir: str = "images"):
    """
    Pre-compute and cache visual feature vectors for all dataset images.
    Returns (L2-normalised matrix, list of valid item IDs).
    """
    vectors  = []
    valid_ids = []

    for _, row in df.iterrows():
        p = get_image_path(str(row["id"]), images_dir)
        if p is None:
            continue
        try:
            img = Image.open(p).convert("RGB")
            vectors.append(_extract_visual_features(img))
            valid_ids.append(str(row["id"]))
        except Exception:
            continue

    if not vectors:
        return None, []

    matrix = np.vstack(vectors).astype(np.float32)
    matrix = normalize(matrix, norm="l2")
    return matrix, valid_ids


@st.cache_resource
def build_visual_knn(visual_matrix: np.ndarray, n_neighbors: int = 20):
    k = min(n_neighbors, len(visual_matrix))
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(visual_matrix)
    return knn


def find_similar_by_image(
    uploaded_img: Image.Image,
    df: pd.DataFrame,
    visual_matrix: np.ndarray,
    valid_ids: list,
    visual_knn,
    images_dir: str = "images",
    top_n: int = 8,
) -> list:
    """
    Find visually similar items for ANY uploaded image.
    Returns list of item dicts with similarity %.
    """
    query_vec = _extract_visual_features(uploaded_img)
    norm = np.linalg.norm(query_vec)
    query_vec = (query_vec / (norm + 1e-9)).reshape(1, -1)

    distances, indices = visual_knn.kneighbors(query_vec)
    id_to_row = {str(r["id"]): r for _, r in df.iterrows()}

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        item_id = valid_ids[idx]
        row = id_to_row.get(item_id)
        if row is None:
            continue
        img_path = get_image_path(item_id, images_dir)
        results.append({
            "id":                 item_id,
            "productDisplayName": row["productDisplayName"],
            "subCategory":        row["subCategory"],
            "articleType":        row["articleType"],
            "baseColour":         row["baseColour"],
            "season":             row["season"],
            "usage":              row["usage"],
            "gender":             row["gender"],
            "similarity":         round(max(0.0, 1.0 - float(dist)) * 100, 1),
            "image_path":         img_path,
        })
        if len(results) >= top_n:
            break

    return results


def find_similar_items_by_id(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    knn_model,
    item_id: str,
    images_dir: str = "images",
    top_n: int = 8,
) -> list:
    """Categorical KNN similarity by item ID (fallback / secondary method)."""
    matches = df[df["id"] == str(item_id)]
    if matches.empty:
        return []
    idx = matches.index[0]
    query_vec = feature_matrix[idx].reshape(1, -1)
    distances, indices = knn_model.kneighbors(query_vec)
    results = []
    for dist, i in zip(distances[0], indices[0]):
        row = df.iloc[i]
        if str(row["id"]) == str(item_id):
            continue
        img_path = get_image_path(str(row["id"]), images_dir)
        results.append({
            "id":                 str(row["id"]),
            "productDisplayName": row["productDisplayName"],
            "subCategory":        row["subCategory"],
            "articleType":        row["articleType"],
            "baseColour":         row["baseColour"],
            "season":             row["season"],
            "usage":              row["usage"],
            "gender":             row.get("gender", ""),
            "similarity":         round(max(0.0, 1.0 - float(dist)) * 100, 1),
            "image_path":         img_path,
        })
        if len(results) >= top_n:
            break
    return results
