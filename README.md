# 👗 FashionAI — Intelligent Style Recommender

A production-ready **Streamlit** app that generates complete outfit recommendations
and finds similar clothing items using machine-learning similarity search.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn Pillow
```

### 2a. Use your real dataset (Kaggle Fashion Dataset)

Place your files like this:

```
project/
├── app.py
├── utils.py
├── styles.csv          ← your dataset
└── images/             ← folder of clothing images
    ├── 15970.jpg
    ├── 39386.jpg
    └── ...
```

### 2b. OR generate demo data to test immediately

```bash
python generate_demo_data.py   # creates styles.csv + images/ with 600 items
```

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎯 Features

### ✦ Outfit Generator
- Filter by **Gender**, **Season**, **Usage**, **Preferred Colour**
- Generates a complete outfit across 4 slots:
  - 👕 Topwear
  - 👖 Bottomwear
  - 👟 Footwear
  - ⌚ Accessories
- Uses **cosine similarity** on encoded categorical features
- Ensures diversity — no repeated article types per slot
- Shows product images, names, colours, and categories

### ✦ Similar Items Finder
- **By Item ID** — enter any product ID to find the 5–12 most similar items
- **By Attributes** — specify gender, colour, article type, etc. to browse similar products
- Shows **similarity score** per result
- KNN model with cosine distance for fast inference

---

## 🛠 Architecture

```
app.py          — Streamlit UI, tabs, sidebar, rendering
utils.py        — ML logic (pure functions, no UI)
  ├─ load_data()              → loads & cleans CSV
  ├─ build_feature_matrix()   → LabelEncoder on 7 features
  ├─ build_knn_model()        → NearestNeighbors (cosine)
  ├─ recommend_outfit()       → filters + cosine sim + slot grouping
  ├─ find_similar_items()     → KNN lookup by item ID
  └─ find_similar_by_features() → cosine sim from attribute query
```

### ML approach

| Step | Method |
|------|--------|
| Feature encoding | `LabelEncoder` per column (7 features) |
| Outfit similarity | Cosine similarity + bonus scoring for season/usage match |
| Similar item search | K-Nearest Neighbours (`metric='cosine'`) |
| Diversity | Slot-based grouping prevents duplicate article types |

---

## 📦 Dataset Format (styles.csv)

| Column | Description |
|--------|-------------|
| `id` | Unique item ID (matches image filename) |
| `gender` | Men / Women / Boys / Girls / Unisex |
| `masterCategory` | Apparel / Footwear / Accessories |
| `subCategory` | Topwear / Bottomwear / Shoes / Bags / … |
| `articleType` | Shirts / Jeans / Sneakers / Watches / … |
| `baseColour` | Blue / Black / White / … |
| `season` | Summer / Winter / Fall / Spring |
| `year` | 2011–2015 (or any year) |
| `usage` | Casual / Formal / Sports / Party / … |
| `productDisplayName` | Human-readable product name |

---

## ⚙️ Performance Notes

- `@st.cache_data` on data loading and feature matrix — only runs once per session
- `@st.cache_resource` on KNN model — shared across all users
- No deep learning; all inference is CPU-only and sub-second
- Works with 1k–44k+ item datasets

---

## 🎨 Design

- **Typography**: Cormorant Garamond (serif) + DM Sans (sans-serif)
- **Palette**: warm cream, umber, clay, rose accents
- **Layout**: responsive card grid, slot-based outfit display
- Runs fully offline — no external API calls

---

*Built with Streamlit · pandas · scikit-learn · Pillow*
