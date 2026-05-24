"""
generate_demo_data.py
Run this to create a sample styles.csv and placeholder images so you can
test the app before placing your real dataset.

Usage:
    python generate_demo_data.py
"""

import os
import csv
import random
from PIL import Image, ImageDraw, ImageFont

# ── palette for placeholder images ──
COLOURS_HEX = {
    "Black":  "#1a1a1a", "White":  "#f5f5f0", "Blue":   "#4a6fa5",
    "Red":    "#c94040", "Green":  "#4a8a5a", "Yellow": "#d4b84a",
    "Pink":   "#d48a9a", "Brown":  "#8a5a3a", "Navy Blue": "#1a2a6a",
    "Grey":   "#8a8a8a", "Purple": "#6a3a8a", "Orange": "#d4703a",
    "Beige":  "#d4c4a0", "Cream":  "#f0ead0", "Maroon": "#8a1a1a",
}

GENDERS   = ["Men", "Women", "Boys", "Girls", "Unisex"]
MCATS     = ["Apparel", "Footwear", "Accessories"]
SCATS     = {
    "Apparel":     ["Topwear", "Bottomwear", "Innerwear", "Dress"],
    "Footwear":    ["Shoes", "Sandals", "Flip Flops"],
    "Accessories": ["Watches", "Bags", "Belts", "Sunglasses", "Wallets"],
}
ATYPES    = {
    "Topwear":    ["Shirts", "Tshirts", "Tops", "Sweatshirts", "Jackets", "Kurtas"],
    "Bottomwear": ["Jeans", "Trousers", "Shorts", "Skirts", "Leggings", "Track Pants"],
    "Innerwear":  ["Bra", "Briefs", "Vests"],
    "Dress":      ["Dresses", "Jumpsuits"],
    "Shoes":      ["Sports Shoes", "Formal Shoes", "Casual Shoes", "Sneakers"],
    "Sandals":    ["Sandals", "Heels", "Flats"],
    "Flip Flops": ["Flip Flops"],
    "Watches":    ["Watches"],
    "Bags":       ["Handbags", "Backpacks", "Clutches"],
    "Belts":      ["Belts"],
    "Sunglasses": ["Sunglasses"],
    "Wallets":    ["Wallets"],
}
SEASONS   = ["Summer", "Winter", "Fall", "Spring"]
USAGES    = ["Casual", "Formal", "Sports", "Party", "Ethnic", "Travel"]
COLOURS   = list(COLOURS_HEX.keys())
YEARS     = [2011, 2012, 2013, 2014, 2015]

NAMES = [
    "Classic {colour} {type}", "Premium {colour} {type}",
    "Stylish {colour} {type}", "Trendy {colour} {type}",
    "Elegant {colour} {type}", "Modern {colour} {type}",
    "Essential {colour} {type}", "Urban {colour} {type}",
]

random.seed(42)


def make_placeholder_image(item_id: int, colour: str, atype: str,
                             size=(300, 400)) -> Image.Image:
    hex_col = COLOURS_HEX.get(colour, "#888888")
    r = int(hex_col[1:3], 16)
    g = int(hex_col[3:5], 16)
    b = int(hex_col[5:7], 16)

    img = Image.new("RGB", size, color=(r, g, b))
    draw = ImageDraw.Draw(img)

    # Subtle pattern
    for y in range(0, size[1], 20):
        draw.line([(0, y), (size[0], y)], fill=(min(r+20, 255), min(g+20, 255), min(b+20, 255)), width=1)

    # Text label
    label = f"ID:{item_id}\n{atype}"
    text_r, text_g, text_b = 255-r, 255-g, 255-b
    draw.multiline_text((10, 10), label, fill=(text_r, text_g, text_b))

    # Simple clothing icon (rectangle)
    pad = 40
    draw.rectangle([pad, pad*2, size[0]-pad, size[1]-pad],
                   outline=(text_r, text_g, text_b), width=3)

    return img


def generate(n: int = 500, out_dir: str = "."):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    start_id = 15000

    for i in range(n):
        item_id  = start_id + i
        gender   = random.choice(GENDERS)
        mcat     = random.choice(MCATS)
        scat     = random.choice(SCATS[mcat])
        atype    = random.choice(ATYPES.get(scat, [scat]))
        colour   = random.choice(COLOURS)
        season   = random.choice(SEASONS)
        year     = random.choice(YEARS)
        usage    = random.choice(USAGES)
        tmpl     = random.choice(NAMES)
        name     = tmpl.format(colour=colour, type=atype)

        rows.append({
            "id":                 item_id,
            "gender":             gender,
            "masterCategory":     mcat,
            "subCategory":        scat,
            "articleType":        atype,
            "baseColour":         colour,
            "season":             season,
            "year":               year,
            "usage":              usage,
            "productDisplayName": name,
        })

        # Generate placeholder image
        img = make_placeholder_image(item_id, colour, atype)
        img.save(os.path.join(img_dir, f"{item_id}.jpg"), "JPEG", quality=85)

    csv_path = os.path.join(out_dir, "styles.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Generated {n} items → {csv_path}")
    print(f"✅ Generated {n} placeholder images → {img_dir}/")
    print(f"\nRun: streamlit run app.py")


if __name__ == "__main__":
    generate(n=600)
