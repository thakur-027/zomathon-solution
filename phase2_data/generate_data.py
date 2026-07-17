"""
=============================================================
  CSAO (Cart Super Add-On) — Synthetic Data Generator
  Zomathon Hackathon | Phase 2
=============================================================
Tables generated:
  1. users.csv
  2. restaurants.csv
  3. menu_items.csv
  4. orders.csv
  5. order_items.csv
  6. cart_sessions.csv   ← key table for training the model
=============================================================
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# ── Reproducibility ──────────────────────────────────────────
np.random.seed(42)
random.seed(42)

OUTPUT_DIR = "raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Constants ────────────────────────────────────────────────
CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata"]

CITY_ZONES = {
    "Mumbai":    ["Andheri", "Bandra", "Dadar", "Powai", "Thane"],
    "Delhi":     ["Connaught Place", "Lajpat Nagar", "Saket", "Rohini", "Dwarka"],
    "Bangalore": ["Koramangala", "Indiranagar", "Whitefield", "HSR Layout", "Jayanagar"],
    "Hyderabad": ["Jubilee Hills", "Banjara Hills", "Gachibowli", "Hitech City", "Madhapur"],
    "Chennai":   ["T Nagar", "Anna Nagar", "Velachery", "Adyar", "OMR"],
    "Pune":      ["Koregaon Park", "Kothrud", "Wakad", "Baner", "Hinjewadi"],
    "Kolkata":   ["Park Street", "Salt Lake", "New Town", "Ballygunge", "Howrah"],
}

CUISINES = ["North Indian", "South Indian", "Chinese", "Italian", "Biryani",
            "Fast Food", "Desserts", "Beverages", "Continental", "Street Food"]

USER_SEGMENTS = ["budget", "mid_range", "premium", "occasional", "frequent"]

# ── Meal composition rules (for realistic cart sequences) ────
# Maps "what's in cart" → "what's likely next"
MEAL_COMBOS = {
    "main_course":  ["side_dish", "dessert", "beverage", "starter"],
    "starter":      ["main_course", "beverage"],
    "side_dish":    ["dessert", "beverage", "main_course"],
    "dessert":      ["beverage"],
    "beverage":     ["dessert", "side_dish"],
    "snack":        ["beverage", "side_dish"],
}

ITEM_CATEGORIES = list(MEAL_COMBOS.keys())


# ══════════════════════════════════════════════════════════════
#  TABLE 1 — USERS
# ══════════════════════════════════════════════════════════════
def generate_users(n=2000):
    print("  Generating users...")

    segment_weights = [0.30, 0.35, 0.15, 0.12, 0.08]   # budget → frequent
    city_weights    = [0.22, 0.20, 0.18, 0.13, 0.10, 0.10, 0.07]

    cities   = np.random.choice(CITIES, size=n, p=city_weights)
    segments = np.random.choice(USER_SEGMENTS, size=n, p=segment_weights)

    # Segment drives spend behaviour
    avg_order_value_map = {
        "budget": (80, 150), "mid_range": (150, 350),
        "premium": (400, 900), "occasional": (120, 300), "frequent": (200, 500)
    }
    order_freq_map = {
        "budget": (1, 4), "mid_range": (3, 8),
        "premium": (4, 12), "occasional": (1, 2), "frequent": (8, 20)
    }

    aov, freq = [], []
    for seg in segments:
        lo, hi = avg_order_value_map[seg]; aov.append(round(np.random.uniform(lo, hi), 2))
        lo, hi = order_freq_map[seg];      freq.append(int(np.random.uniform(lo, hi)))

    # Registration date — spread over last 3 years
    base_date = datetime(2022, 1, 1)
    reg_dates = [base_date + timedelta(days=int(np.random.uniform(0, 1095))) for _ in range(n)]

    # Veg preference — South India skews more veg
    veg_prob = [0.6 if c in ["Chennai", "Bangalore"] else 0.35 for c in cities]
    is_veg   = [np.random.rand() < p for p in veg_prob]

    df = pd.DataFrame({
        "user_id":            [f"U{str(i).zfill(5)}" for i in range(1, n+1)],
        "city":               cities,
        "zone":               [random.choice(CITY_ZONES[c]) for c in cities],
        "segment":            segments,
        "is_veg":             is_veg,
        "avg_order_value":    aov,
        "orders_per_month":   freq,
        "preferred_cuisine":  np.random.choice(CUISINES, size=n),
        "registration_date":  [d.strftime("%Y-%m-%d") for d in reg_dates],
        # Cold start flag — 15% of users have < 3 orders (sparse history)
        "is_cold_start":      np.random.choice([True, False], size=n, p=[0.15, 0.85]),
    })

    df.to_csv(f"{OUTPUT_DIR}/users.csv", index=False)
    print(f"    ✓ {len(df)} users saved")
    return df


# ══════════════════════════════════════════════════════════════
#  TABLE 2 — RESTAURANTS
# ══════════════════════════════════════════════════════════════
def generate_restaurants(n=500):
    print("  Generating restaurants...")

    cities = np.random.choice(CITIES, size=n, p=[0.22,0.20,0.18,0.13,0.10,0.10,0.07])

    price_tiers   = np.random.choice(["budget","mid","premium"], size=n, p=[0.45,0.40,0.15])
    avg_price_map = {"budget":(60,150), "mid":(150,350), "premium":(350,800)}
    avg_prices    = [round(np.random.uniform(*avg_price_map[t]), 0) for t in price_tiers]

    # Chains vs independent
    is_chain = np.random.choice([True, False], size=n, p=[0.30, 0.70])

    chain_names = ["McDonald's","Domino's","KFC","Pizza Hut","Subway","Burger King",
                   "Haldiram's","Bikanervala","Wow! Momo","Chai Point"]
    indie_names = [f"Spice Garden #{i}" for i in range(1,200)] + \
                  [f"The Food Corner #{i}" for i in range(1,200)]

    names = []
    for chain in is_chain:
        names.append(random.choice(chain_names) if chain else random.choice(indie_names))

    df = pd.DataFrame({
        "restaurant_id":   [f"R{str(i).zfill(4)}" for i in range(1, n+1)],
        "name":            names,
        "city":            cities,
        "zone":            [random.choice(CITY_ZONES[c]) for c in cities],
        "cuisine":         np.random.choice(CUISINES, size=n),
        "price_tier":      price_tiers,
        "avg_item_price":  avg_prices,
        "rating":          np.round(np.random.uniform(3.0, 5.0, size=n), 1),
        "num_ratings":     np.random.randint(50, 5000, size=n),
        "is_chain":        is_chain,
        "is_pure_veg":     np.random.choice([True, False], size=n, p=[0.25, 0.75]),
        "avg_prep_time_mins": np.random.randint(15, 45, size=n),
    })

    df.to_csv(f"{OUTPUT_DIR}/restaurants.csv", index=False)
    print(f"    ✓ {len(df)} restaurants saved")
    return df


# ══════════════════════════════════════════════════════════════
#  TABLE 3 — MENU ITEMS
# ══════════════════════════════════════════════════════════════
def generate_menu_items(restaurants_df):
    print("  Generating menu items...")

    # Realistic item names by category
    item_pool = {
        "main_course": [
            "Butter Chicken","Paneer Butter Masala","Dal Makhani","Chicken Biryani",
            "Veg Biryani","Mutton Biryani","Margherita Pizza","Chicken Pizza",
            "Pasta Arrabiata","Fried Rice","Hakka Noodles","Masala Dosa",
            "Chole Bhature","Rajma Chawal","Fish Curry","Egg Curry"
        ],
        "starter": [
            "Chicken Tikka","Veg Spring Rolls","Samosa (2 pcs)","Paneer Tikka",
            "Chicken Wings","Veg Seekh Kebab","Fish Tikka","Crispy Corn",
            "Pav Bhaji","Aloo Tikki","Onion Rings","Chicken Nuggets"
        ],
        "side_dish": [
            "Raita","Salan","Laccha Paratha","Butter Naan","Garlic Naan",
            "Jeera Rice","Papad","Green Salad","Coleslaw","Pita Bread",
            "Garlic Bread","Steamed Rice"
        ],
        "dessert": [
            "Gulab Jamun","Rasgulla","Kheer","Brownie","Cheesecake Slice",
            "Ice Cream (Vanilla)","Ice Cream (Chocolate)","Halwa","Gajar Ka Halwa",
            "Kulfi","Chocolate Mousse","Fruit Custard"
        ],
        "beverage": [
            "Mango Lassi","Sweet Lassi","Cold Coffee","Masala Chai","Fresh Lime Soda",
            "Coca-Cola","Sprite","Mineral Water","Buttermilk","Rose Sharbat",
            "Watermelon Juice","Mango Shake"
        ],
        "snack": [
            "French Fries","Masala Fries","Garlic Bread (plain)","Veg Sandwich",
            "Chicken Sandwich","Corn on the Cob","Bhel Puri","Sev Puri",
            "Nachos","Popcorn Chicken","Mini Burgers","Veg Wrap"
        ]
    }

    records = []
    item_id = 1

    for _, rest in restaurants_df.iterrows():
        # Each restaurant has 8–20 items
        num_items = np.random.randint(8, 21)
        selected_cats = random.choices(ITEM_CATEGORIES, k=num_items)

        for cat in selected_cats:
            name  = random.choice(item_pool[cat])
            price = round(np.random.uniform(
                float(rest["avg_item_price"]) * 0.4,
                float(rest["avg_item_price"]) * 1.8
            ), 0)

            is_veg = True if rest["is_pure_veg"] else np.random.choice(
                [True, False],
                p=[0.55, 0.45] if cat in ["starter","main_course"] else [0.75, 0.25]
            )

            records.append({
                "item_id":        f"I{str(item_id).zfill(6)}",
                "restaurant_id":  rest["restaurant_id"],
                "item_name":      name,
                "category":       cat,
                "price":          max(price, 30.0),   # floor ₹30
                "is_veg":         is_veg,
                "is_available":   np.random.choice([True, False], p=[0.92, 0.08]),
                "popularity_rank":np.random.randint(1, 100),   # lower = more popular
                "calories":       np.random.randint(100, 900),
            })
            item_id += 1

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/menu_items.csv", index=False)
    print(f"    ✓ {len(df)} menu items saved")
    return df


# ══════════════════════════════════════════════════════════════
#  TABLE 4 — ORDERS  (header-level)
# ══════════════════════════════════════════════════════════════
def generate_orders(users_df, restaurants_df, n=15000):
    print("  Generating orders...")

    user_ids = users_df["user_id"].tolist()
    rest_ids = restaurants_df["restaurant_id"].tolist()

    # Simulate peak-hour ordering pattern
    # Hour weights: peak at 13 (lunch) and 20 (dinner)
    hour_weights = np.array([
        0.3, 0.2, 0.1, 0.1, 0.1, 0.2,          # 0–5  (night)
        0.4, 0.8, 1.2, 1.5, 1.8, 2.5,           # 6–11 (morning)
        4.5, 5.0, 3.5, 2.5, 2.0, 2.5,           # 12–17 (afternoon)
        3.5, 4.0, 5.5, 4.5, 3.0, 1.5            # 18–23 (evening/night peak)
    ])
    hour_weights /= hour_weights.sum()

    start_date = datetime(2024, 1, 1)
    end_date   = datetime(2025, 1, 1)
    date_range = (end_date - start_date).days

    records = []
    for i in range(1, n+1):
        user_id = random.choice(user_ids)
        rest_id = random.choice(rest_ids)

        order_date = start_date + timedelta(days=int(np.random.uniform(0, date_range)))
        order_hour = np.random.choice(range(24), p=hour_weights)
        order_dt   = order_date.replace(hour=order_hour,
                                         minute=np.random.randint(0, 60))

        # Meal time bucket
        if   6  <= order_hour < 11: meal_time = "breakfast"
        elif 11 <= order_hour < 16: meal_time = "lunch"
        elif 16 <= order_hour < 19: meal_time = "snack"
        elif 19 <= order_hour < 23: meal_time = "dinner"
        else:                        meal_time = "late_night"

        # Weekend flag
        is_weekend = order_dt.weekday() >= 5

        records.append({
            "order_id":        f"ORD{str(i).zfill(7)}",
            "user_id":         user_id,
            "restaurant_id":   rest_id,
            "order_datetime":  order_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "meal_time":       meal_time,
            "is_weekend":      is_weekend,
            "order_status":    np.random.choice(
                                ["delivered","cancelled","failed"],
                                p=[0.88, 0.09, 0.03]),
            "payment_method":  np.random.choice(
                                ["UPI","card","COD","wallet"],
                                p=[0.55, 0.25, 0.12, 0.08]),
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/orders.csv", index=False)
    print(f"    ✓ {len(df)} orders saved")
    return df


# ══════════════════════════════════════════════════════════════
#  TABLE 5 — ORDER ITEMS
# ══════════════════════════════════════════════════════════════
def generate_order_items(orders_df, menu_items_df):
    print("  Generating order items...")

    records = []
    # Build a lookup: restaurant_id → list of item_ids
    rest_items = menu_items_df[menu_items_df["is_available"] == True].groupby(
        "restaurant_id")["item_id"].apply(list).to_dict()

    delivered_orders = orders_df[orders_df["order_status"] == "delivered"]

    for _, order in delivered_orders.iterrows():
        rest_id  = order["restaurant_id"]
        items    = rest_items.get(rest_id, [])
        if not items:
            continue

        # 1–5 items per order (weighted towards 2–3)
        num_items = np.random.choice([1,2,3,4,5], p=[0.15,0.35,0.30,0.13,0.07])
        num_items = min(num_items, len(items))
        chosen    = random.sample(items, num_items)

        for item_id in chosen:
            records.append({
                "order_id":  order["order_id"],
                "item_id":   item_id,
                "quantity":  np.random.choice([1,2,3], p=[0.75, 0.20, 0.05]),
            })

    df = pd.DataFrame(records)
    # Add price from menu_items for convenience
    df = df.merge(
        menu_items_df[["item_id","price","category","item_name"]],
        on="item_id", how="left"
    )
    df["line_total"] = df["price"] * df["quantity"]
    df.to_csv(f"{OUTPUT_DIR}/order_items.csv", index=False)
    print(f"    ✓ {len(df)} order item rows saved")
    return df


# ══════════════════════════════════════════════════════════════
#  TABLE 6 — CART SESSIONS  ← THE MOST IMPORTANT TABLE
#
#  Each row = one step in a cart-building session
#  It records:
#    - what was already in the cart (cart_state)
#    - which item was shown as recommendation (candidate_item)
#    - whether the user ADDED it (label = 1) or ignored it (label = 0)
#
#  This is the training data for your ranking model.
# ══════════════════════════════════════════════════════════════
def generate_cart_sessions(orders_df, order_items_df, menu_items_df, users_df, n_sessions=10000):
    print("  Generating cart sessions (training data)...")

    # Merge orders with users for context
    orders_with_ctx = orders_df.merge(
        users_df[["user_id","city","segment","is_veg","avg_order_value"]],
        on="user_id", how="left"
    )

    # Build actual order baskets (items bought together)
    baskets = order_items_df.groupby("order_id")["item_id"].apply(list).to_dict()
    item_info = menu_items_df.set_index("item_id")[
        ["item_name","category","price","is_veg","restaurant_id","popularity_rank"]
    ].to_dict("index")

    # Build co-occurrence counts (item A bought with item B how many times)
    # Used to generate "negative" candidates realistically
    cooccur = {}
    for _, items in baskets.items():
        for a in items:
            for b in items:
                if a != b:
                    cooccur.setdefault(a, {})
                    cooccur[a][b] = cooccur[a].get(b, 0) + 1

    delivered = orders_with_ctx[orders_with_ctx["order_status"] == "delivered"]
    order_ids = delivered["order_id"].tolist()

    records = []
    session_id = 1

    for _ in range(n_sessions):
        order_id = random.choice(order_ids)
        order_row = delivered[delivered["order_id"] == order_id].iloc[0]
        basket    = baskets.get(order_id, [])

        if len(basket) < 2:
            continue   # Need at least 2 items to simulate add-on sequence

        # Simulate session: user adds items one by one
        # First item = anchor (already in cart)
        random.shuffle(basket)
        cart   = [basket[0]]          # start with 1 item
        added  = set(basket)          # items user actually bought = positives

        rest_id     = order_row["restaurant_id"]
        all_rest_items = menu_items_df[
            (menu_items_df["restaurant_id"] == rest_id) &
            (menu_items_df["is_available"] == True)
        ]["item_id"].tolist()

        if not all_rest_items:
            continue

        # For each step, generate candidates (positives + negatives)
        for step in range(1, min(len(basket), 5)):
            pos_item = basket[step]   # item user actually added next

            # Negatives = other items from same restaurant not in basket
            negatives = [i for i in all_rest_items
                         if i not in added and i in item_info]
            negatives  = random.sample(negatives, min(7, len(negatives)))

            candidates = [(pos_item, 1)] + [(n, 0) for n in negatives]

            for (cand_item, label) in candidates:
                if cand_item not in item_info:
                    continue

                cart_cats   = [item_info[c]["category"] for c in cart if c in item_info]
                cart_prices = [item_info[c]["price"]    for c in cart if c in item_info]
                cand_info   = item_info[cand_item]

                # ── Features ────────────────────────────────────
                records.append({
                    # IDs
                    "session_id":          f"S{str(session_id).zfill(7)}",
                    "order_id":            order_id,
                    "user_id":             order_row["user_id"],
                    "restaurant_id":       rest_id,
                    "step":                step,

                    # Cart state features
                    "cart_size":           len(cart),
                    "cart_total_value":    round(sum(cart_prices), 2),
                    "cart_avg_price":      round(np.mean(cart_prices), 2) if cart_prices else 0,
                    "has_main_course":     int("main_course" in cart_cats),
                    "has_starter":         int("starter" in cart_cats),
                    "has_side_dish":       int("side_dish" in cart_cats),
                    "has_dessert":         int("dessert" in cart_cats),
                    "has_beverage":        int("beverage" in cart_cats),
                    "has_snack":           int("snack" in cart_cats),
                    "cart_is_complete":    int(  # has main + beverage = "complete"
                        "main_course" in cart_cats and "beverage" in cart_cats
                    ),

                    # Candidate item features
                    "cand_item_id":        cand_item,
                    "cand_item_name":      cand_info["item_name"],
                    "cand_category":       cand_info["category"],
                    "cand_price":          cand_info["price"],
                    "cand_is_veg":         int(cand_info["is_veg"]),
                    "cand_popularity":     cand_info["popularity_rank"],
                    "price_ratio":         round(   # candidate price vs cart avg
                        cand_info["price"] / max(np.mean(cart_prices), 1), 3
                    ) if cart_prices else 1.0,
                    # Co-occurrence score between candidate and cart items
                    "cooccur_score":       sum(
                        cooccur.get(cart_item, {}).get(cand_item, 0)
                        for cart_item in cart
                    ),

                    # User features
                    "user_segment":        order_row["segment"],
                    "user_is_veg":         int(order_row["is_veg"]),
                    "user_avg_spend":      order_row["avg_order_value"],
                    "user_city":           order_row["city"],

                    # Context features
                    "meal_time":           order_row["meal_time"],
                    "is_weekend":          int(order_row["is_weekend"]),

                    # TARGET LABEL
                    "label":               label,   # 1 = added, 0 = not added
                })

            cart.append(pos_item)   # advance cart state
        session_id += 1

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/cart_sessions.csv", index=False)
    print(f"    ✓ {len(df)} cart session rows saved  |  "
          f"Positives: {df['label'].sum()} | Negatives: {(df['label']==0).sum()}")
    return df


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🚀 Starting Synthetic Data Generation...\n")

    users_df       = generate_users(n=2000)
    restaurants_df = generate_restaurants(n=500)
    menu_items_df  = generate_menu_items(restaurants_df)
    orders_df      = generate_orders(users_df, restaurants_df, n=15000)
    order_items_df = generate_order_items(orders_df, menu_items_df)
    cart_df        = generate_cart_sessions(orders_df, order_items_df,
                                            menu_items_df, users_df, n_sessions=10000)

    print("\n✅ All tables generated successfully!")
    print(f"\n📁 Files saved to: {OUTPUT_DIR}/")
    print("\n📊 Summary:")
    for fname in os.listdir(OUTPUT_DIR):
        path = f"{OUTPUT_DIR}/{fname}"
        df   = pd.read_csv(path)
        print(f"   {fname:<25} → {len(df):>7,} rows  ×  {len(df.columns):>2} columns")

    print("\n🎯 Key insight: cart_sessions.csv is your TRAINING DATA.")
    print("   'label' column = 1 (user added item) or 0 (user ignored it)")
    print("   Next step: Phase 3 — Feature Engineering!\n")
