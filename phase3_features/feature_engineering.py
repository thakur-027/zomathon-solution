"""
=============================================================
  CSAO Recommendation System — Phase 3
  Feature Engineering + Exploratory Data Analysis (EDA)
=============================================================
What this script does:
  1. Loads all 6 tables
  2. Performs EDA (understand the data with charts + stats)
  3. Engineers new features on top of existing ones
  4. Encodes categorical columns for ML
  5. Saves final feature-rich training dataset → model_input.csv
  6. Saves all EDA charts → eda_charts/
=============================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no popup needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
DATA_DIR   = "../phase2_data/raw"
OUTPUT_DIR = "."
CHART_DIR  = os.path.join(OUTPUT_DIR, "eda_charts")
os.makedirs(CHART_DIR, exist_ok=True)

print("\n🚀 Phase 3 — Feature Engineering + EDA\n")

# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════
print("📂 Step 1: Loading data...")

cart_df    = pd.read_csv(f"{DATA_DIR}/cart_sessions.csv")
users_df   = pd.read_csv(f"{DATA_DIR}/users.csv")
rest_df    = pd.read_csv(f"{DATA_DIR}/restaurants.csv")
menu_df    = pd.read_csv(f"{DATA_DIR}/menu_items.csv")
orders_df  = pd.read_csv(f"{DATA_DIR}/orders.csv")
oi_df      = pd.read_csv(f"{DATA_DIR}/order_items.csv")

print(f"  cart_sessions : {cart_df.shape}")
print(f"  users         : {users_df.shape}")
print(f"  restaurants   : {rest_df.shape}")
print(f"  menu_items    : {menu_df.shape}")
print(f"  orders        : {orders_df.shape}")
print(f"  order_items   : {oi_df.shape}")


# ══════════════════════════════════════════════════════════════
#  STEP 2 — EDA (Exploratory Data Analysis)
# ══════════════════════════════════════════════════════════════
print("\n📊 Step 2: Running EDA...")

# ── 2a. Label Distribution ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("EDA — Chart 1: Label Distribution", fontsize=14, fontweight="bold")

label_counts = cart_df["label"].value_counts()
colors = ["#FF4D4D", "#00C853"]
axes[0].pie(label_counts, labels=["Not Added (0)", "Added (1)"],
            autopct="%1.1f%%", colors=colors, startangle=90)
axes[0].set_title("Add-On Accept vs Reject")

axes[1].bar(["Not Added (0)", "Added (1)"], label_counts.values, color=colors)
axes[1].set_title("Count of Labels")
axes[1].set_ylabel("Count")
for i, v in enumerate(label_counts.values):
    axes[1].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart1_label_distribution.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 1: Label distribution")

# ── 2b. Acceptance Rate by Category ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("EDA — Chart 2: Acceptance Rate by Item Category", fontsize=14, fontweight="bold")

cat_accept = cart_df.groupby("cand_category")["label"].mean().sort_values(ascending=False)
bars = ax.bar(cat_accept.index, cat_accept.values * 100, color="#FF6B35", edgecolor="white")
ax.set_ylabel("Acceptance Rate (%)")
ax.set_xlabel("Item Category")
ax.set_ylim(0, max(cat_accept.values * 100) * 1.2)
for bar, val in zip(bars, cat_accept.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val*100:.1f}%", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart2_acceptance_by_category.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 2: Acceptance by category")

# ── 2c. Acceptance Rate by Meal Time ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("EDA — Chart 3: Acceptance Rate by Meal Time", fontsize=14, fontweight="bold")

meal_order = ["breakfast", "lunch", "snack", "dinner", "late_night"]
meal_accept = cart_df.groupby("meal_time")["label"].mean().reindex(meal_order)
colors_meal = ["#FFB347", "#FF6B35", "#FF8C42", "#E84855", "#6B2D8B"]
bars = ax.bar(meal_accept.index, meal_accept.values * 100, color=colors_meal)
ax.set_ylabel("Acceptance Rate (%)")
ax.set_xlabel("Meal Time")
for bar, val in zip(bars, meal_accept.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{val*100:.1f}%", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart3_acceptance_by_mealtime.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 3: Acceptance by meal time")

# ── 2d. Acceptance Rate by User Segment ──────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("EDA — Chart 4: Acceptance Rate by User Segment", fontsize=14, fontweight="bold")

seg_accept = cart_df.groupby("user_segment")["label"].mean().sort_values(ascending=False)
bars = ax.bar(seg_accept.index, seg_accept.values * 100, color="#4ECDC4", edgecolor="white")
ax.set_ylabel("Acceptance Rate (%)")
ax.set_xlabel("User Segment")
for bar, val in zip(bars, seg_accept.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{val*100:.1f}%", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart4_acceptance_by_segment.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 4: Acceptance by user segment")

# ── 2e. Cart Size vs Acceptance ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("EDA — Chart 5: Cart Size vs Acceptance Rate", fontsize=14, fontweight="bold")

cart_size_accept = cart_df.groupby("cart_size")["label"].mean()
ax.plot(cart_size_accept.index, cart_size_accept.values * 100,
        marker="o", color="#E84855", linewidth=2.5, markersize=8)
ax.fill_between(cart_size_accept.index, cart_size_accept.values * 100, alpha=0.2, color="#E84855")
ax.set_xlabel("Number of Items Already in Cart")
ax.set_ylabel("Acceptance Rate (%)")
ax.set_xticks(cart_size_accept.index)

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart5_cartsize_vs_acceptance.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 5: Cart size vs acceptance")

# ── 2f. Co-occurrence Score vs Acceptance ────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("EDA — Chart 6: Co-occurrence Score vs Acceptance Rate", fontsize=14, fontweight="bold")

# Bin co-occurrence scores
cart_df["cooccur_bin"] = pd.cut(cart_df["cooccur_score"],
                                 bins=[-1, 0, 2, 5, 10, 999],
                                 labels=["0 (none)", "1–2", "3–5", "6–10", "10+"])
cooccur_accept = cart_df.groupby("cooccur_bin", observed=True)["label"].mean()
bars = ax.bar(cooccur_accept.index.astype(str), cooccur_accept.values * 100,
              color="#7B2FBE", edgecolor="white")
ax.set_ylabel("Acceptance Rate (%)")
ax.set_xlabel("Co-occurrence Score (how often item appears with cart items)")
for bar, val in zip(bars, cooccur_accept.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val*100:.1f}%", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart6_cooccur_vs_acceptance.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 6: Co-occurrence vs acceptance")

# ── 2g. City-wise order volume ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("EDA — Chart 7: Order Volume by City", fontsize=14, fontweight="bold")

city_counts = cart_df["user_city"].value_counts()
ax.barh(city_counts.index, city_counts.values, color="#FF6B35")
ax.set_xlabel("Number of Cart Session Rows")
for i, v in enumerate(city_counts.values):
    ax.text(v + 200, i, f"{v:,}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart7_city_volume.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 7: City-wise volume")

# ── 2h. Price Ratio vs Acceptance ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("EDA — Chart 8: Price Ratio vs Acceptance Rate", fontsize=14, fontweight="bold")

cart_df["price_ratio_bin"] = pd.cut(cart_df["price_ratio"],
                                     bins=[0, 0.5, 1.0, 1.5, 2.0, 999],
                                     labels=["<0.5x", "0.5–1x", "1–1.5x", "1.5–2x", ">2x"])
pr_accept = cart_df.groupby("price_ratio_bin", observed=True)["label"].mean()
bars = ax.bar(pr_accept.index.astype(str), pr_accept.values * 100, color="#00B4D8")
ax.set_ylabel("Acceptance Rate (%)")
ax.set_xlabel("Candidate Price / Cart Average Price")
for bar, val in zip(bars, pr_accept.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{val*100:.1f}%", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart8_priceratio_vs_acceptance.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 8: Price ratio vs acceptance")


# ══════════════════════════════════════════════════════════════
#  STEP 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n⚙️  Step 3: Engineering new features...")

df = cart_df.copy()

# ── 3a. Meal Completeness Score ──────────────────────────────
# Score 0–5: how many meal components are covered
df["meal_completeness_score"] = (
    df["has_main_course"] +
    df["has_starter"] +
    df["has_side_dish"] +
    df["has_dessert"] +
    df["has_beverage"]
)

# ── 3b. Complementary Category Flag ─────────────────────────
# Is the candidate item filling a MISSING meal slot?
# Logic: if cart has main course but no beverage, beverage is "complementary"
def is_complementary(row):
    cat = row["cand_category"]
    if cat == "beverage"   and row["has_beverage"] == 0:   return 1
    if cat == "dessert"    and row["has_dessert"] == 0:     return 1
    if cat == "side_dish"  and row["has_side_dish"] == 0:   return 1
    if cat == "starter"    and row["has_starter"] == 0:     return 1
    if cat == "main_course"and row["has_main_course"] == 0: return 1
    return 0

df["is_complementary_category"] = df.apply(is_complementary, axis=1)
print("  ✓ Meal completeness score + complementary category flag")

# ── 3c. Veg-User Veg-Item Match ──────────────────────────────
# If a veg user is shown a veg item → higher chance of acceptance
df["veg_match"] = ((df["user_is_veg"] == 1) & (df["cand_is_veg"] == 1)).astype(int)
df["veg_mismatch"] = ((df["user_is_veg"] == 1) & (df["cand_is_veg"] == 0)).astype(int)
print("  ✓ Veg match / mismatch flags")

# ── 3d. Affordability Score ──────────────────────────────────
# How affordable is the candidate relative to user's average spend?
# Close to 1.0 = very affordable, >3 = expensive for this user
df["affordability"] = np.round(
    df["cand_price"] / df["user_avg_spend"].replace(0, 1), 3
)
print("  ✓ Affordability score")

# ── 3e. Cart Value After Add ─────────────────────────────────
# What would cart total be if item is added?
df["cart_value_after_add"] = df["cart_total_value"] + df["cand_price"]

# Upsell potential — how much % does this item increase cart value?
df["upsell_pct"] = np.round(
    df["cand_price"] / df["cart_total_value"].replace(0, 1) * 100, 2
)
print("  ✓ Cart value after add + upsell %")

# ── 3f. Popularity Score (inverted rank) ─────────────────────
# popularity_rank: lower = more popular
# Convert to 0–1 score where 1 = most popular
df["popularity_score"] = np.round(1 - (df["cand_popularity"] / 100), 3)
print("  ✓ Popularity score (normalized)")

# ── 3g. Co-occurrence Strength ──────────────────────────────
# Log-transform cooccur_score (reduces effect of outliers)
df["cooccur_log"] = np.log1p(df["cooccur_score"])
print("  ✓ Log co-occurrence score")

# ── 3h. Session Step Ratio ───────────────────────────────────
# How far into the session are we? (step / max possible steps)
df["step_ratio"] = df["step"] / 4.0     # max step is 4 in our data
print("  ✓ Session step ratio")

# ── 3i. Meal Time Encoding ───────────────────────────────────
# Some categories are more relevant at certain meal times
meal_time_map = {
    "breakfast": 0, "lunch": 1, "snack": 2, "dinner": 3, "late_night": 4
}
df["meal_time_encoded"] = df["meal_time"].map(meal_time_map)

# Is dinner / late night (higher add-on tendency)?
df["is_dinner_or_late"] = df["meal_time"].isin(["dinner", "late_night"]).astype(int)
print("  ✓ Meal time encoded")

# ── 3j. User Segment Encoding ────────────────────────────────
segment_map = {
    "budget": 0, "occasional": 1, "mid_range": 2, "frequent": 3, "premium": 4
}
df["segment_encoded"] = df["user_segment"].map(segment_map)
print("  ✓ User segment encoded")

# ── 3k. City Tier ────────────────────────────────────────────
# Tier 1 cities tend to have higher order values & acceptance
tier1 = ["Mumbai", "Delhi", "Bangalore"]
df["is_tier1_city"] = df["user_city"].isin(tier1).astype(int)
print("  ✓ City tier flag")

# ── 3l. Category Encoded ─────────────────────────────────────
category_map = {
    "main_course": 0, "starter": 1, "side_dish": 2,
    "dessert": 3, "beverage": 4, "snack": 5
}
df["cand_category_encoded"] = df["cand_category"].map(category_map)
print("  ✓ Category encoded")

# ── 3m. Weekend Premium Interaction ─────────────────────────
# Premium users on weekends tend to order more add-ons
df["weekend_premium"] = (
    (df["is_weekend"] == 1) & (df["user_segment"] == "premium")
).astype(int)
print("  ✓ Weekend × premium interaction feature")

# ── 3n. Drop helper/bin columns used for EDA only ────────────
df.drop(columns=["cooccur_bin", "price_ratio_bin"], inplace=True, errors="ignore")


# ══════════════════════════════════════════════════════════════
#  STEP 4 — DEFINE FINAL FEATURE SET
# ══════════════════════════════════════════════════════════════
print("\n📋 Step 4: Defining final feature set for model...")

FEATURE_COLS = [
    # Cart state
    "cart_size", "cart_total_value", "cart_avg_price",
    "has_main_course", "has_starter", "has_side_dish",
    "has_dessert", "has_beverage", "has_snack",
    "cart_is_complete", "meal_completeness_score",

    # Candidate item
    "cand_price", "cand_is_veg", "cand_popularity",
    "cand_category_encoded", "popularity_score",

    # Relationship between cart and candidate
    "price_ratio", "cooccur_score", "cooccur_log",
    "is_complementary_category", "veg_match", "veg_mismatch",
    "affordability", "upsell_pct",

    # User
    "user_is_veg", "user_avg_spend", "segment_encoded",
    "is_tier1_city",

    # Context
    "meal_time_encoded", "is_weekend", "is_dinner_or_late",
    "step", "step_ratio", "weekend_premium",
]

TARGET_COL = "label"
ID_COLS    = ["session_id", "order_id", "user_id", "restaurant_id",
              "cand_item_id", "cand_item_name", "cand_category"]

print(f"  Total features: {len(FEATURE_COLS)}")


# ══════════════════════════════════════════════════════════════
#  STEP 5 — FEATURE CORRELATION WITH LABEL
# ══════════════════════════════════════════════════════════════
print("\n📊 Step 5: Feature correlation with label...")

correlations = df[FEATURE_COLS + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
correlations_sorted = correlations.abs().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 10))
fig.suptitle("EDA — Chart 9: Feature Correlation with Label (|r|)",
             fontsize=14, fontweight="bold")

colors_corr = ["#00C853" if correlations[f] > 0 else "#FF4D4D"
               for f in correlations_sorted.index]
ax.barh(correlations_sorted.index, correlations_sorted.values, color=colors_corr)
ax.set_xlabel("Absolute Correlation with Label")
ax.axvline(x=0.05, color="gray", linestyle="--", alpha=0.5, label="0.05 threshold")
ax.legend()

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart9_feature_correlation.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 9: Feature correlations")

# Print top 10 most correlated features
print("\n  Top 10 features by correlation with label:")
for feat, val in correlations_sorted.head(10).items():
    direction = "↑" if correlations[feat] > 0 else "↓"
    print(f"    {direction} {feat:<35} r = {correlations[feat]:+.4f}")


# ══════════════════════════════════════════════════════════════
#  STEP 6 — SAVE MODEL INPUT
# ══════════════════════════════════════════════════════════════
print("\n💾 Step 6: Saving model_input.csv...")

model_df = df[ID_COLS + FEATURE_COLS + [TARGET_COL]]
model_df.to_csv(f"{OUTPUT_DIR}/model_input.csv", index=False)
print(f"  ✓ model_input.csv saved → {model_df.shape[0]:,} rows × {model_df.shape[1]} columns")

# Save feature list for reference in Phase 4
with open(f"{OUTPUT_DIR}/feature_list.txt", "w") as f:
    f.write("# CSAO Model — Final Feature List (Phase 3 output)\n")
    f.write(f"# Total features: {len(FEATURE_COLS)}\n\n")
    for i, feat in enumerate(FEATURE_COLS, 1):
        corr_val = correlations.get(feat, 0)
        f.write(f"{i:02d}. {feat:<40} corr={corr_val:+.4f}\n")

print(f"  ✓ feature_list.txt saved")


# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("✅ Phase 3 Complete!")
print("="*60)
print(f"\n📁 Output files saved to: {OUTPUT_DIR}/")
print(f"   model_input.csv   → {model_df.shape[0]:,} rows × {model_df.shape[1]} cols")
print(f"   feature_list.txt  → {len(FEATURE_COLS)} features documented")
print(f"\n📊 EDA charts saved to: {CHART_DIR}/")
charts = [f for f in os.listdir(CHART_DIR) if f.endswith(".png")]
for c in sorted(charts):
    print(f"   {c}")

print(f"""
📌 New features engineered ({len(FEATURE_COLS) - 23} added on top of original 23):
   • meal_completeness_score     — how many meal components are in cart
   • is_complementary_category   — does item fill a missing meal slot?
   • veg_match / veg_mismatch    — user preference vs item type
   • affordability               — item price vs user's avg spend
   • upsell_pct                  — how much item increases cart value
   • popularity_score            — normalized from rank
   • cooccur_log                 — log-transformed co-occurrence
   • step_ratio                  — session progress (0→1)
   • meal_time_encoded           — ordinal encoding of meal time
   • is_dinner_or_late           — high add-on tendency flag
   • segment_encoded             — ordinal user segment
   • is_tier1_city               — city tier flag
   • cand_category_encoded       — encoded item category
   • weekend_premium             — weekend × premium interaction

🎯 Next step: Phase 4 — Train LightGBM Ranking Model!
""")
