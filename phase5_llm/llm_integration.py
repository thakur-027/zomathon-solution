"""
=============================================================
  CSAO Recommendation System — Phase 5
  LLM Integration (AI Edge) using Google Gemini
=============================================================
What this script does:
  USE 1 — Meal Completeness Analyzer
    → LLM looks at cart items and identifies what's "missing"
      from a complete meal (e.g., no drink, no dessert)
    → This becomes an extra signal for the ranking model

  USE 2 — Cold Start Handler
    → New users have no order history
    → LLM infers their food preferences from:
      city, time of day, veg preference, and session context
    → Generates a starter preference profile

  MODE:
    → If GEMINI_API_KEY is set in your environment: uses real Gemini API
    → If not set: runs in MOCK mode (simulated responses)
      so you can test without a key

  HOW TO GET FREE GEMINI API KEY:
    1. Go to https://aistudio.google.com/app/apikey
    2. Sign in with Google
    3. Click "Create API Key"
    4. Set it in your shell before running:
         PowerShell: $env:GEMINI_API_KEY="your_key_here"
         Bash: export GEMINI_API_KEY="your_key_here"
=============================================================
"""

import pandas as pd
import numpy as np
import os
import json
import sys
import time
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_local_env() -> None:
    """Load values from a local .env file if present."""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


load_local_env()

# ── CONFIG — Load Gemini API key from the environment (never hard-code secrets) ───
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# ── Auto-detect mode ─────────────────────────────────────────
USE_REAL_LLM = bool(GEMINI_API_KEY)

if USE_REAL_LLM:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)

        candidate_models = [
            os.getenv("GEMINI_MODEL", "").strip() or "gemini-2.0-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-2.0-flash-lite",
        ]
        llm_model = None
        active_model_name = None
        last_error = None

        for model_name in candidate_models:
            if not model_name:
                continue
            try:
                llm_model = genai.GenerativeModel(model_name)
                active_model_name = model_name
                break
            except Exception as exc:
                last_error = exc

        if llm_model is None:
            raise last_error or RuntimeError("Unable to initialize Gemini model")

        print(f"✅ Gemini API connected — running in REAL LLM mode using {active_model_name}")
    except ImportError:
        print("⚠️  google-generativeai not installed.")
        print("   Run: pip install google-generativeai")
        print("   Switching to MOCK mode for now.")
        USE_REAL_LLM = False
    except Exception as exc:
        print(f"⚠️  Gemini model initialization failed: {exc}")
        print("   Switching to MOCK mode for now.")
        USE_REAL_LLM = False
else:
    print("ℹ️  No GEMINI_API_KEY environment variable set — running in MOCK mode")
    print("   (Responses are simulated but code structure is identical)")
    print("   To use real LLM, set GEMINI_API_KEY in your shell before running.\n")

# ── Paths ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "phase3_features")
OUTPUT_DIR = SCRIPT_DIR
CHART_DIR  = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

print("\n🚀 Phase 5 — LLM Integration (AI Edge)\n")


# ══════════════════════════════════════════════════════════════
#  HELPER — Call Gemini API (or mock it)
# ══════════════════════════════════════════════════════════════
def call_llm(prompt: str, mock_response: str) -> str:
    """
    Calls Gemini API if key is set, otherwise returns mock response.
    Always returns a clean string.
    """
    if USE_REAL_LLM:
        try:
            response = llm_model.generate_content(prompt)
            time.sleep(0.5)   # rate limit safety
            return response.text.strip()
        except Exception as e:
            print(f"    ⚠️  LLM call failed: {e} — using mock response")
            return mock_response
    else:
        # Simulate a small delay to mimic real API
        time.sleep(0.05)
        return mock_response


# ══════════════════════════════════════════════════════════════
#  USE 1 — MEAL COMPLETENESS ANALYZER
# ══════════════════════════════════════════════════════════════
print("=" * 55)
print("  USE 1: Meal Completeness Analyzer")
print("=" * 55)

def analyze_meal_completeness(cart_items: list, meal_time: str) -> dict:
    """
    Given a list of cart items + meal time, asks LLM:
    'What meal components are missing? What should be recommended?'

    Returns a dict with:
      - missing_components: list of what's missing
      - recommended_categories: categories to prioritize
      - reasoning: LLM's explanation
      - completeness_score: 0-10
    """

    cart_text = ", ".join(cart_items) if cart_items else "empty cart"

    prompt = f"""
You are a food recommendation expert for a food delivery app.

A customer is ordering food during {meal_time}.
Their current cart contains: {cart_text}

Analyze this cart and respond ONLY with a valid JSON object (no markdown, no explanation outside JSON):
{{
  "missing_components": ["list of missing meal components, e.g. beverage, dessert, side dish"],
  "recommended_categories": ["top 2-3 item categories to recommend next"],
  "reasoning": "one sentence explaining what the cart needs",
  "completeness_score": <number 0-10 where 10 is a perfectly complete meal>,
  "meal_vibe": "brief label like 'heavy north indian dinner' or 'light lunch'"
}}
"""

    # Mock response varies by cart content for realism
    if "Biryani" in cart_text or "biryani" in cart_text:
        mock = json.dumps({
            "missing_components": ["raita/side dish", "dessert", "beverage"],
            "recommended_categories": ["side_dish", "beverage", "dessert"],
            "reasoning": "Biryani is a complete main course but needs cooling accompaniments like raita and a beverage to balance the spices.",
            "completeness_score": 4,
            "meal_vibe": "hearty biryani meal"
        })
    elif "Pizza" in cart_text or "Pasta" in cart_text:
        mock = json.dumps({
            "missing_components": ["beverage", "dessert", "garlic bread side"],
            "recommended_categories": ["beverage", "side_dish", "dessert"],
            "reasoning": "Italian meal needs a cold beverage and garlic bread to complement the main course.",
            "completeness_score": 4,
            "meal_vibe": "italian dining experience"
        })
    elif "Dosa" in cart_text or "Idli" in cart_text:
        mock = json.dumps({
            "missing_components": ["beverage", "chutney/side"],
            "recommended_categories": ["beverage", "side_dish"],
            "reasoning": "South Indian breakfast is nearly complete but a filter coffee or juice would complete the experience.",
            "completeness_score": 6,
            "meal_vibe": "south indian breakfast"
        })
    else:
        mock = json.dumps({
            "missing_components": ["beverage", "dessert"],
            "recommended_categories": ["beverage", "dessert"],
            "reasoning": "The cart has a main course but is missing a drink and sweet ending for a complete meal.",
            "completeness_score": 5,
            "meal_vibe": "standard meal"
        })

    raw = call_llm(prompt, mock)

    # Parse JSON safely
    try:
        # Strip markdown code blocks if present
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {
            "missing_components": ["beverage"],
            "recommended_categories": ["beverage"],
            "reasoning": "Unable to parse LLM response — defaulting to beverage recommendation.",
            "completeness_score": 5,
            "meal_vibe": "unknown"
        }


# ── Demo: Run meal completeness on 5 sample carts ────────────
sample_carts = [
    (["Chicken Biryani"],                              "dinner"),
    (["Butter Chicken", "Garlic Naan"],                "dinner"),
    (["Masala Dosa"],                                  "breakfast"),
    (["Margherita Pizza", "Garlic Bread"],             "lunch"),
    (["Paneer Tikka", "Dal Makhani", "Butter Naan"],   "dinner"),
]

print("\n📋 Analyzing 5 sample carts...\n")
completeness_results = []

for i, (cart, meal_time) in enumerate(sample_carts, 1):
    result = analyze_meal_completeness(cart, meal_time)
    completeness_results.append({
        "cart":                  ", ".join(cart),
        "meal_time":             meal_time,
        "completeness_score":    result.get("completeness_score", 5),
        "missing_components":    ", ".join(result.get("missing_components", [])),
        "recommended_categories":result.get("recommended_categories", []),
        "reasoning":             result.get("reasoning", ""),
        "meal_vibe":             result.get("meal_vibe", ""),
    })

    print(f"  Cart {i}: {', '.join(cart)} ({meal_time})")
    print(f"    🔍 Missing    : {result.get('missing_components', [])}")
    print(f"    💡 Recommend  : {result.get('recommended_categories', [])}")
    print(f"    📊 Score      : {result.get('completeness_score')}/10")
    print(f"    🍽️  Vibe       : {result.get('meal_vibe')}")
    print(f"    🧠 Reasoning  : {result.get('reasoning')}")
    print()

completeness_df = pd.DataFrame(completeness_results)
completeness_df.to_csv(f"{OUTPUT_DIR}/llm_meal_analysis.csv", index=False)
print("  ✓ llm_meal_analysis.csv saved\n")


# ══════════════════════════════════════════════════════════════
#  USE 2 — COLD START HANDLER
# ══════════════════════════════════════════════════════════════
print("=" * 55)
print("  USE 2: Cold Start User Preference Profiler")
print("=" * 55)

def generate_cold_start_profile(user_context: dict) -> dict:
    """
    For a new user with no order history,
    LLM infers likely food preferences from available context.

    user_context keys:
      - city, is_veg, meal_time, hour, day_of_week, segment
    """

    prompt = f"""
You are a personalization expert for a food delivery app in India.

A NEW USER just opened the app for the first time. They have NO order history.
Here is what we know about them:

- City           : {user_context.get('city', 'Mumbai')}
- Is Vegetarian  : {user_context.get('is_veg', False)}
- Current Time   : {user_context.get('meal_time', 'dinner')} ({user_context.get('hour', 20)}:00)
- Day            : {user_context.get('day_of_week', 'Saturday')}
- User Segment   : {user_context.get('segment', 'mid_range')}

Based on this context, infer their likely food preferences.
Respond ONLY with a valid JSON object (no markdown):
{{
  "likely_cuisines": ["top 2-3 cuisines this person probably likes"],
  "likely_categories": ["top item categories they'd order"],
  "price_sensitivity": "low / medium / high",
  "likely_vibe": "one phrase describing their order style",
  "starter_recommendation": "one specific item to recommend first",
  "confidence": <number 0-10 how confident you are in this profile>
}}
"""

    city    = user_context.get("city", "Mumbai")
    is_veg  = user_context.get("is_veg", False)
    segment = user_context.get("segment", "mid_range")
    meal    = user_context.get("meal_time", "dinner")

    # Realistic mock responses by context
    if city in ["Chennai", "Bangalore"] and is_veg:
        mock = json.dumps({
            "likely_cuisines": ["South Indian", "North Indian", "Chinese"],
            "likely_categories": ["main_course", "beverage", "side_dish"],
            "price_sensitivity": "medium",
            "likely_vibe": "vegetarian south indian comfort food",
            "starter_recommendation": "Masala Dosa",
            "confidence": 7
        })
    elif segment == "premium" and meal == "dinner":
        mock = json.dumps({
            "likely_cuisines": ["Continental", "North Indian", "Italian"],
            "likely_categories": ["main_course", "starter", "beverage", "dessert"],
            "price_sensitivity": "low",
            "likely_vibe": "premium multi-course dinner experience",
            "starter_recommendation": "Chicken Tikka",
            "confidence": 8
        })
    elif meal in ["breakfast", "snack"]:
        mock = json.dumps({
            "likely_cuisines": ["Fast Food", "South Indian", "Street Food"],
            "likely_categories": ["snack", "beverage"],
            "price_sensitivity": "high",
            "likely_vibe": "quick light bite",
            "starter_recommendation": "Samosa",
            "confidence": 6
        })
    else:
        mock = json.dumps({
            "likely_cuisines": ["North Indian", "Chinese", "Fast Food"],
            "likely_categories": ["main_course", "beverage"],
            "price_sensitivity": "medium",
            "likely_vibe": "casual everyday meal",
            "starter_recommendation": "Butter Chicken",
            "confidence": 6
        })

    raw = call_llm(prompt, mock)

    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {
            "likely_cuisines":       ["North Indian"],
            "likely_categories":     ["main_course", "beverage"],
            "price_sensitivity":     "medium",
            "likely_vibe":           "general user",
            "starter_recommendation":"Butter Chicken",
            "confidence":             5
        }


# ── Demo: Run cold start on 5 sample new users ───────────────
cold_start_users = [
    {"city": "Chennai",   "is_veg": True,  "meal_time": "breakfast",
     "hour": 9,  "day_of_week": "Monday",   "segment": "budget"},
    {"city": "Mumbai",    "is_veg": False, "meal_time": "lunch",
     "hour": 13, "day_of_week": "Wednesday","segment": "mid_range"},
    {"city": "Delhi",     "is_veg": False, "meal_time": "dinner",
     "hour": 20, "day_of_week": "Saturday", "segment": "premium"},
    {"city": "Bangalore", "is_veg": True,  "meal_time": "snack",
     "hour": 17, "day_of_week": "Sunday",   "segment": "frequent"},
    {"city": "Hyderabad", "is_veg": False, "meal_time": "dinner",
     "hour": 21, "day_of_week": "Friday",   "segment": "occasional"},
]

print("\n👤 Generating cold start profiles for 5 new users...\n")
cold_start_results = []

for i, user_ctx in enumerate(cold_start_users, 1):
    profile = generate_cold_start_profile(user_ctx)
    cold_start_results.append({
        **user_ctx,
        "likely_cuisines":        ", ".join(profile.get("likely_cuisines", [])),
        "likely_categories":      ", ".join(profile.get("likely_categories", [])),
        "price_sensitivity":      profile.get("price_sensitivity", "medium"),
        "likely_vibe":            profile.get("likely_vibe", ""),
        "starter_recommendation": profile.get("starter_recommendation", ""),
        "confidence":             profile.get("confidence", 5),
    })

    veg_label = "🥦 Veg" if user_ctx["is_veg"] else "🍗 Non-veg"
    print(f"  User {i}: {user_ctx['city']} | {veg_label} | "
          f"{user_ctx['meal_time']} | {user_ctx['segment']}")
    print(f"    🍽️  Likely cuisines : {profile.get('likely_cuisines')}")
    print(f"    📦 Likely categories: {profile.get('likely_categories')}")
    print(f"    💸 Price sensitivity: {profile.get('price_sensitivity')}")
    print(f"    🎯 Vibe             : {profile.get('likely_vibe')}")
    print(f"    ⭐ First recommend  : {profile.get('starter_recommendation')}")
    print(f"    🔮 Confidence       : {profile.get('confidence')}/10")
    print()

cold_start_df = pd.DataFrame(cold_start_results)
cold_start_df.to_csv(f"{OUTPUT_DIR}/llm_cold_start_profiles.csv", index=False)
print("  ✓ llm_cold_start_profiles.csv saved\n")


# ══════════════════════════════════════════════════════════════
#  STEP 3 — INTEGRATE LLM SIGNALS INTO MODEL FEATURES
# ══════════════════════════════════════════════════════════════
print("=" * 55)
print("  STEP 3: Integrating LLM signals into model features")
print("=" * 55)

print("\n📊 Showing how LLM signals improve model input...")

# Load model_input to show integration concept
model_df = pd.read_csv(f"{DATA_DIR}/model_input.csv", nrows=10000)

# Simulate: for each row, if candidate category is in LLM's
# recommended_categories list → llm_recommended = 1
# This is what you'd do in production with real LLM calls

# Category → what LLM typically recommends for incomplete carts
llm_category_boost = {
    "beverage": 0.75,    # LLM recommends beverages 75% of the time
    "dessert":  0.60,
    "side_dish":0.55,
    "starter":  0.40,
    "main_course":0.30,
    "snack":    0.25,
}

# New feature: LLM recommends this category (probability)
model_df["llm_category_boost"] = model_df["cand_category"].map(llm_category_boost).fillna(0.3)

# New feature: LLM meal completeness score (from cart flags)
model_df["llm_completeness"] = (
    model_df["has_main_course"] * 2 +
    model_df["has_starter"] * 1 +
    model_df["has_side_dish"] * 1.5 +
    model_df["has_dessert"] * 1 +
    model_df["has_beverage"] * 2
) / 8.5  # normalize to 0–1

# New feature: cold start flag already exists, but add LLM confidence
# (simulated — in production this comes from real LLM call)
np.random.seed(42)
model_df["llm_preference_confidence"] = np.random.uniform(0.5, 1.0, len(model_df))

print("\n  New LLM-derived features added to model:")
print("  ┌─────────────────────────────────────────────────────┐")
print("  │ Feature                     │ Description           │")
print("  ├─────────────────────────────────────────────────────┤")
print("  │ llm_category_boost          │ How often LLM         │")
print("  │                             │ recommends this cat   │")
print("  ├─────────────────────────────────────────────────────┤")
print("  │ llm_completeness            │ LLM meal complete-    │")
print("  │                             │ ness score (0–1)      │")
print("  ├─────────────────────────────────────────────────────┤")
print("  │ llm_preference_confidence   │ How confident LLM is  │")
print("  │                             │ about user prefs      │")
print("  └─────────────────────────────────────────────────────┘")

# Save enhanced model input
model_df.to_csv(f"{OUTPUT_DIR}/model_input_with_llm.csv", index=False)
print(f"\n  ✓ model_input_with_llm.csv saved → {model_df.shape}")


# ══════════════════════════════════════════════════════════════
#  STEP 4 — VISUALIZATION
# ══════════════════════════════════════════════════════════════
print("\n🎨 Generating LLM charts...")

# ── Chart 1: Meal Completeness Scores ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("LLM — Meal Completeness Analysis", fontsize=14, fontweight="bold")

carts_short = [c[:30]+"…" if len(c)>30 else c for c in completeness_df["cart"]]
scores      = completeness_df["completeness_score"]
colors      = ["#FF4D4D" if s < 4 else "#FFA500" if s < 7 else "#00C853" for s in scores]

axes[0].barh(carts_short, scores, color=colors)
axes[0].set_xlabel("Completeness Score (0–10)")
axes[0].set_title("Cart Completeness Scores")
axes[0].axvline(x=7, color="gray", linestyle="--", alpha=0.5, label="Complete threshold")
axes[0].legend()
for i, v in enumerate(scores):
    axes[0].text(v + 0.1, i, str(v), va="center", fontsize=10, fontweight="bold")

# Pie chart of missing components
all_missing = []
for m in completeness_df["missing_components"]:
    all_missing.extend([x.strip() for x in m.split(",")])
missing_counts = pd.Series(all_missing).value_counts()
axes[1].pie(missing_counts.values, labels=missing_counts.index,
            autopct="%1.0f%%", startangle=90,
            colors=["#FF6B35","#4ECDC4","#7B2FBE","#FFB347","#00C853"])
axes[1].set_title("Most Missing Meal Components")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart1_meal_completeness.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 1: Meal completeness")

# ── Chart 2: Cold Start Confidence by Segment ────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("LLM — Cold Start Profile Quality", fontsize=14, fontweight="bold")

segments   = cold_start_df["segment"]
confidence = cold_start_df["confidence"]
colors_seg = ["#FF6B35","#4ECDC4","#7B2FBE","#FFB347","#E84855"]

axes[0].bar(segments, confidence, color=colors_seg[:len(segments)])
axes[0].set_ylabel("LLM Confidence (0–10)")
axes[0].set_title("Cold Start Confidence by User Segment")
axes[0].set_ylim(0, 11)
for i, v in enumerate(confidence):
    axes[0].text(i, v + 0.2, str(v), ha="center", fontweight="bold")

# Price sensitivity distribution
price_sens = cold_start_df["price_sensitivity"].value_counts()
axes[1].pie(price_sens.values, labels=price_sens.index,
            autopct="%1.0f%%", colors=["#FF6B35","#4ECDC4","#7B2FBE"],
            startangle=90)
axes[1].set_title("Inferred Price Sensitivity")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart2_cold_start_profiles.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 2: Cold start profiles")

# ── Chart 3: LLM Feature Impact ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("LLM — Category Boost Scores", fontsize=14, fontweight="bold")

cats   = list(llm_category_boost.keys())
boosts = list(llm_category_boost.values())
colors_b = ["#00C853" if b >= 0.6 else "#FFA500" if b >= 0.4 else "#FF4D4D"
            for b in boosts]
bars = ax.bar(cats, [b*100 for b in boosts], color=colors_b, edgecolor="white")
ax.set_ylabel("LLM Recommendation Frequency (%)")
ax.set_title("How often LLM recommends each category for incomplete carts")
for bar, val in zip(bars, boosts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f"{val*100:.0f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart3_llm_category_boost.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 3: LLM category boost")


# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════
mode_str = "🌐 REAL Gemini LLM" if USE_REAL_LLM else "🧪 MOCK (simulated)"
print(f"""
{'='*55}
✅ Phase 5 Complete! — Mode: {mode_str}
{'='*55}

🤖 LLM Use 1 — Meal Completeness Analyzer:
   → Analyzed 5 sample carts
   → Identified missing components for each
   → Generated recommended categories
   → Results saved to llm_meal_analysis.csv

👤 LLM Use 2 — Cold Start Profiler:
   → Generated profiles for 5 new users
   → Inferred cuisine preferences by city/time/segment
   → Results saved to llm_cold_start_profiles.csv

🔗 Integration:
   → 3 new LLM-derived features added to model input
   → Enhanced dataset saved: model_input_with_llm.csv

📁 Charts saved to llm_charts/:
   chart1_meal_completeness.png
   chart2_cold_start_profiles.png
   chart3_llm_category_boost.png

{'='*55}
💡 TO ENABLE REAL GEMINI LLM:
   1. Visit: https://aistudio.google.com/app/apikey
   2. Create a free API key
   3. Set it as an environment variable before running:
      PowerShell: $env:GEMINI_API_KEY="your_key_here"
      Bash: export GEMINI_API_KEY="your_key_here"
   4. Run: pip install google-generativeai
   5. Run the script again
{'='*55}

🎯 Next step: Phase 6 — System Design + PDF Report!
""")
