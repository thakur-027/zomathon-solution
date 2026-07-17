# 🍽️ MealMind — AI-Powered Cart Add-On Recommendation System

> **Zomathon Hackathon 2026 — Problem Statement 2**  
> Built for Zomato's Cart Super Add-On (CSAO) Rail Challenge

---

## 📌 Overview

**MealMind** is an intelligent, real-time recommendation system that suggests relevant add-on items to customers based on their current cart composition, contextual factors, and historical behavior patterns.

When a customer adds **Chicken Biryani** to their cart, MealMind understands the meal is incomplete and recommends **Raita → Mango Lassi → Gulab Jamun** — in that order — because that's how real meals are built.

---

## 🎯 Problem Statement

> *How can we build an intelligent recommendation system that suggests relevant add-on items to customers based on their current cart composition, contextual factors, and historical behavior patterns, while maintaining high acceptance rates and customer satisfaction?*

---

## 🏗️ System Architecture

```
Customer adds item to cart
         ↓
Feature Store (pre-computed, Redis, < 10ms)
         ↓
Stage 1 — Co-occurrence Retrieval → 50 candidates
         ↓
Stage 2 — LightGBM LambdaRank → Top 10 scored items
         ↓
LLM Layer — Gemini Meal Completeness Analysis (async)
         ↓
Top 8 Recommendations returned in < 300ms
```

---

## 🚀 Key Features

- **Two-stage recommendation pipeline** — fast retrieval + smart ranking
- **LLM integration (Google Gemini)** — meal completeness reasoning and cold-start profiling
- **Real-time dynamic updates** — recommendations change as cart items are added
- **Cold start handling** — LLM infers preferences for new users from city, time, and segment
- **< 300ms latency** — production-ready architecture
- **Interactive demo** — single HTML file, no server required

---

## 📊 Model Performance

| Metric | Baseline | MealMind | Improvement |
|--------|----------|----------|-------------|
| AUC-ROC | 0.7530 | **0.7931** | ↑ +5.3% |
| NDCG@8 | — | **0.4968** | — |
| NDCG@10 | — | **0.5622** | — |
| P@10 | — | **0.5000** | — |

**💰 Estimated Business Impact:**
- Average **1.68 extra items** added per session
- Average **₹449 AOV lift** per session
- **21% acceptance rate** on top-8 recommendations

---

## 🧠 LLM Integration (AI Edge)

### Use 1 — Meal Completeness Analyzer
Gemini analyzes the current cart and identifies missing meal components:
```
Cart: Chicken Biryani (dinner)
→ Missing: raita/side dish, dessert, beverage
→ Completeness Score: 4/10
→ Reasoning: "Biryani is a complete main course but needs 
   cooling accompaniments to balance the spices."
```

### Use 2 — Cold Start Profile Generator
For new users with no history, Gemini infers preferences from context:
```
User: Delhi | Non-veg | Dinner | Premium
→ Likely cuisines: Continental, North Indian, Italian
→ Price sensitivity: Low
→ Vibe: "Premium multi-course dinner experience"
→ Confidence: 8/10
```

---

## 📁 Project Structure

The repo is organized by pipeline phase, each folder self-contained with its script, inputs/outputs, and charts.

```
zomathon-solution/
├── phase2_data/
│   ├── generate_data.py          # Synthetic data generation
│   └── raw/                      # Generated datasets (*.csv)
│
├── phase3_features/
│   ├── feature_engineering.py    # EDA + feature engineering
│   ├── model_input.csv           # Feature-rich training dataset
│   ├── feature_list.txt          # Documented feature list
│   └── eda_charts/               # 9 EDA visualizations (generated on run)
│
├── phase4_model/
│   ├── train_model.py            # LightGBM LambdaRank training
│   ├── csao_model.pkl            # Trained model
│   ├── model_results.csv
│   ├── feature_importance.csv
│   └── charts/                   # 5 model evaluation charts
│
├── phase5_llm/
│   ├── llm_integration.py        # Gemini LLM integration
│   ├── llm_meal_analysis.csv
│   ├── llm_cold_start_profiles.csv
│   ├── model_input_with_llm.csv
│   └── charts/                   # 3 LLM analysis charts
│
├── phase6_report/
│   ├── CSAO_Recommendation_System_Report.pdf
│   └── system_design.png
│
└── demo/
    ├── index.html             # 🎮 Interactive live demo markup
    ├── style.css              # Demo styling
    └── script.js              # Demo logic (cart state, recommendations)
```

> **Note:** the two duplicate copies of `index.html` (root + `files/`) from the old layout have been de-duplicated, and the single demo file has been split into `index.html` / `style.css` / `script.js`. Each phase script's `DATA_DIR`/`OUTPUT_DIR`/`CHART_DIR` variables have been updated to point at the new relative paths so the pipeline still runs end-to-end in order.

---

## 📦 Datasets Generated

| File | Rows | Description |
|------|------|-------------|
| `cart_sessions.csv` | 119,517 | Core training data — one row per recommendation moment |
| `orders.csv` | 15,000 | Historical orders with temporal context |
| `order_items.csv` | 34,641 | Item-level order details for co-occurrence |
| `menu_items.csv` | 6,995 | Restaurant menus across 7 cities |
| `restaurants.csv` | 500 | Restaurant profiles |
| `users.csv` | 2,000 | User profiles with segments & preferences |

All live in `phase2_data/raw/`.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- pip

### Install Dependencies
```bash
pip install pandas numpy matplotlib scikit-learn lightgbm google-generativeai
```

### Run Phase by Phase

```bash
# Phase 2 — Generate synthetic data
cd phase2_data
python generate_data.py

# Phase 3 — Feature engineering + EDA
cd ../phase3_features
python feature_engineering.py

# Phase 4 — Train the model
cd ../phase4_model
python train_model.py

# Phase 5 — LLM integration
# Set GEMINI_API_KEY in your shell before running
cd ../phase5_llm
python llm_integration.py
```

> `phase6_report/` contains the pre-generated PDF report and system design diagram as static deliverables — the scripts that originally produced them weren't part of this repo, so they aren't reproducible via a `python` command here.

### 🎮 Run the Demo
Just open `demo/index.html` in any browser — no server needed. It loads `style.css` and `script.js` from the same folder.

---

## 🔑 Gemini API Key Setup

1. Visit [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with Google and click **Create API Key**
3. Set it in your environment before running:
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```
4. Run `python llm_integration.py` (from inside `phase5_llm/`)

> Keep secrets out of source control. Use environment variables or a local `.env` file that is ignored by Git.

---

## 📈 Feature Engineering Highlights

**34 features** engineered across 5 groups:

- **Cart State** — meal completeness score, has_beverage, has_dessert, cart total
- **Candidate Item** — price, category, popularity score, veg flag
- **Cart-Item Relationship** — co-occurrence score (strongest signal, r=+0.30), complementary category flag, price ratio
- **User** — segment, avg spend, city tier, veg preference
- **Context** — meal time, weekend flag, dinner/late-night flag

---

## 🧪 A/B Testing Design

| Group | Strategy | Sample |
|-------|----------|--------|
| Control (50%) | Current popularity-based recommendations | ~50K sessions |
| Treatment (50%) | MealMind LightGBM + Gemini | ~50K sessions |

**Primary Metrics:** AOV lift, CSAO attach rate, acceptance rate  
**Guardrail Metrics:** Cart abandonment rate (must not increase >1%), latency (must stay <300ms)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | LightGBM LambdaRank |
| LLM | Google Gemini 1.5 Flash |
| Data Processing | Python, Pandas, NumPy |
| Visualization | Matplotlib |
| Demo | Vanilla HTML/CSS/JS |
| Feature Store (prod) | Redis |

---

## 📄 Submission

- **PDF Report:** `phase6_report/CSAO_Recommendation_System_Report.pdf`
- **Code:** This repository
- **Demo:** `demo/index.html`

---

## 👨‍💻 Team

**BitByte** — Zomathon 2026
Submitted for **Problem Statement 2: Cart Super Add-On (CSAO) Rail Recommendation System**

---
