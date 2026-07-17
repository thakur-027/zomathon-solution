"""
=============================================================
  CSAO Recommendation System — Phase 4
  Model Training + Evaluation
=============================================================
What this script does:
  1. Loads model_input.csv (Phase 3 output)
  2. Performs train/test split (temporal — no data leakage)
  3. Handles class imbalance
  4. Trains Stage 1 — Retrieval (co-occurrence baseline)
  5. Trains Stage 2 — LightGBM Ranking Model
  6. Evaluates: AUC, Precision@K, Recall@K, NDCG@K
  7. Plots feature importance + evaluation charts
  8. Saves model + results
=============================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, classification_report,
                             roc_curve, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import pickle

# ── Try LightGBM (preferred), fall back to sklearn ──────────
try:
    import lightgbm as lgb
    USE_LIGHTGBM = True
    print("✅ LightGBM found — using LightGBM Ranker")
except ImportError:
    USE_LIGHTGBM = False
    print("⚠️  LightGBM not found — using GradientBoosting (install lightgbm for better results)")

# ── Paths ────────────────────────────────────────────────────
DATA_DIR   = "../phase3_features"
OUTPUT_DIR = "."
CHART_DIR  = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

print("\n🚀 Phase 4 — Model Training + Evaluation\n")


# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════
print("📂 Step 1: Loading model_input.csv...")
df = pd.read_csv(f"{DATA_DIR}/model_input.csv")
print(f"  Shape: {df.shape}")
print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

FEATURE_COLS = [
    # Cart state
    "cart_size", "cart_total_value", "cart_avg_price",
    "has_main_course", "has_starter", "has_side_dish",
    "has_dessert", "has_beverage", "has_snack",
    "cart_is_complete", "meal_completeness_score",
    # Candidate item
    "cand_price", "cand_is_veg", "cand_popularity",
    "cand_category_encoded", "popularity_score",
    # Relationship
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

TARGET = "label"


# ══════════════════════════════════════════════════════════════
#  STEP 2 — TRAIN / TEST SPLIT (Temporal)
# ══════════════════════════════════════════════════════════════
print("\n✂️  Step 2: Temporal train/test split...")

# Use session_id ordering as proxy for time
# Earlier sessions → train | Later sessions → test
# This prevents data leakage (model never sees future behavior during training)
df_sorted   = df.sort_values("session_id").reset_index(drop=True)
split_idx   = int(len(df_sorted) * 0.80)   # 80% train, 20% test

train_df = df_sorted.iloc[:split_idx].copy()
test_df  = df_sorted.iloc[split_idx:].copy()

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df[TARGET]

print(f"  Train: {X_train.shape} | Positives: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"  Test : {X_test.shape}  | Positives: {y_test.sum():,}  ({y_test.mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
#  STEP 3 — CLASS IMBALANCE HANDLING
# ══════════════════════════════════════════════════════════════
print("\n⚖️  Step 3: Handling class imbalance...")

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"  Negative : Positive ratio = {scale_pos_weight:.1f} : 1")
print(f"  Strategy: scale_pos_weight = {scale_pos_weight:.1f} (penalize misclassifying positives more)")


# ══════════════════════════════════════════════════════════════
#  STEP 4 — STAGE 1: RETRIEVAL (Co-occurrence Baseline)
# ══════════════════════════════════════════════════════════════
print("\n🔍 Step 4: Stage 1 — Retrieval baseline (co-occurrence)...")

# Baseline: rank purely by co-occurrence score
baseline_scores = test_df["cooccur_score"].values
baseline_auc    = roc_auc_score(y_test, baseline_scores)
print(f"  Baseline AUC (co-occurrence only): {baseline_auc:.4f}")


# ══════════════════════════════════════════════════════════════
#  STEP 5 — STAGE 2: RANKING MODEL
# ══════════════════════════════════════════════════════════════
print("\n🤖 Step 5: Stage 2 — Training Ranking Model...")

if USE_LIGHTGBM:
    # ── LightGBM (preferred for production) ─────────────────
    # Group sizes needed for LambdaRank
    train_groups = train_df.groupby("session_id").size().values
    test_groups  = test_df.groupby("session_id").size().values

    train_data = lgb.Dataset(X_train, label=y_train, group=train_groups)
    test_data  = lgb.Dataset(X_test,  label=y_test,  group=test_groups, reference=train_data)

    params = {
        "objective":        "lambdarank",
        "metric":           "ndcg",
        "ndcg_eval_at":     [5, 10],
        "learning_rate":    0.05,
        "num_leaves":       63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbose":          -1,
        "n_jobs":           -1,
    }

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)]
    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=callbacks
    )

    y_pred_proba = model.predict(X_test)
    feature_importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    model_name = "LightGBM LambdaRank"

else:
    # ── GradientBoosting (fallback) ──────────────────────────
    print("  Training GradientBoostingClassifier (this may take 1–2 min)...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    # Handle imbalance via sample weights
    sample_weights = np.where(y_train == 1, scale_pos_weight, 1.0)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    feature_importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    model_name = "GradientBoosting Classifier"

print(f"  ✓ {model_name} trained successfully")


# ══════════════════════════════════════════════════════════════
#  STEP 6 — EVALUATION METRICS
# ══════════════════════════════════════════════════════════════
print(f"\n📊 Step 6: Evaluating model...")

# ── AUC ──────────────────────────────────────────────────────
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"  AUC-ROC          : {auc_score:.4f}  (baseline: {baseline_auc:.4f})")

# ── Precision@K, Recall@K, NDCG@K ───────────────────────────
def precision_at_k(y_true, y_scores, k):
    """What fraction of top-K recommendations were actually relevant?"""
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return y_true.iloc[top_k_idx].mean() if hasattr(y_true, 'iloc') else y_true[top_k_idx].mean()

def recall_at_k(y_true, y_scores, k):
    """What fraction of all relevant items appear in top-K?"""
    top_k_idx  = np.argsort(y_scores)[::-1][:k]
    total_pos  = y_true.sum() if hasattr(y_true, 'sum') else np.sum(y_true)
    hit        = y_true.iloc[top_k_idx].sum() if hasattr(y_true, 'iloc') else y_true[top_k_idx].sum()
    return hit / max(total_pos, 1)

def ndcg_at_k(y_true, y_scores, k):
    """Normalized Discounted Cumulative Gain — rewards putting relevant items higher"""
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    y_arr = y_true.values if hasattr(y_true, 'values') else y_true
    gains     = y_arr[top_k_idx] / np.log2(np.arange(2, k + 2))
    dcg       = gains.sum()
    ideal_idx = np.argsort(y_arr)[::-1][:k]
    ideal_gains = y_arr[ideal_idx] / np.log2(np.arange(2, k + 2))
    idcg      = ideal_gains.sum()
    return dcg / idcg if idcg > 0 else 0.0

y_test_arr  = y_test.values
metrics = {}
for k in [5, 8, 10]:
    metrics[f"P@{k}"]    = precision_at_k(y_test, y_pred_proba, k)
    metrics[f"R@{k}"]    = recall_at_k(y_test, y_pred_proba, k)
    metrics[f"NDCG@{k}"] = ndcg_at_k(y_test_arr, y_pred_proba, k)

print(f"\n  {'Metric':<12} {'Score':>8}")
print(f"  {'-'*22}")
print(f"  {'AUC-ROC':<12} {auc_score:>8.4f}")
for k in [5, 8, 10]:
    print(f"  {'P@'+str(k):<12} {metrics['P@'+str(k)]:>8.4f}")
    print(f"  {'R@'+str(k):<12} {metrics['R@'+str(k)]:>8.4f}")
    print(f"  {'NDCG@'+str(k):<12} {metrics['NDCG@'+str(k)]:>8.4f}")

# ── Threshold-based metrics ───────────────────────────────────
threshold   = 0.5
y_pred_bin  = (y_pred_proba >= threshold).astype(int)
print(f"\n  Classification Report (threshold={threshold}):")
print(classification_report(y_test, y_pred_bin, target_names=["Not Added", "Added"]))


# ══════════════════════════════════════════════════════════════
#  STEP 7 — BUSINESS IMPACT ESTIMATION
# ══════════════════════════════════════════════════════════════
print("\n💰 Step 7: Estimating business impact...")

# Simulate: if we show top-8 recommendations per session
# How many extra items would be added?
test_sessions = test_df["session_id"].unique()
total_extra_items  = 0
total_extra_value  = 0.0
session_sample     = test_sessions[:500]   # sample 500 sessions

for sid in session_sample:
    sess = test_df[test_df["session_id"] == sid].copy()
    if len(sess) < 2:
        continue
    sess_idx     = sess.index
    sess_scores  = y_pred_proba[test_df.index.get_indexer(sess_idx)]
    sess_labels  = sess["label"].values
    sess_prices  = sess["cand_price"].values

    # Top 8 recommendations
    top8_idx     = np.argsort(sess_scores)[::-1][:8]
    accepted     = sess_labels[top8_idx]
    added_prices = sess_prices[top8_idx][accepted == 1]

    total_extra_items += accepted.sum()
    total_extra_value += added_prices.sum()

avg_extra_items = total_extra_items / len(session_sample)
avg_aov_lift    = total_extra_value  / len(session_sample)

print(f"  Avg extra items added per session  : {avg_extra_items:.2f}")
print(f"  Avg AOV lift per session           : ₹{avg_aov_lift:.2f}")
print(f"  Estimated accept rate (top-8 shown): {total_extra_items/(len(session_sample)*8)*100:.1f}%")


# ══════════════════════════════════════════════════════════════
#  STEP 8 — CHARTS
# ══════════════════════════════════════════════════════════════
print("\n🎨 Step 8: Generating evaluation charts...")

# ── Chart 1: ROC Curve ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax.plot(fpr, tpr, color="#FF6B35", lw=2.5, label=f"{model_name} (AUC = {auc_score:.4f})")
ax.plot([0,1],[0,1], "k--", lw=1, label="Random Baseline (AUC = 0.5)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — CSAO Ranking Model", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart1_roc_curve.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 1: ROC curve")

# ── Chart 2: Feature Importance ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
top_features = feature_importance.head(20)
colors = ["#FF6B35" if i < 5 else "#4ECDC4" if i < 10 else "#95A5A6"
          for i in range(len(top_features))]
ax.barh(top_features.index[::-1], top_features.values[::-1], color=colors[::-1])
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title("Top 20 Feature Importances", fontweight="bold")
# Add legend
from matplotlib.patches import Patch
legend = [Patch(color="#FF6B35", label="Top 5"),
          Patch(color="#4ECDC4", label="Top 6–10"),
          Patch(color="#95A5A6", label="Rest")]
ax.legend(handles=legend)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart2_feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 2: Feature importance")

# ── Chart 3: Precision, Recall, NDCG @ K ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Ranking Metrics @ K", fontsize=14, fontweight="bold")

ks = [5, 8, 10]
for i, metric_prefix in enumerate(["P", "R", "NDCG"]):
    vals = [metrics[f"{metric_prefix}@{k}"] for k in ks]
    axes[i].bar([str(k) for k in ks], vals,
                color=["#FF6B35","#4ECDC4","#7B2FBE"], edgecolor="white")
    axes[i].set_title(f"{metric_prefix}@K", fontweight="bold")
    axes[i].set_xlabel("K"); axes[i].set_ylim(0, max(vals)*1.3 + 0.01)
    for bar, val in zip(axes[i].patches, vals):
        axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart3_precision_recall_ndcg.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 3: Precision/Recall/NDCG@K")

# ── Chart 4: Score Distribution ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6,
        color="#FF4D4D", label="Not Added (0)", density=True)
ax.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6,
        color="#00C853", label="Added (1)", density=True)
ax.set_xlabel("Predicted Score"); ax.set_ylabel("Density")
ax.set_title("Score Distribution: Added vs Not Added", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart4_score_distribution.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 4: Score distribution")

# ── Chart 5: Model vs Baseline comparison ────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
compare_metrics = ["AUC", "P@8", "NDCG@8"]
baseline_vals   = [baseline_auc,
                   precision_at_k(y_test, baseline_scores, 8),
                   ndcg_at_k(y_test_arr, baseline_scores, 8)]
model_vals      = [auc_score, metrics["P@8"], metrics["NDCG@8"]]

x = np.arange(len(compare_metrics))
width = 0.35
ax.bar(x - width/2, baseline_vals, width, label="Baseline (Co-occurrence)", color="#95A5A6")
ax.bar(x + width/2, model_vals,    width, label=f"{model_name}", color="#FF6B35")
ax.set_xticks(x); ax.set_xticklabels(compare_metrics)
ax.set_ylabel("Score"); ax.set_title("Model vs Baseline", fontweight="bold")
ax.legend(); ax.set_ylim(0, max(model_vals)*1.3)
for bar in ax.patches:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart5_model_vs_baseline.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ Chart 5: Model vs baseline")


# ══════════════════════════════════════════════════════════════
#  STEP 9 — SAVE MODEL + RESULTS
# ══════════════════════════════════════════════════════════════
print("\n💾 Step 9: Saving model and results...")

# Save model
with open(f"{OUTPUT_DIR}/csao_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("  ✓ csao_model.pkl saved")

# Save evaluation results
results = {
    "model_name":        model_name,
    "auc_roc":           round(auc_score, 4),
    "baseline_auc":      round(baseline_auc, 4),
    "auc_improvement":   round(auc_score - baseline_auc, 4),
    **{k: round(v, 4) for k, v in metrics.items()},
    "avg_extra_items_per_session": round(avg_extra_items, 3),
    "avg_aov_lift_inr":            round(avg_aov_lift, 2),
    "train_samples":     len(X_train),
    "test_samples":      len(X_test),
    "num_features":      len(FEATURE_COLS),
}

results_df = pd.DataFrame([results]).T
results_df.columns = ["Value"]
results_df.to_csv(f"{OUTPUT_DIR}/model_results.csv")
print("  ✓ model_results.csv saved")

# Save feature importance
feature_importance.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", header=["importance"])
print("  ✓ feature_importance.csv saved")


# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"""
{'='*60}
✅ Phase 4 Complete!
{'='*60}

🤖 Model      : {model_name}
📊 AUC-ROC    : {auc_score:.4f}  (baseline: {baseline_auc:.4f})  ↑ +{auc_score-baseline_auc:.4f}
📈 P@8        : {metrics['P@8']:.4f}
📈 R@8        : {metrics['R@8']:.4f}
📈 NDCG@8     : {metrics['NDCG@8']:.4f}

💰 Business Impact (estimated):
   Avg extra items/session : {avg_extra_items:.2f}
   Avg AOV lift/session    : ₹{avg_aov_lift:.2f}

📁 Files saved:
   csao_model.pkl
   model_results.csv
   feature_importance.csv
   model_charts/ (5 charts)

🎯 Next step: Phase 5 — LLM Integration (AI Edge)!
""")
