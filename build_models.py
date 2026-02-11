"""
Football Match Prediction - Model Building
============================================
Builds 3 specialized models:
1. Home Win prediction
2. Total Over 2.5 prediction
3. Both Teams To Score (BTTS) prediction

Uses ensemble methods with feature selection optimized per target.
Excludes post-match stats (Goals, Expected Goals) and bookmaker odds.
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, brier_score_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy import stats
from itertools import combinations

# Load processed data
df = pd.read_csv('/home/user/allinone-prediction/processed_data.csv')
print(f"Loaded data: {df.shape}")

# ============================================================
# DEFINE FEATURE SETS (PRE-MATCH ONLY, NO ODDS)
# ============================================================

# Base pre-match features available from both files
base_features = [
    'Team Rating (Home)', 'Team Rating (Away)',
    'Team Form (Home)', 'Team Form (Away)',
    'XG Luckiness (Home)', 'XG Luckiness (Away)',
    'XG Predictability (Home)', 'XG Predictability (Away)',
    'Match Score Prediction (Home)', 'Match Score Prediction (Away)',
    'pred_score_home', 'pred_score_away',
]

# Additional features from file2 (Avg XG)
avg_xg_features = [
    'Avg XG Scored (Home)', 'Avg XG Scored (Away)',
    'Avg XG Conceded (Home)', 'Avg XG Conceded (Away)',
]

# Derived features
derived_features = [
    'rating_diff', 'rating_sum', 'rating_ratio',
    'form_diff', 'form_sum', 'form_ratio',
    'xg_luck_diff', 'xg_luck_sum',
    'xg_pred_diff', 'xg_pred_sum', 'xg_pred_avg',
    'pred_goal_diff', 'pred_total_goals', 'pred_goal_ratio',
    'pred_min_goals', 'pred_max_goals',
    'rating_form_home', 'rating_form_away', 'rating_form_diff',
    'combined_strength_home', 'combined_strength_away', 'combined_strength_diff',
    'pred_cs_total', 'pred_cs_diff', 'pred_cs_min',
    'pred_home_win', 'pred_draw', 'pred_away_win',
    'pred_over', 'pred_bts_yes',
]

# Avg XG derived features
avg_xg_derived = [
    'avg_xg_scored_diff', 'avg_xg_conceded_diff',
    'avg_xg_total_scored', 'avg_xg_total_conceded',
    'home_attack_vs_away_def', 'away_attack_vs_home_def',
]

# All features combined
all_features = base_features + avg_xg_features + derived_features + avg_xg_derived

targets = {
    'home_win': 'actual_home_win',
    'over25': 'actual_total_over25',
    'btts': 'actual_btts',
}

# ============================================================
# FEATURE SELECTION PER TARGET
# ============================================================
print("\n" + "=" * 70)
print("FEATURE SELECTION FOR EACH TARGET")
print("=" * 70)

def select_best_features(X, y, feature_names, target_name, max_features=25):
    """Select best features using multiple methods and ensemble the results."""
    scores = {}

    # Method 1: Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    for i, feat in enumerate(feature_names):
        scores[feat] = scores.get(feat, 0) + mi_scores[i] / mi_scores.max()

    # Method 2: F-statistic
    f_scores, f_pvals = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores)
    if f_scores.max() > 0:
        for i, feat in enumerate(feature_names):
            scores[feat] += f_scores[i] / f_scores.max()

    # Method 3: Point-biserial correlation
    for i, feat in enumerate(feature_names):
        corr, pval = stats.pointbiserialr(y, X[:, i])
        scores[feat] += abs(corr) / 0.5  # normalize roughly

    # Method 4: LightGBM feature importance
    lgbm = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    lgbm.fit(X, y)
    importances = lgbm.feature_importances_
    if importances.max() > 0:
        for i, feat in enumerate(feature_names):
            scores[feat] += importances[i] / importances.max()

    # Sort by combined score
    sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  Top features for {target_name}:")
    for feat, score in sorted_features[:max_features]:
        print(f"    {feat:45s} score={score:.4f}")

    selected = [f for f, s in sorted_features[:max_features]]
    return selected

# Prepare data - use rows that have all base features (no NaN)
# For Avg XG features, we'll handle them separately
feature_selections = {}

for target_name, target_col in targets.items():
    print(f"\n{'='*50}")
    print(f"Feature selection for: {target_name.upper()}")
    print(f"{'='*50}")

    # Use all available data, fill Avg XG NaNs with median
    df_work = df[all_features + [target_col]].copy()

    # Drop rows where target is NaN
    df_work = df_work.dropna(subset=[target_col])

    # Fill NaN in avg_xg features with median
    for col in avg_xg_features + avg_xg_derived:
        if col in df_work.columns:
            median_val = df_work[col].median()
            df_work[col] = df_work[col].fillna(median_val)

    # Drop any remaining NaN rows
    df_work = df_work.dropna()

    X = df_work[all_features].values
    y = df_work[target_col].values.astype(int)

    selected = select_best_features(X, y, all_features, target_name)
    feature_selections[target_name] = selected

# ============================================================
# MODEL BUILDING WITH CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("BUILDING MODELS WITH CROSS-VALIDATION")
print("=" * 70)

def build_stacking_model(X_train, y_train, target_name):
    """Build a calibrated stacking ensemble model."""

    # Base estimators
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                 random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                         min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                         random_state=42, eval_metric='logloss', verbosity=0)
    lgbm = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                           min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbose=-1)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     min_samples_leaf=15, subsample=0.8, random_state=42)

    # Stacking ensemble
    stacking = StackingClassifier(
        estimators=[
            ('lr', lr),
            ('rf', rf),
            ('xgb', xgb),
            ('lgbm', lgbm),
            ('gb', gb),
        ],
        final_estimator=LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )

    # Calibrate probabilities
    calibrated = CalibratedClassifierCV(stacking, cv=3, method='isotonic')

    return calibrated


def evaluate_model(X, y, features, target_name, model_builder):
    """Evaluate model using nested cross-validation."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Build and train model
        model = model_builder(X_train_s, y_train, target_name)
        model.fit(X_train_s, y_train)

        # Predict
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        fold_acc = accuracy_score(y_test, y_pred)
        fold_auc = roc_auc_score(y_test, y_prob)
        fold_metrics.append({'fold': fold+1, 'accuracy': fold_acc, 'auc': fold_auc})
        print(f"    Fold {fold+1}: accuracy={fold_acc:.4f}, AUC={fold_auc:.4f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    print(f"\n  OVERALL RESULTS ({target_name}):")
    print(f"    Accuracy:  {accuracy_score(all_y_true, all_y_pred):.4f}")
    print(f"    Precision: {precision_score(all_y_true, all_y_pred):.4f}")
    print(f"    Recall:    {recall_score(all_y_true, all_y_pred):.4f}")
    print(f"    F1:        {f1_score(all_y_true, all_y_pred):.4f}")
    print(f"    AUC-ROC:   {roc_auc_score(all_y_true, all_y_prob):.4f}")
    print(f"    Brier:     {brier_score_loss(all_y_true, all_y_prob):.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=['No', 'Yes']))

    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")

    return {
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'precision': precision_score(all_y_true, all_y_pred),
        'recall': recall_score(all_y_true, all_y_pred),
        'f1': f1_score(all_y_true, all_y_pred),
        'auc': roc_auc_score(all_y_true, all_y_prob),
        'brier': brier_score_loss(all_y_true, all_y_prob),
    }


# ============================================================
# TRAIN AND EVALUATE EACH TARGET
# ============================================================
model_results = {}
final_models = {}
final_scalers = {}
final_features = {}

for target_name, target_col in targets.items():
    print(f"\n{'='*60}")
    print(f"  MODEL: {target_name.upper()}")
    print(f"{'='*60}")

    features = feature_selections[target_name]

    # Prepare data
    df_work = df[features + [target_col]].copy()
    df_work = df_work.dropna(subset=[target_col])

    for col in avg_xg_features + avg_xg_derived:
        if col in df_work.columns:
            median_val = df_work[col].median()
            df_work[col] = df_work[col].fillna(median_val)

    df_work = df_work.dropna()

    X = df_work[features].values
    y = df_work[target_col].values.astype(int)

    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive rate: {y.mean():.3f}")

    # Evaluate with cross-validation
    metrics = evaluate_model(X, y, features, target_name, build_stacking_model)
    model_results[target_name] = metrics

    # Train final model on all data
    print(f"\n  Training final model on all data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    final_model = build_stacking_model(X_scaled, y, target_name)
    final_model.fit(X_scaled, y)

    final_models[target_name] = final_model
    final_scalers[target_name] = scaler
    final_features[target_name] = features

    print(f"  Final model trained successfully.")

# ============================================================
# ADDITIONAL ANALYSIS: CONFIDENCE THRESHOLDS
# ============================================================
print("\n" + "=" * 70)
print("CONFIDENCE THRESHOLD ANALYSIS")
print("=" * 70)

threshold_results = {}

for target_name, target_col in targets.items():
    print(f"\n{'='*50}")
    print(f"  {target_name.upper()} - Probability Threshold Analysis")
    print(f"{'='*50}")

    features = final_features[target_name]
    df_work = df[features + [target_col]].copy()
    df_work = df_work.dropna(subset=[target_col])

    for col in avg_xg_features + avg_xg_derived:
        if col in df_work.columns:
            df_work[col] = df_work[col].fillna(df_work[col].median())
    df_work = df_work.dropna()

    X = df_work[features].values
    y = df_work[target_col].values.astype(int)

    # Get cross-validated probabilities
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        model = build_stacking_model(X_train, y[train_idx], target_name)
        model.fit(X_train, y[train_idx])
        probs[test_idx] = model.predict_proba(X_test)[:, 1]

    # Analyze different thresholds
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    best_threshold = 0.5
    best_profit_proxy = 0

    print(f"\n  {'Threshold':>10} {'Matches':>8} {'Accuracy':>10} {'Precision':>10} {'Hit Rate':>10} {'% of Total':>10}")
    print(f"  {'-'*60}")

    target_thresholds = {}
    for thr in thresholds:
        mask = probs >= thr
        n_selected = mask.sum()
        if n_selected < 20:
            continue

        acc = accuracy_score(y[mask], (probs[mask] >= 0.5).astype(int))
        prec = y[mask].mean()  # actual hit rate among selected
        pct = n_selected / len(y) * 100

        print(f"  {thr:>10.2f} {n_selected:>8d} {acc:>10.3f} {prec:>10.3f} {prec:>10.3f} {pct:>9.1f}%")

        target_thresholds[thr] = {'matches': int(n_selected), 'hit_rate': float(prec), 'pct': float(pct)}

        # Best threshold: maximize hit_rate * volume
        score = prec * (n_selected / len(y))
        if score > best_profit_proxy and prec > y.mean():
            best_profit_proxy = score
            best_threshold = thr

    threshold_results[target_name] = {
        'thresholds': target_thresholds,
        'recommended': best_threshold
    }
    print(f"\n  Recommended confidence threshold: {best_threshold:.2f}")

# ============================================================
# SAVE MODELS AND CONFIGURATION
# ============================================================
print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

# Save models
for target_name in targets:
    model_path = f'/home/user/allinone-prediction/model_{target_name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': final_models[target_name],
            'scaler': final_scalers[target_name],
            'features': final_features[target_name],
        }, f)
    print(f"  Saved: {model_path}")

# Save model configuration
config = {
    'targets': dict(targets),
    'feature_selections': feature_selections,
    'model_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in model_results.items()},
    'threshold_results': threshold_results,
    'avg_xg_features': avg_xg_features,
    'avg_xg_derived': avg_xg_derived,
}

with open('/home/user/allinone-prediction/model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n  Model configuration saved.")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 70)

for target_name, metrics in model_results.items():
    thr = threshold_results[target_name]['recommended']
    print(f"\n  {target_name.upper()}:")
    print(f"    AUC-ROC:   {metrics['auc']:.4f}")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    F1-Score:  {metrics['f1']:.4f}")
    print(f"    Brier:     {metrics['brier']:.4f}")
    print(f"    Recommended threshold: {thr:.2f}")

print("\nDone! Models saved successfully.")
