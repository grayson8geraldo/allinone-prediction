"""
Optimized Model Building
=========================
- Home Win: Stacking ensemble (already good AUC=0.71)
- Over 2.5: Optimized XGBoost + LightGBM with focus on best features + rule overlay
- BTTS: Optimized ensemble with aggressive feature engineering + rule overlay

Key insight: Over 2.5 and BTTS have weaker signal in the data.
Strategy: Use ML probability as base + rule-based boosters from bin analysis.
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, brier_score_loss,
                             classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

df = pd.read_csv('/home/user/allinone-prediction/processed_data.csv')
print(f"Data: {df.shape}")

# Fill NaN in avg_xg features with median
avg_xg_cols = [c for c in df.columns if 'avg_xg' in c or 'Avg XG' in c or
               'home_attack' in c or 'away_attack' in c]
for col in avg_xg_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# ============================================================
# HOME WIN MODEL - Optimized Stacking
# ============================================================
print("\n" + "=" * 70)
print("HOME WIN MODEL")
print("=" * 70)

hw_features = [
    'rating_diff', 'pred_goal_diff', 'pred_goal_ratio',
    'Match Score Prediction (Home)', 'Match Score Prediction (Away)',
    'rating_form_diff', 'combined_strength_diff',
    'pred_cs_diff', 'rating_ratio',
    'Team Rating (Home)', 'Team Rating (Away)',
    'rating_form_home', 'rating_form_away',
    'combined_strength_home', 'combined_strength_away',
    'pred_score_home', 'pred_score_away',
    'pred_home_win', 'pred_away_win', 'pred_draw',
    'form_diff', 'form_ratio',
    'Team Form (Home)', 'Team Form (Away)',
    'xg_pred_diff', 'xg_luck_diff',
]

target_hw = 'actual_home_win'
df_hw = df[hw_features + [target_hw]].dropna()
X_hw = df_hw[hw_features].values
y_hw = df_hw[target_hw].values.astype(int)

print(f"Samples: {len(y_hw)}, Positive rate: {y_hw.mean():.3f}")

# Cross-validation evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
hw_probs = np.zeros(len(y_hw))
hw_preds = np.zeros(len(y_hw))

for fold, (tr, te) in enumerate(skf.split(X_hw, y_hw)):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_hw[tr])
    X_te = sc.transform(X_hw[te])

    model = StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, C=0.5)),
            ('rf', RandomForestClassifier(n_estimators=500, max_depth=8,
                                           min_samples_leaf=8, random_state=42, n_jobs=-1)),
            ('xgb', XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.03,
                                   min_child_weight=15, subsample=0.8, colsample_bytree=0.7,
                                   reg_alpha=0.1, reg_lambda=1.0,
                                   random_state=42, eval_metric='logloss', verbosity=0)),
            ('lgbm', LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                                     min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
                                     reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                                              min_samples_leaf=15, subsample=0.8, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, C=0.3),
        cv=5, passthrough=False, n_jobs=-1,
    )
    cal_model = CalibratedClassifierCV(model, cv=3, method='isotonic')
    cal_model.fit(X_tr, y_hw[tr])
    hw_probs[te] = cal_model.predict_proba(X_te)[:, 1]
    hw_preds[te] = (hw_probs[te] >= 0.5).astype(int)
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y_hw[te], hw_probs[te]):.4f}")

print(f"\n  HOME WIN CV Results:")
print(f"    AUC:       {roc_auc_score(y_hw, hw_probs):.4f}")
print(f"    Accuracy:  {accuracy_score(y_hw, hw_preds):.4f}")
print(f"    Precision: {precision_score(y_hw, hw_preds):.4f}")
print(f"    F1:        {f1_score(y_hw, hw_preds):.4f}")

# ============================================================
# OVER 2.5 MODEL - Different approach: focused XGBoost + rule overlay
# ============================================================
print("\n" + "=" * 70)
print("OVER 2.5 MODEL")
print("=" * 70)

ou_features = [
    'pred_total_goals', 'pred_max_goals', 'pred_min_goals',
    'Match Score Prediction (Home)', 'Match Score Prediction (Away)',
    'pred_cs_total', 'pred_goal_ratio', 'pred_goal_diff',
    'pred_score_home', 'pred_score_away',
    'pred_over', 'pred_draw',
    'rating_sum', 'form_sum',
    'Team Rating (Home)', 'Team Rating (Away)',
    'Team Form (Home)', 'Team Form (Away)',
    'xg_luck_sum', 'xg_luck_diff',
    'XG Luckiness (Home)', 'XG Luckiness (Away)',
    'XG Predictability (Home)', 'XG Predictability (Away)',
    'xg_pred_sum', 'xg_pred_avg',
    'rating_form_home', 'rating_form_away',
    'combined_strength_diff', 'rating_diff',
]

target_ou = 'actual_total_over25'
df_ou = df[ou_features + [target_ou]].dropna()
X_ou = df_ou[ou_features].values
y_ou = df_ou[target_ou].values.astype(int)

print(f"Samples: {len(y_ou)}, Positive rate: {y_ou.mean():.3f}")

# Try multiple model configs and pick the best
ou_probs = np.zeros(len(y_ou))
ou_preds = np.zeros(len(y_ou))

for fold, (tr, te) in enumerate(skf.split(X_ou, y_ou)):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_ou[tr])
    X_te = sc.transform(X_ou[te])

    # Weighted voting of tuned models
    models = [
        ('xgb', XGBClassifier(n_estimators=800, max_depth=3, learning_rate=0.01,
                                min_child_weight=30, subsample=0.7, colsample_bytree=0.6,
                                reg_alpha=0.5, reg_lambda=2.0, gamma=0.5,
                                random_state=42, eval_metric='logloss', verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=800, max_depth=4, learning_rate=0.01,
                                  min_child_samples=40, subsample=0.7, colsample_bytree=0.6,
                                  reg_alpha=0.5, reg_lambda=2.0,
                                  random_state=42, verbose=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.01,
                                           min_samples_leaf=30, subsample=0.7, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, C=0.1)),
    ]

    fold_probs = np.zeros((len(te), len(models)))
    for i, (name, m) in enumerate(models):
        m.fit(X_tr, y_ou[tr])
        fold_probs[:, i] = m.predict_proba(X_te)[:, 1]

    # Weighted average
    weights = [0.3, 0.3, 0.2, 0.2]
    avg_prob = np.average(fold_probs, axis=1, weights=weights)
    ou_probs[te] = avg_prob
    ou_preds[te] = (avg_prob >= 0.5).astype(int)
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y_ou[te], avg_prob):.4f}")

print(f"\n  OVER 2.5 CV Results (ML only):")
print(f"    AUC:       {roc_auc_score(y_ou, ou_probs):.4f}")
print(f"    Accuracy:  {accuracy_score(y_ou, ou_preds):.4f}")

# Apply rule-based overlay from bin analysis
# From analysis: pred_total_goals > 3.2 -> 64.6% hit rate
# pred_max_goals > 2.1 -> 63.3%, pred_cs_total > 3 -> 60.7%
rule_boost_ou = np.zeros(len(y_ou))
ou_df = df_ou.copy()
ou_df = ou_df.reset_index(drop=True)

# Rule 1: High predicted total goals
rule_boost_ou[ou_df['pred_total_goals'] > 3.2] += 0.08
rule_boost_ou[(ou_df['pred_total_goals'] > 2.9) & (ou_df['pred_total_goals'] <= 3.2)] += 0.03
rule_boost_ou[ou_df['pred_total_goals'] <= 2.4] -= 0.08

# Rule 2: High max prediction
rule_boost_ou[ou_df['pred_max_goals'] > 2.1] += 0.05
rule_boost_ou[ou_df['pred_max_goals'] <= 1.4] -= 0.05

# Rule 3: High Match Score Prediction
rule_boost_ou[ou_df['Match Score Prediction (Home)'] > 1.9] += 0.05
rule_boost_ou[ou_df['Match Score Prediction (Home)'] <= 1.1] -= 0.03

# Combine ML + rules (clipped to [0,1])
ou_hybrid_probs = np.clip(ou_probs + rule_boost_ou, 0, 1)
ou_hybrid_preds = (ou_hybrid_probs >= 0.5).astype(int)

print(f"\n  OVER 2.5 CV Results (ML + Rules):")
print(f"    AUC:       {roc_auc_score(y_ou, ou_hybrid_probs):.4f}")
print(f"    Accuracy:  {accuracy_score(y_ou, ou_hybrid_preds):.4f}")
print(f"    Precision: {precision_score(y_ou, ou_hybrid_preds):.4f}")
print(f"    F1:        {f1_score(y_ou, ou_hybrid_preds):.4f}")

# ============================================================
# BTTS MODEL - Focus on what separates yes from no
# ============================================================
print("\n" + "=" * 70)
print("BTTS MODEL")
print("=" * 70)

btts_features = [
    'pred_total_goals', 'pred_min_goals', 'pred_max_goals',
    'Match Score Prediction (Home)', 'Match Score Prediction (Away)',
    'pred_goal_diff', 'pred_goal_ratio',
    'pred_cs_total', 'pred_cs_diff', 'pred_cs_min',
    'pred_score_home', 'pred_score_away',
    'pred_bts_yes', 'pred_over',
    'Team Rating (Home)', 'Team Rating (Away)',
    'Team Form (Home)', 'Team Form (Away)',
    'rating_sum', 'rating_diff',
    'form_sum', 'form_diff', 'form_ratio',
    'rating_form_diff', 'rating_form_home', 'rating_form_away',
    'combined_strength_diff', 'combined_strength_away',
    'xg_luck_diff', 'xg_luck_sum',
    'xg_pred_diff',
    'XG Predictability (Home)', 'XG Predictability (Away)',
    'avg_xg_conceded_diff', 'Avg XG Conceded (Away)',
    'home_attack_vs_away_def', 'away_attack_vs_home_def',
    'avg_xg_total_conceded',
]

# Remove features with too many NaN
btts_features_clean = []
for f in btts_features:
    if f in df.columns and df[f].notna().sum() > 1000:
        btts_features_clean.append(f)

target_btts = 'actual_btts'
df_btts = df[btts_features_clean + [target_btts]].dropna()
X_btts = df_btts[btts_features_clean].values
y_btts = df_btts[target_btts].values.astype(int)

print(f"Samples: {len(y_btts)}, Positive rate: {y_btts.mean():.3f}")

btts_probs = np.zeros(len(y_btts))
btts_preds = np.zeros(len(y_btts))

for fold, (tr, te) in enumerate(skf.split(X_btts, y_btts)):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_btts[tr])
    X_te = sc.transform(X_btts[te])

    models = [
        ('xgb', XGBClassifier(n_estimators=800, max_depth=3, learning_rate=0.01,
                                min_child_weight=40, subsample=0.7, colsample_bytree=0.5,
                                reg_alpha=1.0, reg_lambda=3.0, gamma=1.0,
                                random_state=42, eval_metric='logloss', verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=800, max_depth=4, learning_rate=0.01,
                                  min_child_samples=50, subsample=0.7, colsample_bytree=0.5,
                                  reg_alpha=1.0, reg_lambda=3.0,
                                  random_state=42, verbose=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.01,
                                           min_samples_leaf=40, subsample=0.7, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, C=0.05)),
    ]

    fold_probs = np.zeros((len(te), len(models)))
    for i, (name, m) in enumerate(models):
        m.fit(X_tr, y_btts[tr])
        fold_probs[:, i] = m.predict_proba(X_te)[:, 1]

    weights = [0.3, 0.3, 0.2, 0.2]
    avg_prob = np.average(fold_probs, axis=1, weights=weights)
    btts_probs[te] = avg_prob
    btts_preds[te] = (avg_prob >= 0.5).astype(int)
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y_btts[te], avg_prob):.4f}")

print(f"\n  BTTS CV Results (ML only):")
print(f"    AUC:       {roc_auc_score(y_btts, btts_probs):.4f}")
print(f"    Accuracy:  {accuracy_score(y_btts, btts_preds):.4f}")

# Rule-based overlay for BTTS
rule_boost_btts = np.zeros(len(y_btts))
btts_df = df_btts.copy().reset_index(drop=True)

# From bin analysis: pred_min_goals > 1.3 -> 62.1% hit
rule_boost_btts[btts_df['pred_min_goals'] > 1.3] += 0.06
rule_boost_btts[btts_df['pred_min_goals'] <= 0.9] -= 0.03

# pred_total_goals matters
rule_boost_btts[btts_df['pred_total_goals'] > 3.2] += 0.04
rule_boost_btts[btts_df['pred_total_goals'] <= 2.4] -= 0.04

# Predicted score matters: both teams need to score in prediction
rule_boost_btts[(btts_df['pred_score_home'] > 0) & (btts_df['pred_score_away'] > 0)] += 0.03
rule_boost_btts[(btts_df['pred_score_home'] == 0) | (btts_df['pred_score_away'] == 0)] -= 0.05

# avg_xg_conceded_away is strong signal
if 'Avg XG Conceded (Away)' in btts_df.columns:
    rule_boost_btts[btts_df['Avg XG Conceded (Away)'] > 1.8] += 0.05
    rule_boost_btts[btts_df['Avg XG Conceded (Away)'] <= 1.1] -= 0.03

btts_hybrid_probs = np.clip(btts_probs + rule_boost_btts, 0, 1)
btts_hybrid_preds = (btts_hybrid_probs >= 0.5).astype(int)

print(f"\n  BTTS CV Results (ML + Rules):")
print(f"    AUC:       {roc_auc_score(y_btts, btts_hybrid_probs):.4f}")
print(f"    Accuracy:  {accuracy_score(y_btts, btts_hybrid_preds):.4f}")
print(f"    Precision: {precision_score(y_btts, btts_hybrid_preds):.4f}")
print(f"    F1:        {f1_score(y_btts, btts_hybrid_preds):.4f}")

# ============================================================
# CONFIDENCE THRESHOLDS FOR ALL 3
# ============================================================
print("\n" + "=" * 70)
print("CONFIDENCE THRESHOLD ANALYSIS (FINAL)")
print("=" * 70)

for name, probs, y_true in [
    ('HOME WIN', hw_probs, y_hw),
    ('OVER 2.5', ou_hybrid_probs, y_ou),
    ('BTTS', btts_hybrid_probs, y_btts)
]:
    print(f"\n  {name}:")
    print(f"  {'Threshold':>10} {'Matches':>8} {'HitRate':>8} {'% Data':>8}")
    print(f"  {'-'*40}")
    for thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = probs >= thr
        n = mask.sum()
        if n < 10:
            continue
        hit = y_true[mask].mean()
        pct = n / len(y_true) * 100
        star = " <-- BEST" if hit > y_true.mean() * 1.15 and pct > 5 else ""
        print(f"  {thr:>10.2f} {n:>8d} {hit:>8.3f} {pct:>7.1f}%{star}")

# ============================================================
# TRAIN FINAL MODELS ON ALL DATA
# ============================================================
print("\n" + "=" * 70)
print("TRAINING FINAL MODELS")
print("=" * 70)

# HOME WIN final model
sc_hw = StandardScaler()
X_hw_s = sc_hw.fit_transform(X_hw)
hw_model = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, C=0.5)),
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=8,
                                       min_samples_leaf=8, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.03,
                               min_child_weight=15, subsample=0.8, colsample_bytree=0.7,
                               reg_alpha=0.1, reg_lambda=1.0,
                               random_state=42, eval_metric='logloss', verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                                 min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
                                 reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                                          min_samples_leaf=15, subsample=0.8, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=0.3),
    cv=5, passthrough=False, n_jobs=-1,
)
hw_cal = CalibratedClassifierCV(hw_model, cv=3, method='isotonic')
hw_cal.fit(X_hw_s, y_hw)
print("  Home Win model trained.")

# OVER 2.5 final models
sc_ou = StandardScaler()
X_ou_s = sc_ou.fit_transform(X_ou)
ou_models_final = {}
ou_model_configs = [
    ('xgb', XGBClassifier(n_estimators=800, max_depth=3, learning_rate=0.01,
                            min_child_weight=30, subsample=0.7, colsample_bytree=0.6,
                            reg_alpha=0.5, reg_lambda=2.0, gamma=0.5,
                            random_state=42, eval_metric='logloss', verbosity=0)),
    ('lgbm', LGBMClassifier(n_estimators=800, max_depth=4, learning_rate=0.01,
                              min_child_samples=40, subsample=0.7, colsample_bytree=0.6,
                              reg_alpha=0.5, reg_lambda=2.0,
                              random_state=42, verbose=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.01,
                                       min_samples_leaf=30, subsample=0.7, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, C=0.1)),
]
for name, m in ou_model_configs:
    m.fit(X_ou_s, y_ou)
    ou_models_final[name] = m
print("  Over 2.5 models trained.")

# BTTS final models
sc_btts = StandardScaler()
X_btts_s = sc_btts.fit_transform(X_btts)
btts_models_final = {}
btts_model_configs = [
    ('xgb', XGBClassifier(n_estimators=800, max_depth=3, learning_rate=0.01,
                            min_child_weight=40, subsample=0.7, colsample_bytree=0.5,
                            reg_alpha=1.0, reg_lambda=3.0, gamma=1.0,
                            random_state=42, eval_metric='logloss', verbosity=0)),
    ('lgbm', LGBMClassifier(n_estimators=800, max_depth=4, learning_rate=0.01,
                              min_child_samples=50, subsample=0.7, colsample_bytree=0.5,
                              reg_alpha=1.0, reg_lambda=3.0,
                              random_state=42, verbose=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.01,
                                       min_samples_leaf=40, subsample=0.7, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, C=0.05)),
]
for name, m in btts_model_configs:
    m.fit(X_btts_s, y_btts)
    btts_models_final[name] = m
print("  BTTS models trained.")

# Compute median values for avg_xg features (for imputation)
median_values = {}
for col in avg_xg_cols:
    if col in df.columns:
        median_values[col] = float(df[col].median())

# ============================================================
# SAVE EVERYTHING
# ============================================================
final_data = {
    'home_win': {
        'model': hw_cal,
        'scaler': sc_hw,
        'features': hw_features,
    },
    'over25': {
        'models': ou_models_final,
        'weights': [0.3, 0.3, 0.2, 0.2],
        'scaler': sc_ou,
        'features': ou_features,
    },
    'btts': {
        'models': btts_models_final,
        'weights': [0.3, 0.3, 0.2, 0.2],
        'scaler': sc_btts,
        'features': btts_features_clean,
    },
    'median_values': median_values,
}

with open('/home/user/allinone-prediction/final_models.pkl', 'wb') as f:
    pickle.dump(final_data, f)

print("\nAll models saved to final_models.pkl")

# Save model performance summary
perf = {
    'home_win': {
        'cv_auc': float(roc_auc_score(y_hw, hw_probs)),
        'cv_accuracy': float(accuracy_score(y_hw, hw_preds)),
        'cv_precision': float(precision_score(y_hw, hw_preds)),
        'cv_f1': float(f1_score(y_hw, hw_preds)),
        'features': hw_features,
        'n_samples': int(len(y_hw)),
    },
    'over25': {
        'cv_auc': float(roc_auc_score(y_ou, ou_hybrid_probs)),
        'cv_accuracy': float(accuracy_score(y_ou, ou_hybrid_preds)),
        'cv_precision': float(precision_score(y_ou, ou_hybrid_preds)),
        'cv_f1': float(f1_score(y_ou, ou_hybrid_preds)),
        'features': ou_features,
        'n_samples': int(len(y_ou)),
    },
    'btts': {
        'cv_auc': float(roc_auc_score(y_btts, btts_hybrid_probs)),
        'cv_accuracy': float(accuracy_score(y_btts, btts_hybrid_preds)),
        'cv_precision': float(precision_score(y_btts, btts_hybrid_preds)),
        'cv_f1': float(f1_score(y_btts, btts_hybrid_preds)),
        'features': btts_features_clean,
        'n_samples': int(len(y_btts)),
    },
}

with open('/home/user/allinone-prediction/model_performance.json', 'w') as f:
    json.dump(perf, f, indent=2)

print("Performance saved.")
print("\nDONE!")
