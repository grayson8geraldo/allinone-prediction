"""
Football Match Data Analysis & Pattern Discovery
==================================================
Step 1: Load, merge, deduplicate data
Step 2: Feature engineering
Step 3: Exploratory data analysis & pattern discovery
Step 4: Statistical testing of all feature combinations
"""

import pandas as pd
import numpy as np
import re
import warnings
import json
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: Load and merge data
# ============================================================
print("=" * 70)
print("STEP 1: LOADING AND MERGING DATA")
print("=" * 70)

df1 = pd.read_excel('/home/user/allinone-prediction/All 1ligi (1).xlsx')
df2 = pd.read_excel('/home/user/allinone-prediction/football_statistics.xlsx')

print(f"File 1 shape: {df1.shape}")
print(f"File 2 shape: {df2.shape}")

# Add League column to df1 if missing (set as Unknown)
if 'League' not in df1.columns:
    df1['League'] = 'Unknown'

# Standardize XG Luckiness columns in df1 (they have % signs)
def parse_percentage(val):
    """Convert percentage string or float to numeric."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip().replace('%', '')
    try:
        return float(val)
    except:
        return np.nan

for col in ['XG Luckiness (Home)', 'XG Luckiness (Away)',
            'XG Predictability (Home)', 'XG Predictability (Away)']:
    df1[col] = df1[col].apply(parse_percentage)
    df2[col] = df2[col].apply(parse_percentage)

# Add missing Avg XG columns to df1
for col in ['Avg XG Scored (Home)', 'Avg XG Scored (Away)',
            'Avg XG Conceded (Home)', 'Avg XG Conceded (Away)']:
    if col not in df1.columns:
        df1[col] = np.nan

# Ensure both dataframes have the same columns in the same order
common_cols = ['League', 'Team (Home)', 'Team (Away)', 'Home / Away', 'Over / Under',
               'Both To Score', 'Correct Score', 'Team Rating (Home)', 'Team Rating (Away)',
               'Team Form (Home)', 'Team Form (Away)', 'XG Luckiness (Home)', 'XG Luckiness (Away)',
               'XG Predictability (Home)', 'XG Predictability (Away)',
               'Avg XG Scored (Home)', 'Avg XG Scored (Away)',
               'Avg XG Conceded (Home)', 'Avg XG Conceded (Away)',
               'Match Score Prediction (Home)', 'Match Score Prediction (Away)',
               'Goals (Home)', 'Goals (Away)', 'Expected Goals (Home)', 'Expected Goals (Away)']

df1 = df1[common_cols]
df2 = df2[common_cols]

# Merge
df_all = pd.concat([df1, df2], ignore_index=True)
print(f"\nMerged shape (before dedup): {df_all.shape}")

# Remove exact duplicates
before = len(df_all)
df_all = df_all.drop_duplicates()
print(f"Removed {before - len(df_all)} exact duplicates")

# Remove near-duplicates (same teams, same goals, same ratings)
dedup_cols = ['Team (Home)', 'Team (Away)', 'Team Rating (Home)', 'Team Rating (Away)',
              'Team Form (Home)', 'Team Form (Away)', 'Goals (Home)', 'Goals (Away)',
              'Match Score Prediction (Home)', 'Match Score Prediction (Away)']
before = len(df_all)
df_all = df_all.drop_duplicates(subset=dedup_cols, keep='last')  # keep last (from df2 with more features)
print(f"Removed {before - len(df_all)} near-duplicates")
print(f"Final dataset shape: {df_all.shape}")

# ============================================================
# STEP 2: Feature Engineering
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 70)

# Parse Home / Away prediction
def parse_home_away(val):
    """Extract prediction type and odds from 'Home Win (1.54)' format."""
    if pd.isna(val):
        return None, None
    match = re.match(r'(Home Win|Away Win|Draw)\s*\(([0-9.]+)\)', str(val))
    if match:
        return match.group(1), float(match.group(2))
    return None, None

# Parse Over / Under prediction
def parse_over_under(val):
    """Extract over/under prediction and odds."""
    if pd.isna(val):
        return None, None
    match = re.match(r'Total (Over|Under) 2\.5\s*\(([0-9.]+)\)', str(val))
    if match:
        return match.group(1), float(match.group(2))
    return None, None

# Parse Both To Score
def parse_bts(val):
    """Extract BTS prediction and odds."""
    if pd.isna(val):
        return None, None
    match = re.match(r'Both To Score:\s*(Yes|No)\s*\(([0-9.]+)\)', str(val))
    if match:
        return match.group(1), float(match.group(2))
    return None, None

# Parse Correct Score
def parse_correct_score(val):
    """Extract predicted score and odds."""
    if pd.isna(val):
        return None, None, None
    match = re.match(r'Correct Score:\s*(\d+)-(\d+)\s*\(([0-9.]+)\)', str(val))
    if match:
        return int(match.group(1)), int(match.group(2)), float(match.group(3))
    return None, None, None

# Apply parsing
df_all[['pred_result', 'pred_result_odds']] = df_all['Home / Away'].apply(
    lambda x: pd.Series(parse_home_away(x)))
df_all[['pred_ou', 'pred_ou_odds']] = df_all['Over / Under'].apply(
    lambda x: pd.Series(parse_over_under(x)))
df_all[['pred_bts', 'pred_bts_odds']] = df_all['Both To Score'].apply(
    lambda x: pd.Series(parse_bts(x)))
df_all[['pred_score_home', 'pred_score_away', 'pred_score_odds']] = df_all['Correct Score'].apply(
    lambda x: pd.Series(parse_correct_score(x)))

# Create target variables (ACTUAL RESULTS from goals)
df_all['actual_home_win'] = (df_all['Goals (Home)'] > df_all['Goals (Away)']).astype(int)
df_all['actual_draw'] = (df_all['Goals (Home)'] == df_all['Goals (Away)']).astype(int)
df_all['actual_away_win'] = (df_all['Goals (Home)'] < df_all['Goals (Away)']).astype(int)
df_all['actual_total_over25'] = ((df_all['Goals (Home)'] + df_all['Goals (Away)']) > 2.5).astype(int)
df_all['actual_btts'] = ((df_all['Goals (Home)'] > 0) & (df_all['Goals (Away)'] > 0)).astype(int)

# Create numerical features
# Prediction is Home Win
df_all['pred_home_win'] = (df_all['pred_result'] == 'Home Win').astype(int)
df_all['pred_draw'] = (df_all['pred_result'] == 'Draw').astype(int)
df_all['pred_away_win'] = (df_all['pred_result'] == 'Away Win').astype(int)
df_all['pred_over'] = (df_all['pred_ou'] == 'Over').astype(int)
df_all['pred_bts_yes'] = (df_all['pred_bts'] == 'Yes').astype(int)

# Rating difference
df_all['rating_diff'] = df_all['Team Rating (Home)'] - df_all['Team Rating (Away)']
df_all['rating_sum'] = df_all['Team Rating (Home)'] + df_all['Team Rating (Away)']
df_all['rating_ratio'] = df_all['Team Rating (Home)'] / df_all['Team Rating (Away)'].replace(0, 0.1)

# Form difference
df_all['form_diff'] = df_all['Team Form (Home)'] - df_all['Team Form (Away)']
df_all['form_sum'] = df_all['Team Form (Home)'] + df_all['Team Form (Away)']
df_all['form_ratio'] = df_all['Team Form (Home)'] / df_all['Team Form (Away)'].replace(0, 0.1)

# XG-based features
df_all['xg_luck_diff'] = df_all['XG Luckiness (Home)'] - df_all['XG Luckiness (Away)']
df_all['xg_luck_sum'] = df_all['XG Luckiness (Home)'] + df_all['XG Luckiness (Away)']
df_all['xg_pred_diff'] = df_all['XG Predictability (Home)'] - df_all['XG Predictability (Away)']
df_all['xg_pred_sum'] = df_all['XG Predictability (Home)'] + df_all['XG Predictability (Away)']
df_all['xg_pred_avg'] = (df_all['XG Predictability (Home)'] + df_all['XG Predictability (Away)']) / 2

# Match Score Prediction features
df_all['pred_goal_diff'] = df_all['Match Score Prediction (Home)'] - df_all['Match Score Prediction (Away)']
df_all['pred_total_goals'] = df_all['Match Score Prediction (Home)'] + df_all['Match Score Prediction (Away)']
df_all['pred_goal_ratio'] = df_all['Match Score Prediction (Home)'] / df_all['Match Score Prediction (Away)'].replace(0, 0.1)
df_all['pred_min_goals'] = df_all[['Match Score Prediction (Home)', 'Match Score Prediction (Away)']].min(axis=1)
df_all['pred_max_goals'] = df_all[['Match Score Prediction (Home)', 'Match Score Prediction (Away)']].max(axis=1)

# Avg XG features (from file2, might have NaN for file1 rows)
df_all['avg_xg_scored_diff'] = df_all['Avg XG Scored (Home)'] - df_all['Avg XG Scored (Away)']
df_all['avg_xg_conceded_diff'] = df_all['Avg XG Conceded (Home)'] - df_all['Avg XG Conceded (Away)']
df_all['avg_xg_total_scored'] = df_all['Avg XG Scored (Home)'] + df_all['Avg XG Scored (Away)']
df_all['avg_xg_total_conceded'] = df_all['Avg XG Conceded (Home)'] + df_all['Avg XG Conceded (Away)']
df_all['home_attack_vs_away_def'] = df_all['Avg XG Scored (Home)'] - df_all['Avg XG Conceded (Away)']
df_all['away_attack_vs_home_def'] = df_all['Avg XG Scored (Away)'] - df_all['Avg XG Conceded (Home)']

# Combined features
df_all['rating_form_home'] = df_all['Team Rating (Home)'] * df_all['Team Form (Home)']
df_all['rating_form_away'] = df_all['Team Rating (Away)'] * df_all['Team Form (Away)']
df_all['rating_form_diff'] = df_all['rating_form_home'] - df_all['rating_form_away']
df_all['combined_strength_home'] = (df_all['Team Rating (Home)'] + df_all['Team Form (Home)']) / 2
df_all['combined_strength_away'] = (df_all['Team Rating (Away)'] + df_all['Team Form (Away)']) / 2
df_all['combined_strength_diff'] = df_all['combined_strength_home'] - df_all['combined_strength_away']

# Predicted score from Correct Score
df_all['pred_cs_total'] = df_all['pred_score_home'].fillna(0) + df_all['pred_score_away'].fillna(0)
df_all['pred_cs_diff'] = df_all['pred_score_home'].fillna(0) - df_all['pred_score_away'].fillna(0)
df_all['pred_cs_min'] = df_all[['pred_score_home', 'pred_score_away']].min(axis=1)

print(f"Total features created. Dataset shape: {df_all.shape}")
print(f"\nTarget distribution:")
print(f"  Home Win:      {df_all['actual_home_win'].mean():.3f} ({df_all['actual_home_win'].sum()} / {len(df_all)})")
print(f"  Total > 2.5:   {df_all['actual_total_over25'].mean():.3f} ({df_all['actual_total_over25'].sum()} / {len(df_all)})")
print(f"  BTTS:          {df_all['actual_btts'].mean():.3f} ({df_all['actual_btts'].sum()} / {len(df_all)})")

# ============================================================
# STEP 3: Pattern Discovery & Correlation Analysis
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: PATTERN DISCOVERY & CORRELATION ANALYSIS")
print("=" * 70)

# PRE-MATCH features only (exclude post-match: Goals, Expected Goals)
# Also exclude bookmaker odds (pred_result_odds, pred_ou_odds, pred_bts_odds, pred_score_odds)
numeric_features = [
    'Team Rating (Home)', 'Team Rating (Away)',
    'Team Form (Home)', 'Team Form (Away)',
    'XG Luckiness (Home)', 'XG Luckiness (Away)',
    'XG Predictability (Home)', 'XG Predictability (Away)',
    'Avg XG Scored (Home)', 'Avg XG Scored (Away)',
    'Avg XG Conceded (Home)', 'Avg XG Conceded (Away)',
    'Match Score Prediction (Home)', 'Match Score Prediction (Away)',
    # Derived features
    'rating_diff', 'rating_sum', 'rating_ratio',
    'form_diff', 'form_sum', 'form_ratio',
    'xg_luck_diff', 'xg_luck_sum',
    'xg_pred_diff', 'xg_pred_sum', 'xg_pred_avg',
    'pred_goal_diff', 'pred_total_goals', 'pred_goal_ratio',
    'pred_min_goals', 'pred_max_goals',
    'avg_xg_scored_diff', 'avg_xg_conceded_diff',
    'avg_xg_total_scored', 'avg_xg_total_conceded',
    'home_attack_vs_away_def', 'away_attack_vs_home_def',
    'rating_form_home', 'rating_form_away', 'rating_form_diff',
    'combined_strength_home', 'combined_strength_away', 'combined_strength_diff',
    'pred_cs_total', 'pred_cs_diff', 'pred_cs_min',
    # Categorical encoded
    'pred_home_win', 'pred_draw', 'pred_away_win',
    'pred_over', 'pred_bts_yes',
    'pred_score_home', 'pred_score_away',
]

targets = ['actual_home_win', 'actual_total_over25', 'actual_btts']

# Correlation analysis for each target
from scipy import stats

results = {}
for target in targets:
    print(f"\n{'='*50}")
    print(f"CORRELATIONS WITH: {target}")
    print(f"{'='*50}")
    corr_data = []
    for feat in numeric_features:
        valid = df_all[[feat, target]].dropna()
        if len(valid) < 50:
            continue
        corr, pval = stats.pointbiserialr(valid[target], valid[feat])
        corr_data.append({
            'feature': feat,
            'correlation': corr,
            'abs_corr': abs(corr),
            'p_value': pval,
            'n_samples': len(valid)
        })

    corr_df = pd.DataFrame(corr_data).sort_values('abs_corr', ascending=False)
    results[target] = corr_df

    print("\nTop 20 features by correlation:")
    for _, row in corr_df.head(20).iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"  {row['feature']:45s} r={row['correlation']:+.4f}  p={row['p_value']:.6f} {sig}  n={row['n_samples']}")

# ============================================================
# STEP 4: Cross-feature interaction analysis
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: CROSS-FEATURE INTERACTION ANALYSIS")
print("=" * 70)

from itertools import combinations

top_features_per_target = {}
for target in targets:
    # Get top 15 features for each target
    top = results[target].head(15)['feature'].tolist()
    top_features_per_target[target] = top

# Test interaction effects for top features
for target in targets:
    print(f"\n{'='*50}")
    print(f"INTERACTION ANALYSIS: {target}")
    print(f"{'='*50}")

    top_feats = top_features_per_target[target][:10]
    interaction_results = []

    for f1, f2 in combinations(top_feats, 2):
        valid = df_all[[f1, f2, target]].dropna()
        if len(valid) < 50:
            continue

        # Create interaction term
        interaction = valid[f1] * valid[f2]
        corr, pval = stats.pointbiserialr(valid[target], interaction)

        # Check if interaction is stronger than individual features
        corr1, _ = stats.pointbiserialr(valid[target], valid[f1])
        corr2, _ = stats.pointbiserialr(valid[target], valid[f2])

        if abs(corr) > max(abs(corr1), abs(corr2)):
            interaction_results.append({
                'feature1': f1,
                'feature2': f2,
                'interaction_corr': corr,
                'f1_corr': corr1,
                'f2_corr': corr2,
                'improvement': abs(corr) - max(abs(corr1), abs(corr2)),
                'p_value': pval
            })

    if interaction_results:
        int_df = pd.DataFrame(interaction_results).sort_values('improvement', ascending=False)
        print(f"\nTop synergistic feature combinations (interaction > individual):")
        for _, row in int_df.head(10).iterrows():
            print(f"  {row['feature1']:30s} x {row['feature2']:30s}")
            print(f"    Interaction r={row['interaction_corr']:+.4f} vs individual r={row['f1_corr']:+.4f}, r={row['f2_corr']:+.4f} (improvement: +{row['improvement']:.4f})")

# ============================================================
# STEP 5: Bin analysis - find optimal thresholds
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: THRESHOLD & BIN ANALYSIS")
print("=" * 70)

for target in targets:
    print(f"\n{'='*50}")
    print(f"BIN ANALYSIS: {target}")
    print(f"{'='*50}")

    top_feats = results[target].head(10)['feature'].tolist()

    for feat in top_feats:
        valid = df_all[[feat, target]].dropna()
        if len(valid) < 100:
            continue

        # Create quantile bins
        try:
            valid['bin'] = pd.qcut(valid[feat], q=5, duplicates='drop')
            bin_stats = valid.groupby('bin')[target].agg(['mean', 'count', 'sum'])
            bin_stats.columns = ['hit_rate', 'total', 'hits']

            best_bin = bin_stats['hit_rate'].idxmax()
            worst_bin = bin_stats['hit_rate'].idxmin()

            if bin_stats['hit_rate'].max() - bin_stats['hit_rate'].min() > 0.10:
                print(f"\n  {feat}:")
                for idx, row in bin_stats.iterrows():
                    bar = "█" * int(row['hit_rate'] * 40)
                    print(f"    {str(idx):35s} hit={row['hit_rate']:.3f} ({int(row['hits'])}/{int(row['total'])}) {bar}")
        except:
            continue

# Save processed data
df_all.to_csv('/home/user/allinone-prediction/processed_data.csv', index=False)
print(f"\nProcessed data saved: {df_all.shape}")

# Save feature importance rankings
summary = {}
for target in targets:
    summary[target] = results[target][['feature', 'correlation', 'p_value', 'n_samples']].to_dict('records')

with open('/home/user/allinone-prediction/feature_analysis.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\nAnalysis complete!")
