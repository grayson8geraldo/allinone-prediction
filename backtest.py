"""
Backtest - Проверка системы на исторических данных
===================================================
Проверяем точность прогнозов при разных уровнях уверенности.
"""

import pandas as pd
import numpy as np
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

# Load models
with open('/home/user/allinone-prediction/final_models.pkl', 'rb') as f:
    models_data = pickle.load(f)

MEDIAN_VALUES = models_data['median_values']

# Load original data
df1 = pd.read_excel('/home/user/allinone-prediction/All 1ligi (1).xlsx')
df2 = pd.read_excel('/home/user/allinone-prediction/football_statistics.xlsx')

# Prepare df2 (has more features)
def parse_percentage(val):
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

def parse_field(val, pattern):
    if pd.isna(val):
        return None
    m = re.match(pattern, str(val))
    return m if m else None

# Use the predict function
from predict import predict_match

def row_to_match_data(row, has_avg_xg=False):
    """Convert dataframe row to match_data dict."""
    # Parse prediction result
    m = parse_field(row['Home / Away'], r'(Home Win|Away Win|Draw)\s*\(([0-9.]+)\)')
    pred_result = m.group(1) if m else 'Draw'

    # Parse over/under
    m = parse_field(row['Over / Under'], r'Total (Over|Under) 2\.5\s*\(([0-9.]+)\)')
    pred_ou = m.group(1) if m else 'Under'

    # Parse BTTS
    m = parse_field(row['Both To Score'], r'Both To Score:\s*(Yes|No)\s*\(([0-9.]+)\)')
    pred_btts = m.group(1) if m else 'No'

    # Parse correct score
    m = parse_field(row['Correct Score'], r'Correct Score:\s*(\d+)-(\d+)\s*\(([0-9.]+)\)')
    pred_sh = int(m.group(1)) if m else 1
    pred_sa = int(m.group(2)) if m else 0

    match = {
        'team_home': row.get('Team (Home)', 'Home'),
        'team_away': row.get('Team (Away)', 'Away'),
        'team_rating_home': row['Team Rating (Home)'],
        'team_rating_away': row['Team Rating (Away)'],
        'team_form_home': row['Team Form (Home)'],
        'team_form_away': row['Team Form (Away)'],
        'xg_luckiness_home': row['XG Luckiness (Home)'],
        'xg_luckiness_away': row['XG Luckiness (Away)'],
        'xg_predictability_home': row['XG Predictability (Home)'],
        'xg_predictability_away': row['XG Predictability (Away)'],
        'match_score_pred_home': row['Match Score Prediction (Home)'],
        'match_score_pred_away': row['Match Score Prediction (Away)'],
        'pred_score_home': pred_sh,
        'pred_score_away': pred_sa,
        'pred_result': pred_result,
        'pred_over_under': pred_ou,
        'pred_btts': pred_btts,
    }

    if has_avg_xg:
        match['avg_xg_scored_home'] = row.get('Avg XG Scored (Home)', MEDIAN_VALUES.get('Avg XG Scored (Home)', 1.3))
        match['avg_xg_scored_away'] = row.get('Avg XG Scored (Away)', MEDIAN_VALUES.get('Avg XG Scored (Away)', 1.2))
        match['avg_xg_conceded_home'] = row.get('Avg XG Conceded (Home)', MEDIAN_VALUES.get('Avg XG Conceded (Home)', 1.2))
        match['avg_xg_conceded_away'] = row.get('Avg XG Conceded (Away)', MEDIAN_VALUES.get('Avg XG Conceded (Away)', 1.3))

    return match


# Run backtest on df2 (has all features)
print("=" * 70)
print("BACKTEST НА ИСТОРИЧЕСКИХ ДАННЫХ (football_statistics.xlsx)")
print("=" * 70)

# Remove rows with NaN goals (no results)
df2_valid = df2.dropna(subset=['Goals (Home)', 'Goals (Away)'])
print(f"Матчей для теста: {len(df2_valid)}")

results_list = []
errors = 0
for idx, row in df2_valid.iterrows():
    try:
        match = row_to_match_data(row, has_avg_xg=True)
        preds = predict_match(match)

        actual_hw = int(row['Goals (Home)'] > row['Goals (Away)'])
        actual_ou = int(row['Goals (Home)'] + row['Goals (Away)'] > 2.5)
        actual_btts = int(row['Goals (Home)'] > 0 and row['Goals (Away)'] > 0)

        results_list.append({
            'hw_prob': preds['home_win']['probability'],
            'ou_prob': preds['over25']['probability'],
            'btts_prob': preds['btts']['probability'],
            'actual_hw': actual_hw,
            'actual_ou': actual_ou,
            'actual_btts': actual_btts,
        })
    except Exception as e:
        errors += 1

if errors:
    print(f"Ошибки при обработке: {errors}")

res_df = pd.DataFrame(results_list)
print(f"Успешно обработано: {len(res_df)}")

# Analyze by confidence thresholds
print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ БЭКТЕСТА ПО УРОВНЯМ УВЕРЕННОСТИ")
print("=" * 70)

for target, prob_col, actual_col, name in [
    ('home_win', 'hw_prob', 'actual_hw', 'ПОБЕДА ХОЗЯЕВ'),
    ('over25', 'ou_prob', 'actual_ou', 'ТОТАЛ > 2.5'),
    ('btts', 'btts_prob', 'actual_btts', 'ОБЕ ЗАБЬЮТ'),
]:
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    base_rate = res_df[actual_col].mean()
    print(f"  Базовая частота события: {base_rate:.1%}")

    print(f"\n  {'Порог':>8} {'Матчей':>8} {'Точность':>10} {'Подъём':>8} {'% данных':>10}")
    print(f"  {'-'*50}")

    for thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = res_df[prob_col] >= thr
        n = mask.sum()
        if n < 10:
            continue
        hit = res_df.loc[mask, actual_col].mean()
        lift = hit / base_rate
        pct = n / len(res_df) * 100
        marker = " ★" if hit > base_rate * 1.15 else ""
        print(f"  {thr:>8.2f} {n:>8} {hit:>10.1%} {lift:>7.2f}x {pct:>9.1f}%{marker}")

# Also test on df1 (without avg_xg features)
print("\n\n" + "=" * 70)
print("BACKTEST НА ДАННЫХ БЕЗ AVG XG (All 1ligi.xlsx)")
print("=" * 70)

df1_valid = df1.dropna(subset=['Goals (Home)', 'Goals (Away)'])
# Drop rows with NaN in predictability
df1_valid = df1_valid.dropna(subset=['XG Predictability (Home)', 'XG Predictability (Away)'])
print(f"Матчей для теста: {len(df1_valid)}")

results_list2 = []
for idx, row in df1_valid.iterrows():
    try:
        match = row_to_match_data(row, has_avg_xg=False)
        preds = predict_match(match)

        actual_hw = int(row['Goals (Home)'] > row['Goals (Away)'])
        actual_ou = int(row['Goals (Home)'] + row['Goals (Away)'] > 2.5)
        actual_btts = int(row['Goals (Home)'] > 0 and row['Goals (Away)'] > 0)

        results_list2.append({
            'hw_prob': preds['home_win']['probability'],
            'ou_prob': preds['over25']['probability'],
            'btts_prob': preds['btts']['probability'],
            'actual_hw': actual_hw,
            'actual_ou': actual_ou,
            'actual_btts': actual_btts,
        })
    except:
        pass

res_df2 = pd.DataFrame(results_list2)
print(f"Успешно обработано: {len(res_df2)}")

for target, prob_col, actual_col, name in [
    ('home_win', 'hw_prob', 'actual_hw', 'ПОБЕДА ХОЗЯЕВ'),
    ('over25', 'ou_prob', 'actual_ou', 'ТОТАЛ > 2.5'),
    ('btts', 'btts_prob', 'actual_btts', 'ОБЕ ЗАБЬЮТ'),
]:
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    base_rate = res_df2[actual_col].mean()
    print(f"  Базовая частота события: {base_rate:.1%}")

    print(f"\n  {'Порог':>8} {'Матчей':>8} {'Точность':>10} {'Подъём':>8} {'% данных':>10}")
    print(f"  {'-'*50}")

    for thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = res_df2[prob_col] >= thr
        n = mask.sum()
        if n < 10:
            continue
        hit = res_df2.loc[mask, actual_col].mean()
        lift = hit / base_rate
        pct = n / len(res_df2) * 100
        marker = " ★" if hit > base_rate * 1.15 else ""
        print(f"  {thr:>8.2f} {n:>8} {hit:>10.1%} {lift:>7.2f}x {pct:>9.1f}%{marker}")

print("\nБэктест завершён!")
