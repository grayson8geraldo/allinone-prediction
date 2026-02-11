#!/usr/bin/env python3
"""
=================================================================
  FOOTBALL MATCH PREDICTION SYSTEM
  3 прогноза за 1 ввод данных:
    1. Победа хозяев (Home Win)
    2. Тотал больше 2.5 (Total Over 2.5)
    3. Обе команды забьют (Both Teams To Score)
=================================================================

Использование:
  python3 predict.py                    # Интерактивный ввод одного матча
  python3 predict.py --batch input.csv  # Пакетный ввод из CSV
  python3 predict.py --example          # Показать пример ввода

Необходимые данные предматчевой статистики:
  - Team Rating (Home/Away)        : Рейтинг команды (напр. 7.5)
  - Team Form (Home/Away)          : Форма команды (напр. 6.0)
  - XG Luckiness (Home/Away)       : XG везение (напр. 2.0, -1.5)
  - XG Predictability (Home/Away)  : XG предсказуемость % (напр. 75)
  - Match Score Prediction (Home/Away): Прогноз голов (напр. 1.8, 0.9)
  - Predicted Score (Home/Away)    : Прогноз точного счёта (напр. 2, 1)
  - Predicted Result               : Home Win / Draw / Away Win
  - Predicted Over/Under           : Over / Under
  - Predicted BTTS                 : Yes / No
  - Avg XG Scored (Home/Away)      : Средний xG забитых (опционально)
  - Avg XG Conceded (Home/Away)    : Средний xG пропущенных (опционально)
"""

import pickle
import numpy as np
import pandas as pd
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD MODELS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'final_models.pkl')

with open(MODEL_PATH, 'rb') as f:
    models_data = pickle.load(f)

MEDIAN_VALUES = models_data['median_values']


def compute_features(match_data):
    """
    Compute all derived features from raw match input.

    match_data: dict with keys:
        team_home, team_away,
        team_rating_home, team_rating_away,
        team_form_home, team_form_away,
        xg_luckiness_home, xg_luckiness_away,
        xg_predictability_home, xg_predictability_away,
        match_score_pred_home, match_score_pred_away,
        pred_score_home, pred_score_away,  (from Correct Score prediction)
        pred_result: 'Home Win' / 'Draw' / 'Away Win',
        pred_over_under: 'Over' / 'Under',
        pred_btts: 'Yes' / 'No',
        avg_xg_scored_home, avg_xg_scored_away,      (optional)
        avg_xg_conceded_home, avg_xg_conceded_away,  (optional)
    """
    d = match_data

    # Base features
    features = {}
    features['Team Rating (Home)'] = d['team_rating_home']
    features['Team Rating (Away)'] = d['team_rating_away']
    features['Team Form (Home)'] = d['team_form_home']
    features['Team Form (Away)'] = d['team_form_away']
    features['XG Luckiness (Home)'] = d['xg_luckiness_home']
    features['XG Luckiness (Away)'] = d['xg_luckiness_away']
    features['XG Predictability (Home)'] = d['xg_predictability_home']
    features['XG Predictability (Away)'] = d['xg_predictability_away']
    features['Match Score Prediction (Home)'] = d['match_score_pred_home']
    features['Match Score Prediction (Away)'] = d['match_score_pred_away']
    features['pred_score_home'] = d['pred_score_home']
    features['pred_score_away'] = d['pred_score_away']

    # Avg XG features (optional - use median if not provided)
    features['Avg XG Scored (Home)'] = d.get('avg_xg_scored_home', MEDIAN_VALUES.get('Avg XG Scored (Home)', 1.3))
    features['Avg XG Scored (Away)'] = d.get('avg_xg_scored_away', MEDIAN_VALUES.get('Avg XG Scored (Away)', 1.2))
    features['Avg XG Conceded (Home)'] = d.get('avg_xg_conceded_home', MEDIAN_VALUES.get('Avg XG Conceded (Home)', 1.2))
    features['Avg XG Conceded (Away)'] = d.get('avg_xg_conceded_away', MEDIAN_VALUES.get('Avg XG Conceded (Away)', 1.3))

    # Categorical features
    features['pred_home_win'] = 1 if d['pred_result'] == 'Home Win' else 0
    features['pred_draw'] = 1 if d['pred_result'] == 'Draw' else 0
    features['pred_away_win'] = 1 if d['pred_result'] == 'Away Win' else 0
    features['pred_over'] = 1 if d['pred_over_under'] == 'Over' else 0
    features['pred_bts_yes'] = 1 if d['pred_btts'] == 'Yes' else 0

    # Derived features
    features['rating_diff'] = features['Team Rating (Home)'] - features['Team Rating (Away)']
    features['rating_sum'] = features['Team Rating (Home)'] + features['Team Rating (Away)']
    tr_away = features['Team Rating (Away)'] if features['Team Rating (Away)'] != 0 else 0.1
    features['rating_ratio'] = features['Team Rating (Home)'] / tr_away

    features['form_diff'] = features['Team Form (Home)'] - features['Team Form (Away)']
    features['form_sum'] = features['Team Form (Home)'] + features['Team Form (Away)']
    tf_away = features['Team Form (Away)'] if features['Team Form (Away)'] != 0 else 0.1
    features['form_ratio'] = features['Team Form (Home)'] / tf_away

    features['xg_luck_diff'] = features['XG Luckiness (Home)'] - features['XG Luckiness (Away)']
    features['xg_luck_sum'] = features['XG Luckiness (Home)'] + features['XG Luckiness (Away)']
    features['xg_pred_diff'] = features['XG Predictability (Home)'] - features['XG Predictability (Away)']
    features['xg_pred_sum'] = features['XG Predictability (Home)'] + features['XG Predictability (Away)']
    features['xg_pred_avg'] = (features['XG Predictability (Home)'] + features['XG Predictability (Away)']) / 2

    msp_away = features['Match Score Prediction (Away)'] if features['Match Score Prediction (Away)'] != 0 else 0.1
    features['pred_goal_diff'] = features['Match Score Prediction (Home)'] - features['Match Score Prediction (Away)']
    features['pred_total_goals'] = features['Match Score Prediction (Home)'] + features['Match Score Prediction (Away)']
    features['pred_goal_ratio'] = features['Match Score Prediction (Home)'] / msp_away
    features['pred_min_goals'] = min(features['Match Score Prediction (Home)'], features['Match Score Prediction (Away)'])
    features['pred_max_goals'] = max(features['Match Score Prediction (Home)'], features['Match Score Prediction (Away)'])

    features['rating_form_home'] = features['Team Rating (Home)'] * features['Team Form (Home)']
    features['rating_form_away'] = features['Team Rating (Away)'] * features['Team Form (Away)']
    features['rating_form_diff'] = features['rating_form_home'] - features['rating_form_away']
    features['combined_strength_home'] = (features['Team Rating (Home)'] + features['Team Form (Home)']) / 2
    features['combined_strength_away'] = (features['Team Rating (Away)'] + features['Team Form (Away)']) / 2
    features['combined_strength_diff'] = features['combined_strength_home'] - features['combined_strength_away']

    features['pred_cs_total'] = features['pred_score_home'] + features['pred_score_away']
    features['pred_cs_diff'] = features['pred_score_home'] - features['pred_score_away']
    features['pred_cs_min'] = min(features['pred_score_home'], features['pred_score_away'])

    features['avg_xg_scored_diff'] = features['Avg XG Scored (Home)'] - features['Avg XG Scored (Away)']
    features['avg_xg_conceded_diff'] = features['Avg XG Conceded (Home)'] - features['Avg XG Conceded (Away)']
    features['avg_xg_total_scored'] = features['Avg XG Scored (Home)'] + features['Avg XG Scored (Away)']
    features['avg_xg_total_conceded'] = features['Avg XG Conceded (Home)'] + features['Avg XG Conceded (Away)']
    features['home_attack_vs_away_def'] = features['Avg XG Scored (Home)'] - features['Avg XG Conceded (Away)']
    features['away_attack_vs_home_def'] = features['Avg XG Scored (Away)'] - features['Avg XG Conceded (Home)']

    return features


def predict_match(match_data):
    """
    Make 3 predictions for a single match.
    Returns dict with predictions and confidence levels.
    """
    features = compute_features(match_data)

    results = {}

    # ========== 1. HOME WIN ==========
    hw_data = models_data['home_win']
    hw_feats = [features[f] for f in hw_data['features']]
    hw_arr = np.array(hw_feats).reshape(1, -1)
    hw_arr_s = hw_data['scaler'].transform(hw_arr)
    hw_prob = hw_data['model'].predict_proba(hw_arr_s)[0][1]

    # Confidence level
    if hw_prob >= 0.75:
        hw_confidence = "ВЫСОКАЯ"
    elif hw_prob >= 0.65:
        hw_confidence = "СРЕДНЯЯ"
    elif hw_prob >= 0.55:
        hw_confidence = "НИЗКАЯ"
    else:
        hw_confidence = "НЕ РЕКОМЕНДУЕТСЯ"

    results['home_win'] = {
        'probability': hw_prob,
        'prediction': 'ДА' if hw_prob >= 0.50 else 'НЕТ',
        'confidence': hw_confidence,
    }

    # ========== 2. OVER 2.5 ==========
    ou_data = models_data['over25']
    ou_feats = [features.get(f, 0) for f in ou_data['features']]
    ou_arr = np.array(ou_feats).reshape(1, -1)
    ou_arr_s = ou_data['scaler'].transform(ou_arr)

    # Weighted ensemble prediction
    ou_probs = []
    for name, model in ou_data['models'].items():
        ou_probs.append(model.predict_proba(ou_arr_s)[0][1])
    ou_prob = np.average(ou_probs, weights=ou_data['weights'])

    # Apply rule-based adjustments
    ptg = features['pred_total_goals']
    pmg = features['pred_max_goals']
    msp_h = features['Match Score Prediction (Home)']

    rule_adj = 0.0
    if ptg > 3.2:
        rule_adj += 0.08
    elif ptg > 2.9:
        rule_adj += 0.03
    elif ptg <= 2.4:
        rule_adj -= 0.08

    if pmg > 2.1:
        rule_adj += 0.05
    elif pmg <= 1.4:
        rule_adj -= 0.05

    if msp_h > 1.9:
        rule_adj += 0.05
    elif msp_h <= 1.1:
        rule_adj -= 0.03

    ou_prob = np.clip(ou_prob + rule_adj, 0, 1)

    if ou_prob >= 0.70:
        ou_confidence = "ВЫСОКАЯ"
    elif ou_prob >= 0.60:
        ou_confidence = "СРЕДНЯЯ"
    elif ou_prob >= 0.55:
        ou_confidence = "НИЗКАЯ"
    else:
        ou_confidence = "НЕ РЕКОМЕНДУЕТСЯ"

    results['over25'] = {
        'probability': ou_prob,
        'prediction': 'ДА' if ou_prob >= 0.55 else 'НЕТ',
        'confidence': ou_confidence,
    }

    # ========== 3. BTTS ==========
    btts_data = models_data['btts']
    btts_feats = [features.get(f, 0) for f in btts_data['features']]
    btts_arr = np.array(btts_feats).reshape(1, -1)
    btts_arr_s = btts_data['scaler'].transform(btts_arr)

    btts_probs = []
    for name, model in btts_data['models'].items():
        btts_probs.append(model.predict_proba(btts_arr_s)[0][1])
    btts_prob = np.average(btts_probs, weights=btts_data['weights'])

    # Apply rule-based adjustments
    pmin = features['pred_min_goals']
    psh = features['pred_score_home']
    psa = features['pred_score_away']
    axc_a = features.get('Avg XG Conceded (Away)', MEDIAN_VALUES.get('Avg XG Conceded (Away)', 1.3))

    rule_adj = 0.0
    if pmin > 1.3:
        rule_adj += 0.06
    elif pmin <= 0.9:
        rule_adj -= 0.03

    if ptg > 3.2:
        rule_adj += 0.04
    elif ptg <= 2.4:
        rule_adj -= 0.04

    if psh > 0 and psa > 0:
        rule_adj += 0.03
    if psh == 0 or psa == 0:
        rule_adj -= 0.05

    if axc_a > 1.8:
        rule_adj += 0.05
    elif axc_a <= 1.1:
        rule_adj -= 0.03

    btts_prob = np.clip(btts_prob + rule_adj, 0, 1)

    if btts_prob >= 0.65:
        btts_confidence = "ВЫСОКАЯ"
    elif btts_prob >= 0.58:
        btts_confidence = "СРЕДНЯЯ"
    elif btts_prob >= 0.52:
        btts_confidence = "НИЗКАЯ"
    else:
        btts_confidence = "НЕ РЕКОМЕНДУЕТСЯ"

    results['btts'] = {
        'probability': btts_prob,
        'prediction': 'ДА' if btts_prob >= 0.55 else 'НЕТ',
        'confidence': btts_confidence,
    }

    return results


def print_prediction(match_data, results):
    """Pretty print prediction results."""
    print("\n" + "=" * 65)
    print(f"  {match_data.get('team_home', 'Home')} vs {match_data.get('team_away', 'Away')}")
    print("=" * 65)

    # Home Win
    hw = results['home_win']
    hw_bar = "█" * int(hw['probability'] * 30) + "░" * (30 - int(hw['probability'] * 30))
    print(f"\n  ⚽ ПОБЕДА ХОЗЯЕВ (Home Win)")
    print(f"     Прогноз:     {hw['prediction']}")
    print(f"     Вероятность:  {hw['probability']:.1%}  [{hw_bar}]")
    print(f"     Уверенность:  {hw['confidence']}")

    # Over 2.5
    ou = results['over25']
    ou_bar = "█" * int(ou['probability'] * 30) + "░" * (30 - int(ou['probability'] * 30))
    print(f"\n  ⚽ ТОТАЛ БОЛЬШЕ 2.5 (Over 2.5)")
    print(f"     Прогноз:     {ou['prediction']}")
    print(f"     Вероятность:  {ou['probability']:.1%}  [{ou_bar}]")
    print(f"     Уверенность:  {ou['confidence']}")

    # BTTS
    bt = results['btts']
    bt_bar = "█" * int(bt['probability'] * 30) + "░" * (30 - int(bt['probability'] * 30))
    print(f"\n  ⚽ ОБЕ ЗАБЬЮТ (Both Teams To Score)")
    print(f"     Прогноз:     {bt['prediction']}")
    print(f"     Вероятность:  {bt['probability']:.1%}  [{bt_bar}]")
    print(f"     Уверенность:  {bt['confidence']}")

    # Summary recommendation
    print(f"\n  {'─' * 60}")
    print(f"  РЕКОМЕНДАЦИЯ:")
    recs = []
    if hw['confidence'] in ['ВЫСОКАЯ', 'СРЕДНЯЯ']:
        recs.append(f"    ✓ Победа хозяев ({hw['probability']:.0%})")
    if ou['confidence'] in ['ВЫСОКАЯ', 'СРЕДНЯЯ']:
        recs.append(f"    ✓ Тотал больше 2.5 ({ou['probability']:.0%})")
    if bt['confidence'] in ['ВЫСОКАЯ', 'СРЕДНЯЯ']:
        recs.append(f"    ✓ Обе забьют ({bt['probability']:.0%})")

    if recs:
        for r in recs:
            print(r)
    else:
        print("    ✗ Нет уверенных прогнозов для этого матча")

    print("=" * 65)


def input_float(prompt, default=None):
    """Get float input with optional default."""
    while True:
        if default is not None:
            val = input(f"  {prompt} [{default}]: ").strip()
            if val == '':
                return default
        else:
            val = input(f"  {prompt}: ").strip()
        try:
            return float(val)
        except ValueError:
            print("  ⚠ Введите числовое значение")


def input_choice(prompt, choices):
    """Get choice input."""
    choices_str = " / ".join(choices)
    while True:
        val = input(f"  {prompt} ({choices_str}): ").strip()
        # Try matching by first letter or full name
        for c in choices:
            if val.lower() == c.lower() or val.lower() == c[0].lower():
                return c
        print(f"  ⚠ Выберите один из вариантов: {choices_str}")


def interactive_input():
    """Interactively collect match data from user."""
    print("\n" + "=" * 65)
    print("  ВВОД ДАННЫХ ПРЕДСТОЯЩЕГО МАТЧА")
    print("=" * 65)

    match_data = {}
    match_data['team_home'] = input("  Команда хозяев: ").strip() or "Home"
    match_data['team_away'] = input("  Команда гостей: ").strip() or "Away"

    print("\n  --- Рейтинги и форма ---")
    match_data['team_rating_home'] = input_float("Team Rating (Home)")
    match_data['team_rating_away'] = input_float("Team Rating (Away)")
    match_data['team_form_home'] = input_float("Team Form (Home)")
    match_data['team_form_away'] = input_float("Team Form (Away)")

    print("\n  --- XG статистика ---")
    match_data['xg_luckiness_home'] = input_float("XG Luckiness (Home)")
    match_data['xg_luckiness_away'] = input_float("XG Luckiness (Away)")
    match_data['xg_predictability_home'] = input_float("XG Predictability (Home) %")
    match_data['xg_predictability_away'] = input_float("XG Predictability (Away) %")

    print("\n  --- Прогнозы голов ---")
    match_data['match_score_pred_home'] = input_float("Match Score Prediction (Home)")
    match_data['match_score_pred_away'] = input_float("Match Score Prediction (Away)")

    print("\n  --- Прогноз точного счёта ---")
    match_data['pred_score_home'] = input_float("Predicted Score (Home)")
    match_data['pred_score_away'] = input_float("Predicted Score (Away)")

    print("\n  --- Прогнозы результата ---")
    match_data['pred_result'] = input_choice("Predicted Result", ["Home Win", "Draw", "Away Win"])
    match_data['pred_over_under'] = input_choice("Predicted Over/Under 2.5", ["Over", "Under"])
    match_data['pred_btts'] = input_choice("Predicted BTTS", ["Yes", "No"])

    print("\n  --- Avg XG (опционально, Enter для пропуска) ---")
    match_data['avg_xg_scored_home'] = input_float("Avg XG Scored (Home)", MEDIAN_VALUES.get('Avg XG Scored (Home)', 1.3))
    match_data['avg_xg_scored_away'] = input_float("Avg XG Scored (Away)", MEDIAN_VALUES.get('Avg XG Scored (Away)', 1.2))
    match_data['avg_xg_conceded_home'] = input_float("Avg XG Conceded (Home)", MEDIAN_VALUES.get('Avg XG Conceded (Home)', 1.2))
    match_data['avg_xg_conceded_away'] = input_float("Avg XG Conceded (Away)", MEDIAN_VALUES.get('Avg XG Conceded (Away)', 1.3))

    return match_data


def batch_predict(csv_path):
    """Process batch predictions from CSV file."""
    df = pd.read_csv(csv_path)
    print(f"\nЗагружено {len(df)} матчей из {csv_path}")

    required_cols = [
        'team_home', 'team_away',
        'team_rating_home', 'team_rating_away',
        'team_form_home', 'team_form_away',
        'xg_luckiness_home', 'xg_luckiness_away',
        'xg_predictability_home', 'xg_predictability_away',
        'match_score_pred_home', 'match_score_pred_away',
        'pred_score_home', 'pred_score_away',
        'pred_result', 'pred_over_under', 'pred_btts',
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠ Отсутствуют колонки: {missing}")
        return

    all_results = []
    for idx, row in df.iterrows():
        match_data = row.to_dict()
        # Handle optional avg_xg columns
        for col in ['avg_xg_scored_home', 'avg_xg_scored_away',
                     'avg_xg_conceded_home', 'avg_xg_conceded_away']:
            if col not in match_data or pd.isna(match_data.get(col)):
                match_data[col] = MEDIAN_VALUES.get(col.replace('avg_xg_', 'Avg XG ').replace('_', ' ').title(), 1.3)

        results = predict_match(match_data)
        print_prediction(match_data, results)

        all_results.append({
            'team_home': match_data['team_home'],
            'team_away': match_data['team_away'],
            'home_win_prob': results['home_win']['probability'],
            'home_win_pred': results['home_win']['prediction'],
            'home_win_conf': results['home_win']['confidence'],
            'over25_prob': results['over25']['probability'],
            'over25_pred': results['over25']['prediction'],
            'over25_conf': results['over25']['confidence'],
            'btts_prob': results['btts']['probability'],
            'btts_pred': results['btts']['prediction'],
            'btts_conf': results['btts']['confidence'],
        })

    # Save results
    out_path = csv_path.replace('.csv', '_predictions.csv')
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\nРезультаты сохранены в: {out_path}")


def show_example():
    """Show example with a sample match."""
    print("\n  Пример прогноза для матча Chelsea vs Fulham:")
    match_data = {
        'team_home': 'Chelsea',
        'team_away': 'Fulham',
        'team_rating_home': 7.7,
        'team_rating_away': 6.2,
        'team_form_home': 7.8,
        'team_form_away': 5.5,
        'xg_luckiness_home': 2.0,
        'xg_luckiness_away': 1.3,
        'xg_predictability_home': 76,
        'xg_predictability_away': 75,
        'match_score_pred_home': 2.0,
        'match_score_pred_away': 0.9,
        'pred_score_home': 2,
        'pred_score_away': 1,
        'pred_result': 'Home Win',
        'pred_over_under': 'Over',
        'pred_btts': 'Yes',
        'avg_xg_scored_home': 2.0,
        'avg_xg_scored_away': 1.3,
        'avg_xg_conceded_home': 0.7,
        'avg_xg_conceded_away': 1.9,
    }
    results = predict_match(match_data)
    print_prediction(match_data, results)

    print("\n  Формат CSV для пакетного ввода (--batch):")
    print("  team_home,team_away,team_rating_home,team_rating_away,team_form_home,"
          "team_form_away,xg_luckiness_home,xg_luckiness_away,xg_predictability_home,"
          "xg_predictability_away,match_score_pred_home,match_score_pred_away,"
          "pred_score_home,pred_score_away,pred_result,pred_over_under,pred_btts,"
          "avg_xg_scored_home,avg_xg_scored_away,avg_xg_conceded_home,avg_xg_conceded_away")
    print("  Chelsea,Fulham,7.7,6.2,7.8,5.5,2.0,1.3,76,75,2.0,0.9,2,1,Home Win,Over,Yes,2.0,1.3,0.7,1.9")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 65)
    print("  ⚽  СИСТЕМА ПРОГНОЗИРОВАНИЯ ФУТБОЛЬНЫХ МАТЧЕЙ  ⚽")
    print("  3 прогноза: Победа хозяев | Тотал > 2.5 | Обе забьют")
    print("=" * 65)

    if len(sys.argv) > 1:
        if sys.argv[1] == '--example':
            show_example()
        elif sys.argv[1] == '--batch' and len(sys.argv) > 2:
            batch_predict(sys.argv[2])
        else:
            print(f"Использование:")
            print(f"  python3 {sys.argv[0]}              # Интерактивный ввод")
            print(f"  python3 {sys.argv[0]} --example     # Пример")
            print(f"  python3 {sys.argv[0]} --batch f.csv # Пакетный режим")
    else:
        while True:
            match_data = interactive_input()
            results = predict_match(match_data)
            print_prediction(match_data, results)

            again = input("\n  Ещё один матч? (y/n): ").strip().lower()
            if again != 'y':
                print("\n  До свидания! Удачных прогнозов!")
                break
