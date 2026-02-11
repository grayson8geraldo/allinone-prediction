#!/usr/bin/env python3
"""
Football Match Prediction - Web Interface
==========================================
Flask web application for the prediction system.
Uses the exact same models and logic as predict.py.

Run: python3 web_app.py
Open: http://localhost:5000
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template_string, request, jsonify

# Import prediction engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict_match, MEDIAN_VALUES

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Football Prediction System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e17;
            color: #e0e6ed;
            min-height: 100vh;
        }

        .header {
            background: linear-gradient(135deg, #1a1f35 0%, #0d1220 100%);
            border-bottom: 1px solid #1e2a42;
            padding: 20px 0;
            text-align: center;
        }

        .header h1 {
            font-size: 24px;
            font-weight: 700;
            color: #fff;
            letter-spacing: 1px;
        }

        .header p {
            color: #6b7fa3;
            font-size: 14px;
            margin-top: 6px;
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 24px 16px;
        }

        /* Form Section */
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }

        .team-card {
            background: #111827;
            border: 1px solid #1e2a42;
            border-radius: 12px;
            padding: 20px;
        }

        .team-card.home { border-top: 3px solid #3b82f6; }
        .team-card.away { border-top: 3px solid #ef4444; }

        .team-label {
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }

        .team-card.home .team-label { color: #3b82f6; }
        .team-card.away .team-label { color: #ef4444; }

        .form-row {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            gap: 10px;
        }

        .form-row label {
            flex: 1;
            font-size: 13px;
            color: #8896ab;
            white-space: nowrap;
        }

        .form-row input {
            width: 90px;
            padding: 8px 10px;
            border: 1px solid #2a3652;
            border-radius: 8px;
            background: #0d1220;
            color: #e0e6ed;
            font-size: 14px;
            text-align: center;
            transition: border-color 0.2s;
        }

        .form-row input:focus {
            outline: none;
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
        }

        .team-name-input {
            width: 100% !important;
            text-align: left !important;
            margin-bottom: 14px;
            padding: 10px 12px !important;
            font-size: 15px !important;
            font-weight: 500;
        }

        /* Predictions Section */
        .predictions-card {
            background: #111827;
            border: 1px solid #1e2a42;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .section-title {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #6b7fa3;
            margin-bottom: 14px;
        }

        .select-row {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            gap: 10px;
        }

        .select-row label {
            flex: 1;
            font-size: 13px;
            color: #8896ab;
        }

        .select-row select {
            width: 160px;
            padding: 8px 10px;
            border: 1px solid #2a3652;
            border-radius: 8px;
            background: #0d1220;
            color: #e0e6ed;
            font-size: 14px;
            cursor: pointer;
        }

        .select-row select:focus {
            outline: none;
            border-color: #3b82f6;
        }

        /* Optional Section */
        .optional-toggle {
            background: none;
            border: 1px solid #2a3652;
            border-radius: 8px;
            color: #6b7fa3;
            padding: 10px 16px;
            font-size: 13px;
            cursor: pointer;
            width: 100%;
            text-align: left;
            margin-bottom: 12px;
            transition: all 0.2s;
        }

        .optional-toggle:hover {
            border-color: #3b82f6;
            color: #a0b0c8;
        }

        .optional-fields {
            display: none;
        }

        .optional-fields.show {
            display: block;
        }

        /* Button */
        .btn-predict {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            border: none;
            border-radius: 10px;
            color: #fff;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            letter-spacing: 0.5px;
            transition: all 0.3s;
            margin-top: 8px;
        }

        .btn-predict:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
        }

        .btn-predict:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        /* Results */
        #results {
            display: none;
            margin-top: 24px;
        }

        .results-header {
            text-align: center;
            padding: 16px;
            background: #111827;
            border: 1px solid #1e2a42;
            border-radius: 12px 12px 0 0;
            border-bottom: none;
        }

        .results-header h2 {
            font-size: 18px;
            color: #fff;
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0;
        }

        .result-item {
            background: #111827;
            border: 1px solid #1e2a42;
            padding: 24px 20px;
            text-align: center;
        }

        .result-item:first-child { border-radius: 0 0 0 12px; }
        .result-item:last-child { border-radius: 0 0 12px 0; }

        .result-icon {
            font-size: 28px;
            margin-bottom: 8px;
        }

        .result-name {
            font-size: 13px;
            color: #6b7fa3;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 16px;
            font-weight: 600;
        }

        .result-prediction {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .result-prediction.yes { color: #22c55e; }
        .result-prediction.no { color: #ef4444; }

        .probability-bar {
            width: 100%;
            height: 8px;
            background: #1e2a42;
            border-radius: 4px;
            margin: 12px 0 8px;
            overflow: hidden;
        }

        .probability-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.8s ease;
        }

        .prob-value {
            font-size: 22px;
            font-weight: 600;
            color: #e0e6ed;
        }

        .confidence-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-top: 10px;
        }

        .confidence-high {
            background: rgba(34, 197, 94, 0.15);
            color: #22c55e;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }

        .confidence-medium {
            background: rgba(251, 191, 36, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(251, 191, 36, 0.3);
        }

        .confidence-low {
            background: rgba(251, 146, 60, 0.15);
            color: #fb923c;
            border: 1px solid rgba(251, 146, 60, 0.3);
        }

        .confidence-none {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* Recommendation */
        .recommendation {
            background: #111827;
            border: 1px solid #1e2a42;
            border-top: none;
            border-radius: 0 0 12px 12px;
            padding: 20px;
            text-align: center;
        }

        .rec-title {
            font-size: 12px;
            color: #6b7fa3;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 10px;
        }

        .rec-items {
            display: flex;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
        }

        .rec-item {
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
        }

        .rec-item.positive {
            background: rgba(34, 197, 94, 0.1);
            color: #22c55e;
            border: 1px solid rgba(34, 197, 94, 0.2);
        }

        .rec-item.negative {
            background: rgba(107, 114, 128, 0.1);
            color: #6b7280;
            border: 1px solid rgba(107, 114, 128, 0.2);
        }

        /* Loading */
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
            color: #6b7fa3;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid #1e2a42;
            border-top: 3px solid #3b82f6;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 12px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Responsive */
        @media (max-width: 768px) {
            .form-grid { grid-template-columns: 1fr; }
            .results-grid { grid-template-columns: 1fr; }
            .result-item:first-child { border-radius: 0; }
            .result-item:last-child { border-radius: 0 0 12px 12px; }
        }

        /* Tooltip */
        .info-tip {
            color: #4a5568;
            font-size: 11px;
            margin-left: 4px;
            cursor: help;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>FOOTBALL PREDICTION SYSTEM</h1>
        <p>Home Win | Total Over 2.5 | Both Teams To Score</p>
    </div>

    <div class="container">
        <form id="predForm">
            <!-- Team Cards -->
            <div class="form-grid">
                <!-- HOME -->
                <div class="team-card home">
                    <div class="team-label">Home Team</div>
                    <input type="text" name="team_home" class="team-name-input" placeholder="Team name" value="Chelsea">

                    <div class="section-title">Ratings & Form</div>
                    <div class="form-row">
                        <label>Team Rating</label>
                        <input type="number" step="0.1" name="team_rating_home" value="7.7" required>
                    </div>
                    <div class="form-row">
                        <label>Team Form</label>
                        <input type="number" step="0.1" name="team_form_home" value="7.8" required>
                    </div>

                    <div class="section-title" style="margin-top:16px">XG Statistics</div>
                    <div class="form-row">
                        <label>XG Luckiness</label>
                        <input type="number" step="0.1" name="xg_luckiness_home" value="2.0" required>
                    </div>
                    <div class="form-row">
                        <label>XG Predictability %</label>
                        <input type="number" step="1" name="xg_predictability_home" value="76" required>
                    </div>

                    <div class="section-title" style="margin-top:16px">Score Predictions</div>
                    <div class="form-row">
                        <label>Match Score Prediction</label>
                        <input type="number" step="0.1" name="match_score_pred_home" value="2.0" required>
                    </div>
                    <div class="form-row">
                        <label>Predicted Score</label>
                        <input type="number" step="1" name="pred_score_home" value="2" required>
                    </div>
                </div>

                <!-- AWAY -->
                <div class="team-card away">
                    <div class="team-label">Away Team</div>
                    <input type="text" name="team_away" class="team-name-input" placeholder="Team name" value="Fulham">

                    <div class="section-title">Ratings & Form</div>
                    <div class="form-row">
                        <label>Team Rating</label>
                        <input type="number" step="0.1" name="team_rating_away" value="6.2" required>
                    </div>
                    <div class="form-row">
                        <label>Team Form</label>
                        <input type="number" step="0.1" name="team_form_away" value="5.5" required>
                    </div>

                    <div class="section-title" style="margin-top:16px">XG Statistics</div>
                    <div class="form-row">
                        <label>XG Luckiness</label>
                        <input type="number" step="0.1" name="xg_luckiness_away" value="1.3" required>
                    </div>
                    <div class="form-row">
                        <label>XG Predictability %</label>
                        <input type="number" step="1" name="xg_predictability_away" value="75" required>
                    </div>

                    <div class="section-title" style="margin-top:16px">Score Predictions</div>
                    <div class="form-row">
                        <label>Match Score Prediction</label>
                        <input type="number" step="0.1" name="match_score_pred_away" value="0.9" required>
                    </div>
                    <div class="form-row">
                        <label>Predicted Score</label>
                        <input type="number" step="1" name="pred_score_away" value="1" required>
                    </div>
                </div>
            </div>

            <!-- Prediction Selectors -->
            <div class="predictions-card">
                <div class="section-title">Pre-match Predictions</div>
                <div class="select-row">
                    <label>Predicted Result</label>
                    <select name="pred_result">
                        <option value="Home Win" selected>Home Win</option>
                        <option value="Draw">Draw</option>
                        <option value="Away Win">Away Win</option>
                    </select>
                </div>
                <div class="select-row">
                    <label>Predicted Over/Under 2.5</label>
                    <select name="pred_over_under">
                        <option value="Over" selected>Over 2.5</option>
                        <option value="Under">Under 2.5</option>
                    </select>
                </div>
                <div class="select-row">
                    <label>Predicted BTTS</label>
                    <select name="pred_btts">
                        <option value="Yes" selected>Yes</option>
                        <option value="No">No</option>
                    </select>
                </div>
            </div>

            <!-- Optional Avg XG -->
            <div class="predictions-card">
                <button type="button" class="optional-toggle" onclick="toggleOptional()">
                    + Avg XG Statistics (optional - improves accuracy)
                </button>
                <div id="optionalFields" class="optional-fields">
                    <div class="form-grid" style="margin-bottom:0">
                        <div>
                            <div class="form-row">
                                <label>Avg XG Scored (Home)</label>
                                <input type="number" step="0.1" name="avg_xg_scored_home" placeholder="auto">
                            </div>
                            <div class="form-row">
                                <label>Avg XG Conceded (Home)</label>
                                <input type="number" step="0.1" name="avg_xg_conceded_home" placeholder="auto">
                            </div>
                        </div>
                        <div>
                            <div class="form-row">
                                <label>Avg XG Scored (Away)</label>
                                <input type="number" step="0.1" name="avg_xg_scored_away" placeholder="auto">
                            </div>
                            <div class="form-row">
                                <label>Avg XG Conceded (Away)</label>
                                <input type="number" step="0.1" name="avg_xg_conceded_away" placeholder="auto">
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <button type="submit" class="btn-predict" id="btnPredict">
                PREDICT
            </button>
        </form>

        <!-- Loading -->
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing match data...</p>
        </div>

        <!-- Results -->
        <div id="results">
            <div class="results-header">
                <h2 id="matchTitle">Chelsea vs Fulham</h2>
            </div>

            <div class="results-grid">
                <!-- Home Win -->
                <div class="result-item" id="res_hw">
                    <div class="result-name">Home Win</div>
                    <div class="result-prediction" id="hw_pred">-</div>
                    <div class="probability-bar">
                        <div class="probability-fill" id="hw_bar" style="width:0%;background:#3b82f6"></div>
                    </div>
                    <div class="prob-value" id="hw_prob">0%</div>
                    <div><span class="confidence-badge" id="hw_conf">-</span></div>
                </div>

                <!-- Over 2.5 -->
                <div class="result-item" id="res_ou">
                    <div class="result-name">Total Over 2.5</div>
                    <div class="result-prediction" id="ou_pred">-</div>
                    <div class="probability-bar">
                        <div class="probability-fill" id="ou_bar" style="width:0%;background:#8b5cf6"></div>
                    </div>
                    <div class="prob-value" id="ou_prob">0%</div>
                    <div><span class="confidence-badge" id="ou_conf">-</span></div>
                </div>

                <!-- BTTS -->
                <div class="result-item" id="res_btts">
                    <div class="result-name">Both Teams Score</div>
                    <div class="result-prediction" id="btts_pred">-</div>
                    <div class="probability-bar">
                        <div class="probability-fill" id="btts_bar" style="width:0%;background:#f59e0b"></div>
                    </div>
                    <div class="prob-value" id="btts_prob">0%</div>
                    <div><span class="confidence-badge" id="btts_conf">-</span></div>
                </div>
            </div>

            <div class="recommendation" id="recommendation">
                <div class="rec-title">Recommendation</div>
                <div class="rec-items" id="recItems"></div>
            </div>
        </div>
    </div>

    <script>
        function toggleOptional() {
            const el = document.getElementById('optionalFields');
            el.classList.toggle('show');
        }

        function getConfClass(conf) {
            const map = {
                'HIGH': 'confidence-high',
                'MEDIUM': 'confidence-medium',
                'LOW': 'confidence-low',
                'NOT RECOMMENDED': 'confidence-none',
            };
            return map[conf] || 'confidence-none';
        }

        function getProbColor(prob) {
            if (prob >= 0.7) return '#22c55e';
            if (prob >= 0.55) return '#fbbf24';
            return '#ef4444';
        }

        document.getElementById('predForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const btn = document.getElementById('btnPredict');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');

            btn.disabled = true;
            loading.style.display = 'block';
            results.style.display = 'none';

            const formData = new FormData(this);
            const data = {};
            formData.forEach((val, key) => {
                if (val !== '' && val !== null) {
                    data[key] = isNaN(val) ? val : parseFloat(val);
                }
            });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.error) {
                    alert('Error: ' + result.error);
                    return;
                }

                // Update match title
                document.getElementById('matchTitle').textContent =
                    (data.team_home || 'Home') + '  vs  ' + (data.team_away || 'Away');

                // Home Win
                const hw = result.home_win;
                document.getElementById('hw_pred').textContent = hw.prediction;
                document.getElementById('hw_pred').className = 'result-prediction ' + (hw.prediction === 'YES' ? 'yes' : 'no');
                document.getElementById('hw_prob').textContent = (hw.probability * 100).toFixed(1) + '%';
                document.getElementById('hw_bar').style.width = (hw.probability * 100) + '%';
                document.getElementById('hw_bar').style.background = getProbColor(hw.probability);
                document.getElementById('hw_conf').textContent = hw.confidence;
                document.getElementById('hw_conf').className = 'confidence-badge ' + getConfClass(hw.confidence);

                // Over 2.5
                const ou = result.over25;
                document.getElementById('ou_pred').textContent = ou.prediction;
                document.getElementById('ou_pred').className = 'result-prediction ' + (ou.prediction === 'YES' ? 'yes' : 'no');
                document.getElementById('ou_prob').textContent = (ou.probability * 100).toFixed(1) + '%';
                document.getElementById('ou_bar').style.width = (ou.probability * 100) + '%';
                document.getElementById('ou_bar').style.background = getProbColor(ou.probability);
                document.getElementById('ou_conf').textContent = ou.confidence;
                document.getElementById('ou_conf').className = 'confidence-badge ' + getConfClass(ou.confidence);

                // BTTS
                const bt = result.btts;
                document.getElementById('btts_pred').textContent = bt.prediction;
                document.getElementById('btts_pred').className = 'result-prediction ' + (bt.prediction === 'YES' ? 'yes' : 'no');
                document.getElementById('btts_prob').textContent = (bt.probability * 100).toFixed(1) + '%';
                document.getElementById('btts_bar').style.width = (bt.probability * 100) + '%';
                document.getElementById('btts_bar').style.background = getProbColor(bt.probability);
                document.getElementById('btts_conf').textContent = bt.confidence;
                document.getElementById('btts_conf').className = 'confidence-badge ' + getConfClass(bt.confidence);

                // Recommendations
                const recItems = document.getElementById('recItems');
                recItems.innerHTML = '';

                const markets = [
                    { name: 'Home Win', data: hw, threshold: ['HIGH', 'MEDIUM'] },
                    { name: 'Over 2.5', data: ou, threshold: ['HIGH', 'MEDIUM'] },
                    { name: 'BTTS', data: bt, threshold: ['HIGH', 'MEDIUM'] },
                ];

                let hasRec = false;
                markets.forEach(m => {
                    const isGood = m.threshold.includes(m.data.confidence);
                    const div = document.createElement('div');
                    div.className = 'rec-item ' + (isGood ? 'positive' : 'negative');
                    const icon = isGood ? '&#10003;' : '&#10007;';
                    div.innerHTML = icon + ' ' + m.name + ' (' + (m.data.probability * 100).toFixed(0) + '%)';
                    recItems.appendChild(div);
                    if (isGood) hasRec = true;
                });

                if (!hasRec) {
                    const div = document.createElement('div');
                    div.className = 'rec-item negative';
                    div.textContent = 'No confident predictions for this match';
                    recItems.innerHTML = '';
                    recItems.appendChild(div);
                }

                loading.style.display = 'none';
                results.style.display = 'block';
                results.scrollIntoView({ behavior: 'smooth', block: 'start' });

            } catch (err) {
                alert('Connection error: ' + err.message);
            } finally {
                btn.disabled = false;
                loading.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        match_data = {
            'team_home': data.get('team_home', 'Home'),
            'team_away': data.get('team_away', 'Away'),
            'team_rating_home': float(data['team_rating_home']),
            'team_rating_away': float(data['team_rating_away']),
            'team_form_home': float(data['team_form_home']),
            'team_form_away': float(data['team_form_away']),
            'xg_luckiness_home': float(data['xg_luckiness_home']),
            'xg_luckiness_away': float(data['xg_luckiness_away']),
            'xg_predictability_home': float(data['xg_predictability_home']),
            'xg_predictability_away': float(data['xg_predictability_away']),
            'match_score_pred_home': float(data['match_score_pred_home']),
            'match_score_pred_away': float(data['match_score_pred_away']),
            'pred_score_home': float(data['pred_score_home']),
            'pred_score_away': float(data['pred_score_away']),
            'pred_result': data['pred_result'],
            'pred_over_under': data['pred_over_under'],
            'pred_btts': data['pred_btts'],
        }

        # Optional Avg XG fields
        for field in ['avg_xg_scored_home', 'avg_xg_scored_away',
                      'avg_xg_conceded_home', 'avg_xg_conceded_away']:
            if field in data and data[field] not in (None, '', 'NaN'):
                match_data[field] = float(data[field])

        results = predict_match(match_data)

        # Translate confidence for frontend
        conf_map = {
            'ВЫСОКАЯ': 'HIGH',
            'СРЕДНЯЯ': 'MEDIUM',
            'НИЗКАЯ': 'LOW',
            'НЕ РЕКОМЕНДУЕТСЯ': 'NOT RECOMMENDED',
        }
        pred_map = {'ДА': 'YES', 'НЕТ': 'NO'}

        response = {}
        for key in ['home_win', 'over25', 'btts']:
            r = results[key]
            response[key] = {
                'probability': round(float(r['probability']), 4),
                'prediction': pred_map.get(r['prediction'], r['prediction']),
                'confidence': conf_map.get(r['confidence'], r['confidence']),
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("\n  Football Prediction System - Web Interface")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
