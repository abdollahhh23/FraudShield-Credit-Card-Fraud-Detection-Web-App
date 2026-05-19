from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

AMOUNT_MEAN = 88.35
AMOUNT_STD  = 250.12

MERCHANT_LABELS = {
    'grocery':       'Grocery / Supermarket',
    'online':        'Online Retail',
    'travel':        'Travel & Airlines',
    'entertainment': 'Entertainment / Gaming',
    'atm':           'ATM / Cash Withdrawal',
    'luxury':        'Luxury / Jewelry',
    'utility':       'Utility / Bill Payment',
}

PATTERNS = {
    ('grocery', 'day'):      [1.2, 0.1, 0.4, -0.1, 0.5, -0.1, 0.2, 0.1, 0.0, 0.0,
                               0.1, -0.1, 0.2, 0.0, -0.1, 0.1, 0.0, 0.0, 0.1, 0.0,
                               0.0, 0.1, -0.1, 0.0, 0.0, 0.1, 0.0, -0.1],
    ('grocery', 'night'):    [0.8, 0.3, 0.2, 0.1, 0.3, 0.1, 0.0, 0.1, 0.0, 0.1,
                               0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
    ('online', 'day'):       [0.5, -0.2, 0.8, 0.1, -0.3, 0.4, -0.1, 0.2, 0.0, -0.1,
                               0.3, 0.0, -0.2, 0.1, 0.0, 0.2, -0.1, 0.0, 0.1, 0.0,
                               -0.1, 0.2, 0.0, -0.1, 0.1, 0.0, 0.0, 0.1],
    ('online', 'night'):     [-1.5, 1.2, -2.0, 1.5, -0.8, -0.5, -1.8, 0.7, -1.0, -1.2,
                               1.5, -2.0, 0.8, -2.2, 0.2, -1.5, -2.5, -0.9, 0.5, 0.1,
                               0.3, -0.2, -0.1, 0.1, -0.2, -0.3, 0.4, -0.1],
    ('travel', 'day'):       [0.3, 0.5, 0.1, -0.3, 0.8, 0.2, -0.1, 0.4, 0.1, 0.0,
                               -0.2, 0.3, 0.0, 0.1, -0.1, 0.2, 0.1, 0.0, -0.1, 0.1,
                               0.1, -0.1, 0.2, 0.0, 0.1, -0.1, 0.0, 0.1],
    ('travel', 'night'):     [-0.8, 0.6, -1.2, 0.9, -0.5, -0.3, -1.0, 0.4, -0.6, -0.7,
                               0.9, -1.1, 0.5, -1.3, 0.1, -0.8, -1.4, -0.5, 0.3, 0.1,
                               0.1, -0.1, -0.1, 0.0, -0.1, -0.1, 0.2, -0.1],
    ('entertainment', 'day'):[0.6, 0.2, 0.5, -0.2, 0.4, 0.1, 0.0, 0.2, 0.1, 0.0,
                               0.1, 0.2, -0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1,
                               0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
    ('entertainment', 'night'): [-2.0, 1.8, -3.0, 2.5, -1.5, -1.0, -3.2, 1.2, -1.8, -2.2,
                               2.8, -3.5, 1.4, -4.0, 0.3, -2.8, -4.5, -1.6, 1.0, 0.2,
                               0.5, -0.3, -0.2, 0.1, -0.2, -0.4, 0.6, -0.2],
    ('atm', 'day'):          [0.2, 0.8, -0.1, 0.4, 0.6, 0.3, 0.1, 0.5, 0.2, 0.1,
                               -0.3, 0.4, 0.1, 0.2, -0.1, 0.3, 0.2, 0.1, -0.1, 0.1,
                               0.1, 0.0, 0.2, 0.0, 0.1, -0.1, 0.1, 0.0],
    ('atm', 'night'):        [-2.5, 2.0, -3.5, 3.0, -2.0, -1.5, -3.8, 1.6, -2.2, -2.8,
                               3.2, -4.0, 1.6, -4.8, 0.4, -3.2, -5.5, -2.0, 1.2, 0.3,
                               0.6, -0.4, -0.2, 0.1, -0.3, -0.5, 0.7, -0.2],
    ('luxury', 'day'):       [0.1, 0.3, 0.2, 0.1, 0.7, 0.1, 0.2, 0.3, 0.0, 0.1,
                               -0.1, 0.2, 0.1, 0.0, -0.1, 0.1, 0.0, 0.1, 0.0, 0.0,
                               0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
    ('luxury', 'night'):     [-3.0, 2.5, -4.0, 3.8, -2.5, -1.8, -4.5, 2.0, -2.8, -3.5,
                               4.0, -5.2, 2.1, -6.0, 0.5, -4.2, -7.0, -2.5, 1.5, 0.4,
                               0.8, -0.5, -0.3, 0.2, -0.4, -0.6, 0.9, -0.3],
    ('foreign', 'day'):      [-0.5, 0.4, -0.8, 0.6, -0.3, -0.2, -0.6, 0.3, -0.4, -0.5,
                               0.6, -0.7, 0.3, -0.8, 0.1, -0.5, -0.9, -0.3, 0.2, 0.1,
                               0.1, -0.1, -0.1, 0.0, -0.1, -0.1, 0.2, -0.1],
    ('foreign', 'night'):    [-3.5, 3.0, -4.5, 4.2, -3.0, -2.2, -5.0, 2.3, -3.2, -4.0,
                               4.5, -5.8, 2.4, -6.8, 0.6, -4.8, -8.0, -2.8, 1.7, 0.5,
                               0.9, -0.6, -0.4, 0.2, -0.5, -0.7, 1.0, -0.4],
    ('utility', 'day'):      [1.5, 0.0, 0.6, -0.2, 0.3, -0.2, 0.3, 0.0, 0.1, 0.0,
                               0.2, -0.1, 0.1, 0.0, -0.1, 0.0, 0.1, 0.0, 0.0, 0.0,
                               0.0, 0.1, -0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
    ('utility', 'night'):    [0.5, 0.2, 0.1, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

# Risk score weights — used for the breakdown chart sent to the frontend
RISK_WEIGHTS = {
    'amount':    {'label': 'Amount',          'max': 40},
    'merchant':  {'label': 'Merchant Type',   'max': 25},
    'location':  {'label': 'Location',        'max': 20},
    'time':      {'label': 'Time of Day',     'max': 15},
}

MERCHANT_BASE_RISK = {
    'grocery': 2, 'utility': 3, 'entertainment': 8, 'online': 10,
    'travel': 12, 'atm': 18, 'luxury': 15,
}

def _risk_breakdown(amount, merchant, location, time_of_day, fraud_prob):
    # Amount risk: scales with transaction size, capped at 40
    amt_risk = min(40, round((amount / 2000) * 40, 1))

    # Merchant risk: 0–25
    merch_risk = round(min(25, MERCHANT_BASE_RISK.get(merchant, 8) / 18 * 25), 1)

    # Location risk: local=2, international=16
    loc_risk = 16.0 if location == 'foreign' else 2.0

    # Time risk: day=2, night=9
    time_risk = 9.0 if time_of_day == 'night' else 2.0

    total = amt_risk + merch_risk + loc_risk + time_risk

    return {
        'amount':   amt_risk,
        'merchant': merch_risk,
        'location': loc_risk,
        'time':     time_risk,
        'total':    round(total, 1),
        'max':      100,
    }


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data        = request.get_json()
        amount      = float(data['amount'])
        merchant    = data['merchant']
        time_of_day = data['time_of_day']
        location    = data['location']

        # Foreign transactions use their own pattern set
        if location == 'foreign':
            merchant = 'foreign'

        key = (merchant, time_of_day)
        if key not in PATTERNS:
            key = ('grocery', 'day')

        v_features = PATTERNS[key]

        normalized_amount = (amount - AMOUNT_MEAN) / AMOUNT_STD
        features = v_features + [normalized_amount]

        prediction = int(model.predict([np.array(features)])[0])

        try:
            prob       = model.predict_proba([np.array(features)])[0]
            fraud_prob = round(float(prob[1]) * 100, 1)
        except:
            fraud_prob = 75.0 if prediction == 1 else 8.0

        breakdown = _risk_breakdown(amount, merchant, location, time_of_day, fraud_prob)

        # Plain, non-AI-sounding flags
        flags = []
        if location == 'foreign':
            flags.append('Card used outside home country')
        if time_of_day == 'night':
            flags.append('Transaction outside business hours')
        if merchant == 'atm':
            flags.append('Cash withdrawal detected')
        if merchant == 'luxury':
            flags.append('High-value goods category')
        if amount > 500:
            flags.append(f'Amount exceeds $500 threshold')
        if amount > 1000:
            flags.append(f'Amount exceeds $1,000 threshold')
        if not flags:
            flags.append('No unusual indicators')

        return jsonify({
            'prediction':      prediction,
            'fraud_prob':      fraud_prob,
            'breakdown':       breakdown,
            'flags':           flags,
            'merchant_label':  MERCHANT_LABELS.get(merchant, merchant),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
