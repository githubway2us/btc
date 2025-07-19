import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from flask import Flask, render_template, jsonify, request
from binance import ThreadedWebsocketManager
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from html import escape

app = Flask(__name__)
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
limiter.init_app(app)
price_data = []
ml_model = None

# ฟังก์ชันคำนวณ RSI
def calculate_rsi(data, periods=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ฟังก์ชันคำนวณ Moving Averages
def calculate_ma(data, short_period=10, long_period=50):
    ma_short = data['close'].rolling(window=short_period).mean()
    ma_long = data['close'].rolling(window=long_period).mean()
    return ma_short, ma_long

# ฟังก์ชันคำนวณ Bollinger Bands
def calculate_bollinger_bands(data, period=20, std_dev=2):
    sma = data['close'].rolling(window=period).mean()
    rolling_std = data['close'].rolling(window=period).std()
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    return sma, upper_band, lower_band

# ฟังก์ชันคำนวณ ATR
def calculate_atr(data, period=14):
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

# ฟังก์ชันคำนวณความผันผวน
def calculate_volatility(data):
    returns = data['close'].pct_change().dropna()
    volatility = returns.std() * 100
    return volatility

# ฟังก์ชันคำนวณขนาดตำแหน่ง
def calculate_position_size(account_balance, risk_percentage, stop_loss_price, entry_price):
    risk_per_trade = account_balance * (risk_percentage / 100)
    risk_per_unit = abs(entry_price - stop_loss_price)
    position_size = risk_per_trade / risk_per_unit if risk_per_unit != 0 else 0
    return position_size

# ฟังก์ชันฝึกโมเดล Machine Learning
def train_ml_model(data):
    features = ['RSI', 'MA10', 'MA50', 'volume', 'Upper_BB', 'Lower_BB', 'ATR']
    target = (data['close'].shift(-1) > data['close']).astype(int)  # 1 = Long, 0 = Short
    
    X = data[features].dropna()
    y = target[X.index]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {accuracy:.2f}")
    
    return model

# ฟังก์ชันทำนายด้วย ML
def predict_with_ml(model, latest_data):
    features = ['RSI', 'MA10', 'MA50', 'volume', 'Upper_BB', 'Lower_BB', 'ATR']
    X = latest_data[features].iloc[-1:].values
    prediction = model.predict(X)[0]
    return 'Long' if prediction == 1 else 'Short'

# ฟังก์ชันวิเคราะห์สัญญาณและทำนายราคา
def trading_signal(data, random_factor=0.05):
    latest = data.iloc[-1]
    rsi = latest['RSI']
    ma10 = latest['MA10']
    ma50 = latest['MA50']
    current_price = latest['close']
    upper_band = latest['Upper_BB']
    lower_band = latest['Lower_BB']
    atr = latest['ATR']
    volume = latest['volume']
    avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]

    rsi_signal = None
    ma_signal = None
    bb_signal = None
    volume_signal = None
    rsi_success_rate = 0
    ma_success_rate = 0
    bb_success_rate = 0
    volume_success_rate = 0

    # RSI
    if rsi < 30:
        rsi_signal = 'Long'
        rsi_success_rate = 75
    elif rsi > 70:
        rsi_signal = 'Short'
        rsi_success_rate = 70
    else:
        rsi_signal = 'Neutral'
        rsi_success_rate = 50

    # Moving Averages
    if ma10 > ma50 and data.iloc[-2]['MA10'] <= data.iloc[-2]['MA50']:
        ma_signal = 'Long'
        ma_success_rate = 65
    elif ma10 < ma50 and data.iloc[-2]['MA10'] >= data.iloc[-2]['MA50']:
        ma_signal = 'Short'
        ma_success_rate = 60
    else:
        ma_signal = 'Neutral'
        ma_success_rate = 50

    # Bollinger Bands
    if current_price < lower_band:
        bb_signal = 'Long'
        bb_success_rate = 70
    elif current_price > upper_band:
        bb_signal = 'Short'
        bb_success_rate = 65
    else:
        bb_signal = 'Neutral'
        bb_success_rate = 50

    # Volume Analysis
    if volume > avg_volume * 1.5:
        volume_signal = 'Strong'
        volume_success_rate = 60
    else:
        volume_signal = 'Neutral'
        volume_success_rate = 50

    # รวมสัญญาณ
    signals = [rsi_signal, ma_signal, bb_signal]
    valid_signals = [s for s in signals if s != 'Neutral']
    if len(set(valid_signals)) == 1 and valid_signals:
        signal = valid_signals[0]
        success_rate = (rsi_success_rate * 0.3 + ma_success_rate * 0.3 + bb_success_rate * 0.3 + volume_success_rate * 0.1)
    elif valid_signals:
        signal = max(set(valid_signals), key=valid_signals.count)
        success_rate = (rsi_success_rate * 0.4 + ma_success_rate * 0.4 + bb_success_rate * 0.2) if signal != 'Neutral' else 50
    else:
        signal = np.random.choice(['Long', 'Short'])
        success_rate = 50

    # ปรับความผันผวนด้วย ATR
    random_adjustment = np.random.uniform(-random_factor * success_rate, random_factor * success_rate)
    success_rate = min(max(success_rate + random_adjustment, 0), 100)

    # ทำนายราคาด้วย ATR
    if signal == 'Long':
        predicted_price = current_price + (atr * 1.5)
        stop_loss = current_price - (atr * 1)
        take_profit = current_price + (atr * 2)
    else:
        predicted_price = current_price - (atr * 1.5)
        stop_loss = current_price + (atr * 1)
        take_profit = current_price - (atr * 2)

    return signal, success_rate, rsi, ma10, ma50, predicted_price, stop_loss, take_profit

# ดึงข้อมูล OHLCV จาก Binance
def fetch_realtime_data():
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=100)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        
        data['RSI'] = calculate_rsi(data)
        data['MA10'], data['MA50'] = calculate_ma(data)
        data['SMA20'], data['Upper_BB'], data['Lower_BB'] = calculate_bollinger_bands(data)
        data['ATR'] = calculate_atr(data)
        data = data.dropna()
        
        global ml_model
        if ml_model is None:
            ml_model = train_ml_model(data)
        
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# เริ่มต้นฐานข้อมูล SQLite
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (timestamp TEXT, price REAL, rsi REAL, ma10 REAL, ma50 REAL, signal TEXT, 
                 success_rate REAL, predicted_price REAL, stop_loss REAL, take_profit REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (timestamp TEXT, username TEXT, message TEXT)''')
    conn.commit()
    conn.close()

# บันทึกการทาย
def save_prediction(timestamp, price, rsi, ma10, ma50, signal, success_rate, predicted_price, stop_loss, take_profit):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (timestamp, price, rsi, ma10, ma50, signal, success_rate, predicted_price, stop_loss, take_profit))
    conn.commit()
    conn.close()

# ตรวจสอบการทายย้อนหลัง
def check_past_predictions(current_price):
    conn = sqlite3.connect('predictions.db')
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    four_hours_ago = datetime.now() - timedelta(hours=4)
    tolerance = timedelta(minutes=10)
    past_predictions = df[
        (df['timestamp'] >= four_hours_ago - tolerance) &
        (df['timestamp'] <= four_hours_ago + tolerance)
    ]
    
    results = []
    for _, row in past_predictions.iterrows():
        pred_price = row['price']
        signal = row['signal']
        timestamp = row['timestamp']
        predicted_price = row['predicted_price']
        stop_loss = row['stop_loss']
        take_profit = row['take_profit']
        
        is_correct = False
        hit_stop_loss = False
        hit_take_profit = False
        
        if signal == 'Long':
            if current_price >= take_profit:
                hit_take_profit = True
                is_correct = True
            elif current_price <= stop_loss:
                hit_stop_loss = True
                is_correct = False
            elif current_price > pred_price:
                is_correct = True
        else:  # Short
            if current_price <= take_profit:
                hit_take_profit = True
                is_correct = True
            elif current_price >= stop_loss:
                hit_stop_loss = True
                is_correct = False
            elif current_price < pred_price:
                is_correct = True
        
        price_diff = current_price - pred_price
        price_diff_percent = (price_diff / pred_price) * 100
        
        results.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'signal': signal,
            'pred_price': pred_price,
            'current_price': current_price,
            'price_diff': price_diff,
            'price_diff_percent': price_diff_percent,
            'is_correct': is_correct,
            'hit_stop_loss': hit_stop_loss,
            'hit_take_profit': hit_take_profit,
            'predicted_price': predicted_price
        })
    
    return results

# วิเคราะห์สถิติ
def analyze_statistics():
    conn = sqlite3.connect('predictions.db')
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    total = len(df)
    long_count = len(df[df['signal'] == 'Long'])
    short_count = len(df[df['signal'] == 'Short'])
    avg_success_rate = df['success_rate'].mean() if total > 0 else 0
    recent = df.tail(5)[['timestamp', 'price', 'signal', 'success_rate', 'predicted_price']].to_dict('records')
    
    for pred in recent:
        pred['predicted_price'] = pred['predicted_price'] if not pd.isna(pred['predicted_price']) else 'N/A'
    
    return {
        'total': total,
        'long_count': long_count,
        'short_count': short_count,
        'avg_success_rate': avg_success_rate,
        'recent': recent
    }

# บันทึกข้อความแชท
def save_message(username, message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''INSERT INTO messages VALUES (?, ?, ?)''', (timestamp, username, message))
    conn.commit()
    conn.close()

# อ่านข้อความแชท
def get_messages():
    conn = sqlite3.connect('predictions.db')
    df = pd.read_sql_query("SELECT * FROM messages ORDER BY timestamp DESC LIMIT 20", conn)
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    messages = df[['timestamp', 'username', 'message']].to_dict('records')
    
    for msg in messages:
        # แปลง timestamp เป็นสตริงในรูปแบบ HH:MM:SS
        msg['timestamp'] = msg['timestamp'].strftime('%H:%M:%S')
        msg['color'] = f"hsl({hash(msg['username']) % 360}, 70%, 70%)"
    
    return messages

# ฟังก์ชันแฮชสำหรับกำหนดสี
def hash(string):
    h = 0
    for c in string:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h

# ดึงข้อมูลสำหรับกราฟ
@app.route('/price_data')
def get_price_data():
    return jsonify(price_data[-50:])

# ดึงข้อความแชท
@app.route('/get_messages')
def get_messages_route():
    return jsonify(get_messages())

# ส่งข้อความ
@app.route('/send_message', methods=['POST'])
@limiter.limit("10 per minute")
def send_message():
    username = escape(request.form.get('username', 'Anonymous'))
    message = escape(request.form.get('message', ''))
    if username and message and len(username) <= 50 and len(message) <= 500:
        save_message(username, message)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Invalid input'}), 400

# อัปเดตการทำนาย
@app.route('/update')
def update():
    data = fetch_realtime_data()
    if data is None:
        return {"error": "Failed to fetch data"}, 500
    latest_price = data['close'].iloc[-1]
    for i in range(4):
        signal, success_rate, rsi, ma10, ma50, predicted_price, stop_loss, take_profit = trading_signal(data)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        save_prediction(timestamp, latest_price, rsi, ma10, ma50, signal, success_rate, predicted_price, stop_loss, take_profit)
    return {"status": "Predictions updated"}, 200

# เส้นทางหน้าเว็บหลัก
@app.route('/')
def index():
    data = fetch_realtime_data()
    if data is None:
        return render_template('index.html', error="ไม่สามารถดึงข้อมูลราคาได้")
    
    latest_price = data['close'].iloc[-1]
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    predictions = []
    for i in range(4):
        signal, success_rate, rsi, ma10, ma50, predicted_price, stop_loss, take_profit = trading_signal(data)
        ml_signal = predict_with_ml(ml_model, data)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        predictions.append({
            'round': i + 1,
            'signal': signal,
            'ml_signal': ml_signal,
            'success_rate': success_rate,
            'rsi': rsi,
            'ma10': ma10,
            'ma50': ma50,
            'predicted_price': predicted_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        })
        save_prediction(timestamp, latest_price, rsi, ma10, ma50, signal, success_rate, predicted_price, stop_loss, take_profit)
    
    past_predictions = check_past_predictions(latest_price)
    stats = analyze_statistics()
    messages = get_messages()
    
    return render_template(
        'index.html',
        latest_price=latest_price,
        current_time=current_time,
        predictions=predictions,
        past_predictions=past_predictions,
        stats=stats,
        messages=messages
    )

# เริ่มต้น WebSocket
def init_websocket():
    twm = ThreadedWebsocketManager()
    twm.start()
    
    def handle_socket_message(msg):
        if msg['e'] == 'kline':
            price = float(msg['k']['c'])
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            price_data.append({'timestamp': timestamp, 'price': price})
            if len(price_data) > 100:
                price_data.pop(0)
            print(f"WebSocket Price Update: ${price}")
    
    twm.start_kline_socket(callback=handle_socket_message, symbol='BTCUSDT', interval='1m')
    return twm

if __name__ == "__main__":
    init_db()
    twm = init_websocket()
    app.run(host='0.0.0.0', port=5000, debug=True)
    twm.stop()