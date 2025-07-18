import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from flask import Flask, render_template, jsonify, request
from binance import ThreadedWebsocketManager

app = Flask(__name__)

# เก็บข้อมูลราคาสำหรับกราฟ
price_data = []

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

# ฟังก์ชันคำนวณความผันผวน
def calculate_volatility(data):
    returns = data['close'].pct_change().dropna()
    volatility = returns.std() * 100
    return volatility

# ฟังก์ชันวิเคราะห์สัญญาณและทำนายราคา
def trading_signal(data, random_factor=0.1):
    latest = data.iloc[-1]
    rsi = latest['RSI']
    ma10 = latest['MA10']
    ma50 = latest['MA50']
    current_price = latest['close']
    
    rsi_signal = None
    ma_signal = None
    rsi_success_rate = 0
    ma_success_rate = 0
    
    if rsi < 30:
        rsi_signal = 'Long'
        rsi_success_rate = 75
    elif rsi > 70:
        rsi_signal = 'Short'
        rsi_success_rate = 70
    else:
        rsi_signal = 'Neutral'
        rsi_success_rate = 50
    
    if ma10 > ma50 and data.iloc[-2]['MA10'] <= data.iloc[-2]['MA50']:
        ma_signal = 'Long'
        ma_success_rate = 65
    elif ma10 < ma50 and data.iloc[-2]['MA10'] >= data.iloc[-2]['MA50']:
        ma_signal = 'Short'
        ma_success_rate = 60
    else:
        ma_signal = 'Neutral'
        ma_success_rate = 50
    
    if rsi_signal == ma_signal and rsi_signal != 'Neutral':
        signal = rsi_signal
        success_rate = (rsi_success_rate * 0.4 + ma_success_rate * 0.6)
    elif rsi_signal != 'Neutral':
        signal = rsi_signal
        success_rate = rsi_success_rate * 0.7
    elif ma_signal != 'Neutral':
        signal = ma_signal
        success_rate = ma_success_rate * 0.7
    else:
        signal = 'Neutral'
        success_rate = 50
    
    random_adjustment = np.random.uniform(-random_factor * success_rate, random_factor * success_rate)
    success_rate = min(max(success_rate + random_adjustment, 0), 100)
    
    if signal == 'Neutral':
        signal = np.random.choice(['Long', 'Short'])
        success_rate = 50 + random_adjustment
    
    volatility = calculate_volatility(data)
    random_factor_price = np.random.uniform(0.5, 1.5)
    if signal == 'Long':
        predicted_price = current_price * (1 + (volatility / 100) * random_factor_price)
    else:
        predicted_price = current_price * (1 - (volatility / 100) * random_factor_price)
    
    return signal, success_rate, rsi, ma10, ma50, predicted_price

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
        data = data.dropna()
        
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# บันทึกการทาย
def save_prediction(timestamp, price, rsi, ma10, ma50, signal, success_rate, predicted_price):
    data = {
        'timestamp': [timestamp],
        'price': [price],
        'RSI': [rsi],
        'MA10': [ma10],
        'MA50': [ma50],
        'signal': [signal],
        'success_rate': [success_rate],
        'predicted_price': [predicted_price]
    }
    df = pd.DataFrame(data)
    file_exists = os.path.isfile('predictions.csv')
    df.to_csv('predictions.csv', mode='a', header=not file_exists, index=False)

# ตรวจสอบการทายย้อนหลัง
def check_past_predictions(current_price):
    if not os.path.isfile('predictions.csv'):
        return []
    
    try:
        df = pd.read_csv('predictions.csv')
    except pd.errors.ParserError:
        df = pd.read_csv('predictions.csv', names=['timestamp', 'price', 'RSI', 'MA10', 'MA50', 'signal', 'success_rate', 'predicted_price'], skiprows=1)
    
    if 'predicted_price' not in df.columns:
        df['predicted_price'] = np.nan
    
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
        predicted_price = row.get('predicted_price', np.nan)
        
        is_correct = False
        if signal == 'Long' and current_price > pred_price:
            is_correct = True
        elif signal == 'Short' and current_price < pred_price:
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
            'predicted_price': predicted_price if not np.isnan(predicted_price) else 'N/A'
        })
    
    return results

# วิเคราะห์สถิติ
def analyze_statistics():
    if not os.path.isfile('predictions.csv'):
        return {'total': 0, 'long_count': 0, 'short_count': 0, 'avg_success_rate': 0, 'recent': []}
    
    try:
        df = pd.read_csv('predictions.csv')
    except pd.errors.ParserError:
        df = pd.read_csv('predictions.csv', names=['timestamp', 'price', 'RSI', 'MA10', 'MA50', 'signal', 'success_rate', 'predicted_price'], skiprows=1)
    
    if 'predicted_price' not in df.columns:
        df['predicted_price'] = np.nan
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    total = len(df)
    long_count = len(df[df['signal'] == 'Long'])
    short_count = len(df[df['signal'] == 'Short'])
    avg_success_rate = df['success_rate'].mean() if total > 0 else 0
    recent = df.tail(5)[['timestamp', 'price', 'signal', 'success_rate', 'predicted_price']].to_dict('records')
    
    for pred in recent:
        pred['predicted_price'] = pred['predicted_price'] if not np.isnan(pred['predicted_price']) else 'N/A'
    
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
    data = {
        'timestamp': [timestamp],
        'username': [username],
        'message': [message]
    }
    df = pd.DataFrame(data)
    file_exists = os.path.isfile('messages.csv')
    df.to_csv('messages.csv', mode='a', header=not file_exists, index=False)

# อ่านข้อความแชท
def get_messages():
    if not os.path.isfile('messages.csv'):
        return []
    
    df = pd.read_csv('messages.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    messages = df.tail(20)[['timestamp', 'username', 'message']].to_dict('records')
    
    # สร้างสีสำหรับชื่อผู้ใช้
    for msg in messages:
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

# ส่งข้อความ
@app.route('/send_message', methods=['POST'])
def send_message():
    username = request.form.get('username', 'Anonymous')
    message = request.form.get('message', '')
    if username and message:
        save_message(username, message)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Username and message required'}), 400

@app.route('/update')
def update():
    data = fetch_realtime_data()
    if data is None:
        return {"error": "Failed to fetch data"}, 500
    latest_price = data['close'].iloc[-1]
    for i in range(4):
        signal, success_rate, rsi, ma10, ma50, predicted_price = trading_signal(data)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        save_prediction(timestamp, latest_price, rsi, ma10, ma50, signal, success_rate, predicted_price)
    return {"status": "Predictions updated"}, 200

# เส้นทางหน้าเว็บหลัก
@app.route('/')
def index():
    data = fetch_realtime_data()
    if data is None:
        return render_template('index.html', error="ไม่สามารถดึงข้อมูลราคาได้")
    
    latest_price = data['close'].iloc[-1]
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # ทำนาย 4 ครั้ง
    predictions = []
    for i in range(4):
        signal, success_rate, rsi, ma10, ma50, predicted_price = trading_signal(data, random_factor=0.1)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        predictions.append({
            'round': i + 1,
            'signal': signal,
            'success_rate': success_rate,
            'rsi': rsi,
            'ma10': ma10,
            'ma50': ma50,
            'predicted_price': predicted_price
        })
        save_prediction(timestamp, latest_price, rsi, ma10, ma50, signal, success_rate, predicted_price)
    
    # ตรวจสอบการทายย้อนหลัง
    past_predictions = check_past_predictions(latest_price)
    
    # วิเคราะห์สถิติ
    stats = analyze_statistics()
    
    # ดึงข้อความแชท
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

if __name__ == "__main__":
    # เริ่ม WebSocket
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
    
    # รัน Flask
    app.run(debug=True)
    
    # หยุด WebSocket
    twm.stop()