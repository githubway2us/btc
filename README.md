# BTC/USDT Trade Predictor

## ภาษาไทย

### ภาพรวมโปรเจกต์

**BTC/USDT Trade Predictor** เป็นแอปพลิเคชันเว็บที่พัฒนาด้วย **Flask** และ **Python** เพื่อวิเคราะห์และทำนายทิศทางการซื้อขายของคู่เงิน **BTC/USDT** ในกรอบเวลา 4 ชั่วโมง โดยใช้ตัวชี้วัดทางเทคนิค เช่น RSI, Moving Averages (MA10, MA50), Bollinger Bands และ ATR รวมถึงโมเดล Machine Learning (Random Forest) เพื่อเพิ่มความแม่นยำในการทำนาย แอปพลิเคชันนี้มีระบบแชทเรียลไทม์ การแสดงกราฟราคาด้วย Chart.js และการจัดการความเสี่ยงผ่านการคำนวณ Stop-Loss, Take-Profit และ Position Sizing

### คุณสมบัติหลัก
- **การทำนายทิศทางราคา**: ใช้ตัวชี้วัดทางเทคนิค (RSI, MA10, MA50, Bollinger Bands, ATR) และ Random Forest เพื่อทำนายสัญญาณ Long/Short
- **การจัดการความเสี่ยง**: คำนวณ Stop-Loss, Take-Profit และขนาดตำแหน่ง (Position Sizing) ตาม ATR
- **ข้อมูลเรียลไทม์**: ดึงข้อมูลราคา BTC/USDT จาก Binance ผ่าน WebSocket และ ccxt
- **กราฟราคา**: แสดงกราฟราคาเรียลไทม์พร้อม MA10 และ MA50 โดยใช้ Chart.js
- **ระบบแชท**: รองรับการแชทแบบเรียลไทม์สำหรับผู้ใช้ พร้อมสีประจำตัวผู้ใช้
- **ฐานข้อมูล SQLite**: บันทึกการทำนายและข้อความแชทเพื่อการวิเคราะห์ย้อนหลัง
- **ความปลอดภัย**: ใช้ Rate Limiting และการป้องกัน XSS ด้วย `html.escape`
- **Dark Mode**: รองรับการสลับธีม Light/Dark ตามการตั้งค่าผู้ใช้หรือระบบ

### ความต้องการของระบบ
- **Python**: 3.8 หรือสูงกว่า
- **Dependencies**:
  - `ccxt`
  - `pandas`
  - `numpy`
  - `flask`
  - `python-binance`
  - `scikit-learn`
  - `flask-limiter`
- **เบราว์เซอร์**: รองรับ JavaScript (เช่น Chrome, Firefox, Edge)
- **การเชื่อมต่ออินเทอร์เน็ต**: เพื่อดึงข้อมูลจาก Binance API

### การติดตั้ง
1. **โคลนโปรเจกต์**:
   ```bash
   git clone <repository-url>
   cd btc-usdt-trade-predictor
