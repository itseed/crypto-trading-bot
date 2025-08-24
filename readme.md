# Crypto Trading Bot (Python + CCXT) — Starter Kit

> **คำเตือน**: ตัวอย่างนี้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน ใช้โหมด **paper** ทดสอบก่อนเสมอ และเข้าใจความเสี่ยงของคริปโตที่ผันผวนสูง

---

## โครงสร้างโปรเจกต์
```
crypto-bot/
├─ bot.py                # main loop รันบอท
├─ strategy.py           # กลยุทธ์ (EMA cross + RSI filter)
├─ broker.py             # Paper broker + Live (ผ่าน CCXT)
├─ data.py               # ดึง/แปลง OHLCV + indicators
├─ backtest.py           # backtest ง่ายๆ จากไฟล์ OHLCV
├─ config.yaml           # การตั้งค่า (symbol, timeframe, risk ฯลฯ)
├─ .env.sample           # ตัวอย่างคีย์ API
├─ requirements.txt      # ไลบรารีที่ต้องใช้
├─ Dockerfile            # สร้างคอนเทนเนอร์
└─ docker-compose.yml    # รันด้วย compose + .env
```

---

## requirements.txt
```
ccxt>=4.0.0
pandas>=2.0.0
pandas_ta>=0.3.14b
python-dotenv>=1.0.0
PyYAML>=6.0.0
tenacity>=8.2.3
```

---

## .env.sample
```
# เลือก exchange ที่ ccxt รองรับ เช่น binance, bybit, okx (ค่า default ใช้ config.yaml กำหนด)
API_KEY=
API_SECRET=
# สำหรับบาง exchange (เช่น bybit) ถ้าใช้ testnet จะสลับ url ให้อัตโนมัติเมื่อ USE_TESTNET=true
USE_TESTNET=true
```

---

## config.yaml
```yaml
mode: paper  # paper | live
exchange: binance
symbol: BTC/USDT
timeframe: 15m
lookback_bars: 400
poll_interval_sec: 30

risk:
  equity: 10000          # ทุนจำลองสำหรับ paper mode (USDT)
  risk_per_trade: 0.01   # 1% ของ equity ต่อการเทรดหนึ่งครั้ง
  fee_rate: 0.001        # ค่าธรรมเนียมคร่าวๆ 0.1%
  min_notional: 10       # สั่งออเดอร์ไม่น้อยกว่า 10 USDT

strategy:
  ema_fast: 20
  ema_slow: 50
  rsi_period: 14
  rsi_buy_below: 55
  rsi_sell_above: 45
  atr_period: 14
  sl_atr: 2.0            # คำนวณ stop-loss ~ 2*ATR
  tp_rr: 1.5             # take-profit = RR 1:1.5 เทียบกับความเสี่ยง

logs:
  level: INFO            # DEBUG | INFO | WARNING | ERROR
  file: bot.log
```

---

## data.py
```python
import pandas as pd
import pandas_ta as ta
from typing import Optional

COLS = ["timestamp", "open", "high", "low", "close", "volume"]

def ohlcv_to_df(ohlcv: list) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df

def add_indicators(df: pd.DataFrame, ema_fast: int, ema_slow: int, rsi_period: int, atr_period: int) -> pd.DataFrame:
    out = df.copy()
    out[f"ema_{ema_fast}"] = ta.ema(out["close"], length=ema_fast)
    out[f"ema_{ema_slow}"] = ta.ema(out["close"], length=ema_slow)
    out["rsi"] = ta.rsi(out["close"], length=rsi_period)
    atr = ta.atr(high=out["high"], low=out["low"], close=out["close"], length=atr_period)
    out["atr"] = atr
    return out.dropna()

def last_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or len(df) == 0:
        return None
    return df.iloc[-1]
```

---

## strategy.py
```python
import pandas as pd

class EMARsiStrategy:
    def __init__(self, ema_fast: int, ema_slow: int, rsi_period: int, rsi_buy_below: float, rsi_sell_above: float, sl_atr: float, tp_rr: float):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_buy_below = rsi_buy_below
        self.rsi_sell_above = rsi_sell_above
        self.sl_atr = sl_atr
        self.tp_rr = tp_rr

    def signal(self, df: pd.DataFrame):
        # ใช้สัญญาณ cross + RSI filter
        if len(df) < 3:
            return None
        fast = df[f"ema_{self.ema_fast}"]
        slow = df[f"ema_{self.ema_slow}"]
        rsi = df["rsi"]
        # cross ล่าสุดและก่อนหน้า
        fast_prev, fast_now = float(fast.iloc[-2]), float(fast.iloc[-1])
        slow_prev, slow_now = float(slow.iloc[-2]), float(slow.iloc[-1])
        rsi_now = float(rsi.iloc[-1])

        if fast_prev <= slow_prev and fast_now > slow_now and rsi_now <= self.rsi_buy_below:
            return "buy"
        if fast_prev >= slow_prev and fast_now < slow_now and rsi_now >= self.rsi_sell_above:
            return "sell"
        return None

    def stops(self, entry: float, atr: float):
        sl = entry - self.sl_atr * atr
        tp = entry + self.tp_rr * (entry - sl)
        return max(0.0, sl), tp
```

---

## broker.py
```python
import json
import os
from typing import Optional

class PaperBroker:
    def __init__(self, equity: float, fee_rate: float, state_path: str = "paper_state.json"):
        self.equity = float(equity)
        self.fee_rate = float(fee_rate)
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.state_path = state_path
        self._load()

    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r") as f:
                s = json.load(f)
                self.equity = s.get("equity", self.equity)
                self.position_qty = s.get("position_qty", 0.0)
                self.entry_price = s.get("entry_price", 0.0)

    def _save(self):
        with open(self.state_path, "w") as f:
            json.dump({
                "equity": self.equity,
                "position_qty": self.position_qty,
                "entry_price": self.entry_price,
            }, f, indent=2)

    @property
    def has_position(self) -> bool:
        return self.position_qty > 0

    def buy(self, price: float, qty: float):
        cost = price * qty
        fee = cost * self.fee_rate
        if cost + fee > self.equity:
            raise ValueError("Not enough equity for this trade")
        self.equity -= (cost + fee)
        # สมมติ 1 position เท่านั้น — ถ้ามีของเดิมรวมเฉลี่ย
        total_cost = self.entry_price * self.position_qty + cost
        total_qty = self.position_qty + qty
        self.entry_price = total_cost / total_qty
        self.position_qty = total_qty
        self._save()
        return {"side": "buy", "price": price, "qty": qty, "fee": fee}

    def sell(self, price: float, qty: Optional[float] = None):
        if not self.has_position:
            return {"side": "sell", "skipped": True}
        if qty is None or qty > self.position_qty:
            qty = self.position_qty
        proceeds = price * qty
        fee = proceeds * self.fee_rate
        pnl = (price - self.entry_price) * qty
        self.equity += (proceeds - fee)
        self.position_qty -= qty
        if self.position_qty == 0:
            self.entry_price = 0.0
        self._save()
        return {"side": "sell", "price": price, "qty": qty, "fee": fee, "pnl": pnl}

# โครงสำหรับ LiveBroker ผ่าน CCXT (สาธิต ไม่ผูก exchange ใด ๆ เฉพาะ)
import ccxt

class LiveBroker:
    def __init__(self, exchange_id: str, api_key: str, api_secret: str, use_testnet: bool = False):
        ex_class = getattr(ccxt, exchange_id)
        self.exchange = ex_class({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })
        # สลับ testnet ถ้ารองรับ
        if use_testnet and hasattr(self.exchange, "urls") and "test" in self.exchange.urls:
            self.exchange.urls["api"] = self.exchange.urls["test"]

    def market_buy(self, symbol: str, amount: float):
        return self.exchange.create_market_buy_order(symbol, amount)

    def market_sell(self, symbol: str, amount: float):
        return self.exchange.create_market_sell_order(symbol, amount)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
```

---

## bot.py
```python
import os
import time
import yaml
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

from broker import PaperBroker, LiveBroker
from data import ohlcv_to_df, add_indicators, last_row
from strategy import EMARsiStrategy

@dataclass
class RiskCfg:
    equity: float
    risk_per_trade: float
    fee_rate: float
    min_notional: float

@dataclass
class Cfg:
    mode: str
    exchange: str
    symbol: str
    timeframe: str
    lookback_bars: int
    poll_interval_sec: int
    risk: RiskCfg
    strategy: dict
    logs: dict


def load_config(path: str = "config.yaml") -> Cfg:
    with open(path, "r") as f:
        c = yaml.safe_load(f)
    return Cfg(
        mode=c["mode"],
        exchange=c["exchange"],
        symbol=c["symbol"],
        timeframe=c["timeframe"],
        lookback_bars=int(c["lookback_bars"]),
        poll_interval_sec=int(c["poll_interval_sec"]),
        risk=RiskCfg(**c["risk"]),
        strategy=c["strategy"],
        logs=c.get("logs", {"level": "INFO", "file": "bot.log"}),
    )


def setup_logger(cfg: Cfg):
    level = getattr(logging, cfg.logs.get("level", "INFO"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(cfg.logs.get("file", "bot.log"))
        ],
    )


def position_size(equity: float, risk_perc: float, entry: float, stop: float, min_notional: float) -> float:
    risk_amount = max(0.0, equity * risk_perc)
    stop_dist = max(1e-6, entry - stop)  # ป้องกันหารศูนย์
    qty = risk_amount / stop_dist
    # ตรวจ min notional
    notional = qty * entry
    if notional < min_notional:
        qty = min_notional / entry
    return qty


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_df(broker, symbol: str, timeframe: str, limit: int):
    ohlcv = broker.fetch_ohlcv(symbol, timeframe, limit)
    return ohlcv_to_df(ohlcv)


def main():
    load_dotenv()
    cfg = load_config()
    setup_logger(cfg)

    logging.info(f"Starting bot in {cfg.mode.upper()} mode on {cfg.exchange} {cfg.symbol} {cfg.timeframe}")

    use_testnet = os.getenv("USE_TESTNET", "false").lower() == "true"
    api_key = os.getenv("API_KEY", "")
    api_secret = os.getenv("API_SECRET", "")

    if cfg.mode == "live":
        live = LiveBroker(cfg.exchange, api_key, api_secret, use_testnet=use_testnet)
        data_broker = live
    else:
        # paper mode ใช้ ccxt ดึงราคาแบบ public endpoint
        from broker import LiveBroker as MarketData
        data_broker = MarketData(cfg.exchange, "", "", use_testnet=use_testnet)

    paper = PaperBroker(cfg.risk.equity, cfg.risk.fee_rate)

    strat = EMARsiStrategy(
        ema_fast=cfg.strategy["ema_fast"],
        ema_slow=cfg.strategy["ema_slow"],
        rsi_period=cfg.strategy["rsi_period"],
        rsi_buy_below=cfg.strategy["rsi_buy_below"],
        rsi_sell_above=cfg.strategy["rsi_sell_above"],
        sl_atr=cfg.strategy["sl_atr"],
        tp_rr=cfg.strategy["tp_rr"],
    )

    last_candle_time = None

    while True:
        try:
            df = fetch_df(data_broker, cfg.symbol, cfg.timeframe, cfg.lookback_bars)
            df = add_indicators(
                df,
                ema_fast=cfg.strategy["ema_fast"],
                ema_slow=cfg.strategy["ema_slow"],
                rsi_period=cfg.strategy["rsi_period"],
                atr_period=cfg.strategy["atr_period"],
            )
            row = last_row(df)
            if row is None:
                time.sleep(cfg.poll_interval_sec)
                continue

            # ทำงานเฉพาะเมื่อเกิดแท่งใหม่
            if last_candle_time is not None and row.name == last_candle_time:
                time.sleep(cfg.poll_interval_sec)
                continue

            last_candle_time = row.name
            price = float(row["close"])
            atr = float(row["atr"]) if "atr" in row else max(1e-6, price * 0.02)

            sig = strat.signal(df)
            logging.info(f"New candle: price={price:.2f} signal={sig} position={'LONG' if paper.has_position else 'FLAT'} equity={paper.equity:.2f}")

            if sig == "buy" and not paper.has_position:
                sl, tp = strat.stops(price, atr)
                qty = position_size(paper.equity, cfg.risk.risk_per_trade, price, sl, cfg.risk.min_notional)
                if cfg.mode == "live":
                    logging.warning("Live mode sizing shown, but order execution is not enabled in this template.")
                res = paper.buy(price, qty)
                logging.info(f"BUY qty={qty:.6f} @ {price:.2f} SL={sl:.2f} TP={tp:.2f} -> {res}")

            elif sig == "sell" and paper.has_position:
                res = paper.sell(price)
                logging.info(f"SELL ALL @ {price:.2f} -> {res}")

        except Exception as e:
            logging.exception(f"Loop error: {e}")
        time.sleep(cfg.poll_interval_sec)

if __name__ == "__main__":
    main()
```

---

## backtest.py
```python
import yaml
import pandas as pd
from data import ohlcv_to_df, add_indicators
from strategy import EMARsiStrategy
from broker import LiveBroker

# backtest แบบง่าย: เข้าซื้อเต็มไม้เมื่อสัญญาณ BUY และขายทั้งหมดเมื่อ SELL

def backtest(exchange: str, symbol: str, timeframe: str, lookback: int, strat_cfg: dict, equity: float = 10000, fee_rate: float = 0.001):
    mkt = LiveBroker(exchange, "", "")
    df = ohlcv_to_df(mkt.fetch_ohlcv(symbol, timeframe, limit=lookback))
    df = add_indicators(df, strat_cfg["ema_fast"], strat_cfg["ema_slow"], strat_cfg["rsi_period"], strat_cfg["atr_period"]).copy()

    strat = EMARsiStrategy(
        ema_fast=strat_cfg["ema_fast"],
        ema_slow=strat_cfg["ema_slow"],
        rsi_period=strat_cfg["rsi_period"],
        rsi_buy_below=strat_cfg["rsi_buy_below"],
        rsi_sell_above=strat_cfg["rsi_sell_above"],
        sl_atr=strat_cfg["sl_atr"],
        tp_rr=strat_cfg["tp_rr"],
    )

    position = 0.0
    cash = equity
    entry = 0.0

    prev_time = None
    for t, row in df.iterrows():
        sig = strat.signal(df.loc[:t])
        price = float(row["close"])
        if sig == "buy" and position == 0.0:
            # ซื้อเต็มไม้ด้วยเงินทั้งหมด (หัก fee)
            qty = (cash * (1 - fee_rate)) / price
            position = qty
            entry = price
            cash = 0.0
        elif sig == "sell" and position > 0.0:
            cash = position * price * (1 - fee_rate)
            position = 0.0
            entry = 0.0
        prev_time = t

    # มูลค่าพอร์ตสุดท้าย
    final_value = cash + position * float(df.iloc[-1]["close"]) * (1 - fee_rate)
    ret = (final_value / equity) - 1.0
    print(f"Final equity: {final_value:.2f}  | Return: {ret*100:.2f}% | Bars: {len(df)}")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    backtest(
        exchange=cfg["exchange"],
        symbol=cfg["symbol"],
        timeframe=cfg["timeframe"],
        lookback=cfg["lookback_bars"],
        strat_cfg=cfg["strategy"],
        equity=cfg["risk"]["equity"],
        fee_rate=cfg["risk"]["fee_rate"],
    )
```

---

## Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "bot.py"]
```

---

## docker-compose.yml
```yaml
version: "3.9"
services:
  crypto-bot:
    build: .
    env_file: .env
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./paper_state.json:/app/paper_state.json
      - ./bot.log:/app/bot.log
    restart: unless-stopped
```

---

## วิธีเริ่มต้นแบบรวดเร็ว (Quickstart)
1) สร้าง virtualenv แล้วติดตั้งไลบรารี
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
2) คัดลอก `.env.sample` เป็น `.env` (ถ้าโหมด `paper` ไม่ต้องใส่คีย์ก็ได้)
3) ปรับ `config.yaml` (symbol, timeframe, risk)
4) **Backtest** ก่อนเสมอ
```
python backtest.py
```
5) รันบอท (paper)
```
python bot.py
```
6) ถ้าพร้อมค่อยสลับ `mode: live` และใส่คีย์ของ exchange ให้ถูกต้อง + ทดสอบ testnet หากรองรับ

---

## แนวทางต่อยอด
- เพิ่ม Trailing Stop / OCO (สำหรับ exchange ที่รองรับ) และบันทึก trade log เป็น CSV/SQLite
- เพิ่ม Position Sizing แบบ ATR เต็มรูปแบบ, Max concurrent positions, Daily loss limit
- Webhook แจ้งเตือนเข้า LINE หรือ n8n เมื่อมีสัญญาณ/เข้าออกออเดอร์
- แยกกลยุทธ์เป็นหลายไฟล์และเลือกใช้จาก `config.yaml` (เช่น Breakout, Mean Reversion, Grid)

---

## หมายเหตุสำคัญ
- ใช้ timeframe ที่ปริมาณเทรดเพียงพอ (15m/1h เริ่มต้นดี)
- ขนาดคำสั่งต้องผ่านเกณฑ์ขั้นต่ำของตลาด (min_notional)
- ผล backtest ไม่การันตีกำไรในอนาคต ควรทำ Walk-forward/Out-of-sample เพิ่มเติม

