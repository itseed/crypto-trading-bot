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

    # Check if we have enough data
    if len(df) == 0:
        print("No data available for backtesting")
        return
    
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
    if len(df) > 0:
        final_value = cash + position * float(df.iloc[-1]["close"]) * (1 - fee_rate)
        ret = (final_value / equity) - 1.0
        print(f"Final equity: {final_value:.2f}  | Return: {ret*100:.2f}% | Bars: {len(df)}")
    else:
        print("No data available for backtesting")

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