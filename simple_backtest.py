import yaml
import pandas as pd
from data import ohlcv_to_df
from simple_strategy import SimplePriceStrategy
from broker import LiveBroker

def simple_backtest(exchange: str, symbol: str, timeframe: str, lookback: int, period: int = 5, equity: float = 10000, fee_rate: float = 0.001):
    mkt = LiveBroker(exchange, "", "")
    df = ohlcv_to_df(mkt.fetch_ohlcv(symbol, timeframe, limit=lookback))
    
    if len(df) == 0:
        print("No data available for backtesting")
        return
    
    strat = SimplePriceStrategy(period=period)

    position = 0.0
    cash = equity

    for t, row in df.iterrows():
        sig = strat.signal(df.loc[:t])
        price = float(row["close"])
        if sig == "buy" and position == 0.0:
            # ซื้อเต็มไม้ด้วยเงินทั้งหมด (หัก fee)
            qty = (cash * (1 - fee_rate)) / price
            position = qty
            cash = 0.0
        elif sig == "sell" and position > 0.0:
            cash = position * price * (1 - fee_rate)
            position = 0.0

    # มูลค่าพอร์ตสุดท้าย
    if len(df) > 0:
        final_value = cash + position * float(df.iloc[-1]["close"])
        ret = (final_value / equity) - 1.0
        print(f"Final equity: {final_value:.2f}  | Return: {ret*100:.2f}% | Bars: {len(df)}")
    else:
        print("No data available for backtesting")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    simple_backtest(
        exchange=cfg["exchange"],
        symbol=cfg["symbol"],
        timeframe=cfg["timeframe"],
        lookback=10,  # 10 minutes of data
        period=3,     # Simple moving average period
        equity=cfg["risk"]["equity"],
        fee_rate=cfg["risk"]["fee_rate"],
    )