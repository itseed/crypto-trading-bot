import yaml
import pandas as pd
import importlib
from data import ohlcv_to_df, add_indicators
from broker import LiveBroker

# backtest แบบง่าย: เข้าซื้อเต็มไม้เมื่อสัญญาณ BUY และขายทั้งหมดเมื่อ SELL

def backtest(exchange: str, symbol: str, timeframe: str, lookback: int, strat_cfg: dict, equity: float = 10000, fee_rate: float = 0.001):
    mkt = LiveBroker(exchange, "", "")
    df = ohlcv_to_df(mkt.fetch_ohlcv(symbol, timeframe, limit=lookback))
    
    # Check if we have enough data
    if len(df) == 0:
        print("No data available for backtesting")
        return
    
    # Load strategy dynamically based on config
    strategy_name = strat_cfg.pop("name", "ema_rsi")
    try:
        # Use absolute import instead of relative import
        strategy_module = importlib.import_module(f"strategies.{strategy_name}_strategy")
        # Correctly format the class name
        if strategy_name == "ema_rsi":
            strategy_class = getattr(strategy_module, "EMARsiStrategy")
        else:
            strategy_class = getattr(strategy_module, strategy_name.replace("_", "").title() + "Strategy")
        strat = strategy_class(**strat_cfg)
    except (ImportError, AttributeError) as e:
        print(f"Error loading strategy {strategy_name}: {e}")
        return
    
    # Add indicators if required by the strategy
    if hasattr(strat, 'indicator_params') and strat.indicator_params:
        df = add_indicators(df, **strat.indicator_params).copy()
    else:
        df = df.copy()

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
        strat_cfg=cfg["strategy"].copy(),
        equity=cfg["risk"]["equity"],
        fee_rate=cfg["risk"]["fee_rate"],
    )