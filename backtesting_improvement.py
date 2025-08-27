import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import yaml
import pandas as pd
import numpy as np
import itertools
import importlib
import matplotlib.pyplot as plt
from data import ohlcv_to_df, add_indicators
from broker import LiveBroker

def sharpe_ratio(returns, risk_free_rate=0.0):
    # Calculate the average return
    avg_return = np.mean(returns)
    # Calculate the standard deviation of returns
    std_dev = np.std(returns)
    # Calculate the Sharpe ratio
    sharpe = (avg_return - risk_free_rate) / std_dev if std_dev != 0 else 0
    return sharpe

def generate_param_grid(params):
    keys, values = zip(*params.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

def plot_results(df, equity_curve, trades, strat_cfg):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot prices and EMAs
    ax1.plot(df.index, df['close'], label='Close Price')
    ax1.plot(df.index, df[f'ema_{strat_cfg["ema_fast"]}'], label=f'EMA {strat_cfg["ema_fast"]}')
    ax1.plot(df.index, df[f'ema_{strat_cfg["ema_slow"]}'], label=f'EMA {strat_cfg["ema_slow"]}')

    # Plot buy/sell signals
    buy_signals = [trade['timestamp'] for trade in trades if trade['type'] == 'buy']
    sell_signals = [trade['timestamp'] for trade in trades if trade['type'] == 'sell']
    ax1.plot(buy_signals, df.loc[buy_signals]['close'], '^', markersize=10, color='g', label='Buy Signal')
    ax1.plot(sell_signals, df.loc[sell_signals]['close'], 'v', markersize=10, color='r', label='Sell Signal')
    
    ax1.set_title('Price, EMAs, and Trading Signals')
    ax1.legend()

    # Plot equity curve
    ax2.plot(df.index, equity_curve, label='Equity Curve')
    ax2.set_title('Equity Curve')
    ax2.legend()

    plt.xlabel('Date')
    plt.tight_layout()
    strat_name = strat_cfg.get("name", "unknown_strategy")
    plt.savefig(f'backtest_results_{strat_name}_{strat_cfg["ema_fast"]}_{strat_cfg["ema_slow"]}.png')
    plt.close()

def backtest(exchange: str, symbol: str, timeframe: str, lookback: int, strat_cfg: dict, equity: float = 10000, fee_rate: float = 0.001):
    mkt = LiveBroker(exchange, "", "")
    
    strategy_name = strat_cfg.pop("name")
    strategy_module = importlib.import_module(f".strategies.{strategy_name}", package=__name__)
    strategy_class = getattr(strategy_module, strategy_name.replace("_", "").title() + "Strategy")
    strat = strategy_class(**strat_cfg)
    strat_cfg["name"] = strategy_name

    df = ohlcv_to_df(mkt.fetch_ohlcv(symbol, timeframe, limit=lookback))
    df = add_indicators(df, **strat.indicator_params).copy()

    # Check if we have enough data
    if len(df) == 0:
        print("No data available for backtesting")
        return

    position = 0.0
    cash = equity
    entry = 0.0
    trades = []
    equity_curve = []

    for t, row in df.iterrows():
        sig = strat.signal(df.loc[:t])
        price = float(row["close"])
        
        # Update equity curve
        current_value = cash + position * price * (1 - fee_rate)
        equity_curve.append(current_value)

        if sig == "buy" and position == 0.0:
            qty = (cash * (1 - fee_rate)) / price
            position = qty
            entry = price
            cash = 0.0
            trades.append({"type": "buy", "price": price, "qty": qty, "timestamp": t})
        elif sig == "sell" and position > 0.0:
            cash = position * price * (1 - fee_rate)
            pnl = (price - entry) * position
            position = 0.0
            entry = 0.0
            trades.append({"type": "sell", "price": price, "pnl": pnl, "timestamp": t})

    # Final portfolio value
    if len(df) > 0:
        final_value = cash + position * float(df.iloc[-1]["close"]) * (1 - fee_rate)
        returns = (final_value / equity) - 1.0
        
        # Performance Metrics
        returns_list = pd.Series(equity_curve).pct_change().dropna()
        sharpe = sharpe_ratio(returns_list)
        
        # Max Drawdown
        equity_series = pd.Series(equity_curve)
        cumulative_max = equity_series.cummax()
        drawdown = (equity_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        # Win/Loss Ratio
        wins = len([trade for trade in trades if trade.get("pnl", 0) > 0])
        losses = len([trade for trade in trades if trade.get("pnl", 0) < 0])
        win_loss_ratio = wins / losses if losses > 0 else float('inf')

        print(f"--- Strategy: {strat_cfg} ---")
        print(f"Final Equity: {final_value:.2f}")
        print(f"Return: {returns*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
        print(f"Total Trades: {len(trades)}")
        print(f"Bars: {len(df)}")
        print("---------------------\\n")

        plot_results(df, equity_curve, trades, strat_cfg)

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    param_grid = {
        'name': ['ema_rsi'],
        'ema_fast': [10, 20],
        'ema_slow': [50, 100],
        'rsi_period': [14],
        'rsi_buy_below': [50, 55],
        'rsi_sell_above': [45, 50],
        'atr_period': [14],
        'sl_atr': [2.0],
        'tp_rr': [1.5]
    }

    for params in generate_param_grid(param_grid):
        strat_cfg = cfg['strategy'].copy()
        strat_cfg.update(params)
        backtest(
            exchange=cfg["exchange"],
            symbol=cfg["symbol"],
            timeframe=cfg["timeframe"],
            lookback=cfg["lookback_bars"],
            strat_cfg=strat_cfg,
            equity=cfg["risk"]["equity"],
            fee_rate=cfg["risk"]["fee_rate"],
        )