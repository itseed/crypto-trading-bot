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