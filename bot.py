import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import time
import yaml
import logging
import importlib
import pandas as pd
from dataclasses import dataclass
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

from broker import PaperBroker, LiveBroker
from data import ohlcv_to_df, add_indicators, last_row

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


def validate_strategy_config(strategy_name: str, strategy_params: dict) -> bool:
    """
    Validate strategy configuration parameters
    """
    required_params = {
        "ema_rsi": ["ema_fast", "ema_slow", "rsi_period", "rsi_buy_below", "rsi_sell_above", "atr_period", "sl_atr", "tp_rr"],
        "simple_price": ["period"]
    }
    
    if strategy_name not in required_params:
        logging.warning(f"Unknown strategy {strategy_name}, skipping validation")
        return True
        
    missing_params = [param for param in required_params[strategy_name] if param not in strategy_params]
    if missing_params:
        logging.error(f"Missing required parameters for {strategy_name} strategy: {missing_params}")
        return False
        
    # Validate parameter values
    try:
        if strategy_name == "ema_rsi":
            if not (0 < strategy_params["rsi_buy_below"] < 100):
                logging.error("rsi_buy_below must be between 0 and 100")
                return False
            if not (0 < strategy_params["rsi_sell_above"] < 100):
                logging.error("rsi_sell_above must be between 0 and 100")
                return False
            if strategy_params["ema_fast"] >= strategy_params["ema_slow"]:
                logging.error("ema_fast must be less than ema_slow")
                return False
                
        elif strategy_name == "simple_price":
            if strategy_params["period"] <= 0:
                logging.error("period must be positive")
                return False
                
        return True
    except Exception as e:
        logging.error(f"Error validating strategy parameters: {e}")
        return False


def load_config(path: str = "config.yaml") -> Cfg:
    try:
        with open(path, "r") as f:
            c = yaml.safe_load(f)
            
        # Validate strategy configuration
        strategy_name = c["strategy"].get("name", "ema_rsi")
        if not validate_strategy_config(strategy_name, c["strategy"]):
            raise ValueError("Invalid strategy configuration")
            
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
    except FileNotFoundError:
        logging.error(f"Configuration file {path} not found")
        raise
    except KeyError as e:
        logging.error(f"Missing required configuration key: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise


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
    try:
        risk_amount = max(0.0, equity * risk_perc)
        stop_dist = max(1e-6, entry - stop)  # ป้องกันหารศูนย์
        qty = risk_amount / stop_dist
        # ตรวจ min notional
        notional = qty * entry
        if notional < min_notional:
            qty = min_notional / entry
        return qty
    except Exception as e:
        logging.error(f"Error calculating position size: {e}")
        return 0.0


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_df(broker, symbol: str, timeframe: str, limit: int):
    try:
        ohlcv = broker.fetch_ohlcv(symbol, timeframe, limit)
        return ohlcv_to_df(ohlcv)
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        raise


def main():
    try:
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

        # Load strategy dynamically
        strategy_name = cfg.strategy.get("name", "ema_rsi")
        if "name" in cfg.strategy:
            cfg.strategy.pop("name")
            
        try:
            # Use absolute import instead of relative import
            strategy_module = importlib.import_module(f"strategies.{strategy_name}_strategy")
            # Correctly format the class name
            if strategy_name == "ema_rsi":
                strategy_class = getattr(strategy_module, "EMARsiStrategy")
            else:
                strategy_class = getattr(strategy_module, strategy_name.replace("_", "").title() + "Strategy")
            strat = strategy_class(**cfg.strategy)
        except (ImportError, AttributeError) as e:
            logging.error(f"Error loading strategy {strategy_name}: {e}")
            raise

        last_candle_time = None

        while True:
            try:
                df = fetch_df(data_broker, cfg.symbol, cfg.timeframe, cfg.lookback_bars)
                
                # Add indicators if required by the strategy
                if hasattr(strat, 'indicator_params') and strat.indicator_params:
                    df = add_indicators(df, **strat.indicator_params)
                
                row = last_row(df)
                if row is None:
                    logging.warning("No data received, skipping this iteration")
                    time.sleep(cfg.poll_interval_sec)
                    continue

                # ทำงานเฉพาะเมื่อเกิดแท่งใหม่
                if last_candle_time is not None and row.name == last_candle_time:
                    time.sleep(cfg.poll_interval_sec)
                    continue

                last_candle_time = row.name
                price = float(row["close"])
                
                # Calculate ATR or use default value
                atr = max(1e-6, price * 0.02)
                if "atr" in row.index:
                    atr_value = row["atr"]
                    # Check if atr_value is not NaN by comparing with itself (NaN != NaN is True)
                    if atr_value == atr_value:  # This is False for NaN values
                        atr = float(atr_value)

                sig = strat.signal(df)
                logging.info(f"New candle: price={price:.2f} signal={sig} position={'LONG' if paper.has_position else 'FLAT'} equity={paper.equity:.2f}")

                if sig == "buy" and not paper.has_position:
                    try:
                        sl, tp = strat.stops(price, atr)
                        qty = position_size(paper.equity, cfg.risk.risk_per_trade, price, sl, cfg.risk.min_notional)
                        if qty <= 0:
                            logging.warning("Calculated quantity is zero or negative, skipping buy order")
                            continue
                        if cfg.mode == "live":
                            logging.warning("Live mode sizing shown, but order execution is not enabled in this template.")
                        res = paper.buy(price, qty)
                        logging.info(f"BUY qty={qty:.6f} @ {price:.2f} SL={sl:.2f} TP={tp:.2f} -> {res}")
                    except Exception as e:
                        logging.error(f"Error executing buy order: {e}")

                elif sig == "sell" and paper.has_position:
                    try:
                        res = paper.sell(price)
                        logging.info(f"SELL ALL @ {price:.2f} -> {res}")
                    except Exception as e:
                        logging.error(f"Error executing sell order: {e}")

            except Exception as e:
                logging.exception(f"Loop error: {e}")
                # Continue running even if there's an error in one iteration
            time.sleep(cfg.poll_interval_sec)
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.exception(f"Fatal error in main loop: {e}")
        raise

if __name__ == "__main__":
    main()