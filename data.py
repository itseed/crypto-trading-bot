import pandas as pd
import numpy as np
from typing import Optional, List, Union
import pandas.core.common as com

COLS = ["timestamp", "open", "high", "low", "close", "volume"]

def ohlcv_to_df(ohlcv: list) -> pd.DataFrame:
    # Convert COLS to pandas Index to satisfy type requirements
    columns = pd.Index(COLS)
    df = pd.DataFrame(ohlcv, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df

def ema(series: pd.Series, length: int) -> pd.Series:
    result = series.ewm(span=length, adjust=False).mean()
    return pd.Series(result, name=series.name)

def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    result = 100 - (100 / (1 + rs))
    return pd.Series(result, name=series.name)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift())
    tr2 = abs(low - close.shift())
    tr = pd.DataFrame({'tr0': tr0, 'tr1': tr1, 'tr2': tr2}).max(axis=1)
    result = tr.rolling(period).mean()
    return pd.Series(result, name='atr')

def add_indicators(df: pd.DataFrame, ema_fast: int, ema_slow: int, rsi_period: int, atr_period: int) -> pd.DataFrame:
    out = df.copy()
    # Explicitly cast to Series to satisfy type checker
    close_series = pd.Series(out["close"])
    high_series = pd.Series(out["high"])
    low_series = pd.Series(out["low"])
    
    out[f"ema_{ema_fast}"] = ema(close_series, ema_fast)
    out[f"ema_{ema_slow}"] = ema(close_series, ema_slow)
    out["rsi"] = rsi(close_series, rsi_period)
    out["atr"] = atr(high_series, low_series, close_series, atr_period)
    return out.dropna()

def last_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or len(df) == 0:
        return None
    return df.iloc[-1]