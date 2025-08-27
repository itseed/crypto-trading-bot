import pandas as pd
from strategies.base_strategy import BaseStrategy

class EMARsiStrategy(BaseStrategy):
    def __init__(self, ema_fast: int, ema_slow: int, rsi_period: int, rsi_buy_below: float, rsi_sell_above: float, atr_period:int, sl_atr: float, tp_rr: float):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_buy_below = rsi_buy_below
        self.rsi_sell_above = rsi_sell_above
        self.sl_atr = sl_atr
        self.tp_rr = tp_rr
        self.atr_period = atr_period

    @property
    def indicator_params(self):
        return {
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "rsi_period": self.rsi_period,
            "atr_period": self.atr_period
        }

    def signal(self, df: pd.DataFrame):
        # ใช้สัญญาณ cross + RSI filter
        if len(df) < max(3, self.ema_slow, self.ema_fast, self.rsi_period):
            return None
            
        # Check if required columns exist and have valid data
        ema_fast_col = f"ema_{self.ema_fast}"
        ema_slow_col = f"ema_{self.ema_slow}"
        
        if ema_fast_col not in df.columns or ema_slow_col not in df.columns or "rsi" not in df.columns:
            return None
            
        # Check if we have valid data for the indicators
        if pd.isna(df[ema_fast_col].iloc[-1]) or pd.isna(df[ema_slow_col].iloc[-1]) or pd.isna(df["rsi"].iloc[-1]):
            return None
            
        fast = df[ema_fast_col]
        slow = df[ema_slow_col]
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
