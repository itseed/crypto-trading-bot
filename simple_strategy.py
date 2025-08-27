import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class SimplePriceStrategy(BaseStrategy):
    def __init__(self, period: int = 5):
        self.period = period

    @property
    def indicator_params(self):
        # Simple strategy doesn't require additional indicators
        return {}

    def signal(self, df: pd.DataFrame):
        # Simple strategy: buy when price crosses above moving average, sell when it crosses below
        if len(df) < self.period + 2:
            return None
            
        # Calculate simple moving average and ensure it's a pandas Series
        sma = pd.Series(df["close"].rolling(window=self.period).mean())
        
        # Check if we have valid data - using proper pandas indexing
        if len(sma) < 2 or pd.isna(sma.iloc[-1]) or pd.isna(sma.iloc[-2]):
            return None
            
        # Current and previous prices and moving averages
        price_prev, price_now = float(df["close"].iloc[-2]), float(df["close"].iloc[-1])
        sma_prev, sma_now = float(sma.iloc[-2]), float(sma.iloc[-1])

        if price_prev <= sma_prev and price_now > sma_now:
            return "buy"
        if price_prev >= sma_prev and price_now < sma_now:
            return "sell"
        return None

    def stops(self, entry: float, atr: float):
        # Simple strategy doesn't use stops, returning default values
        return entry * 0.95, entry * 1.05  # 5% stop loss and take profit