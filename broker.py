import json
import os
from typing import Optional
import ccxt
from tenacity import retry, stop_after_attempt, wait_fixed

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
        # Handle the case where qty is None or exceeds position
        if qty is None or qty > self.position_qty:
            qty = self.position_qty
        # At this point qty is guaranteed to be a float, but we need to convince the type checker
        qty_float: float = qty if qty is not None else self.position_qty
        proceeds = price * qty_float
        fee = proceeds * self.fee_rate
        pnl = (price - self.entry_price) * qty_float
        self.equity += (proceeds - fee)
        self.position_qty -= qty_float
        if self.position_qty == 0:
            self.entry_price = 0.0
        self._save()
        return {"side": "sell", "price": price, "qty": qty_float, "fee": fee, "pnl": pnl}

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

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def market_buy(self, symbol: str, amount: float):
        return self.exchange.create_market_buy_order(symbol, amount)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def market_sell(self, symbol: str, amount: float):
        return self.exchange.create_market_sell_order(symbol, amount)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def limit_buy(self, symbol: str, amount: float, price: float):
        return self.exchange.create_limit_buy_order(symbol, amount, price)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def limit_sell(self, symbol: str, amount: float, price: float):
        return self.exchange.create_limit_sell_order(symbol, amount, price)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def stop_loss(self, symbol: str, amount: float, price: float):
        return self.exchange.create_order(symbol, 'stop-loss', 'sell', amount, price)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
