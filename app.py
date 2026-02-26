# -*- coding: utf-8 -*-
"""
Deep Reinforcement Learning Market Maker with Order Book Dynamics
Author: Spencer Purdy
Description: Production-grade HFT market making system using Deep RL (PPO) for optimal
             bid-ask placement with real-time order book simulation and comprehensive analytics.

Problem Statement: Develop an intelligent market maker that provides liquidity by continuously
                   quoting bid and ask prices while managing inventory risk and maximizing PnL.

Real-World Application: Automated market making for HFT firms, crypto exchanges, and
                        electronic trading desks requiring optimal liquidity provision.

Key Features:
- Pre-trained PPO agent loaded at startup (trained on GPU via Google Colab)
- Real-time limit order book (LOB) simulation with realistic market microstructure
- Order book features: spread, depth, imbalance, order flow toxicity (VPIN)
- Market impact modeling with adverse selection costs
- Transaction cost analysis and inventory risk management
- Live performance metrics: PnL, Sharpe ratio, fill rates, inventory risk
- Interactive order book depth visualization

Technical Components:
- Custom OpenAI Gym environment for market making (12-D obs, 2-D continuous action)
- Proximal Policy Optimization (PPO) with clipped surrogate objective
- Avellaneda-Stoikov inspired reward: PnL change + quadratic inventory penalty
- Poisson-driven order arrival process for realistic background flow
- Square-root market impact model
- Book uncrossing logic to prevent bid-ask inversions
- Advanced order book feature engineering

Model Architecture:
- Policy network: MLP (64 -> 64) with tanh activation
- Value network: MLP (64 -> 64) with tanh activation
- Action space: continuous [bid_offset, ask_offset] in ticks from mid price

Performance Benchmarks (Expected):
- Sharpe Ratio: 0.5-2.0 (after convergence)
- Fill Rate: 40-80%
- Average Spread Capture: 30-60% of bid-ask spread
- Max Drawdown: <5% of starting capital

Limitations:
- Assumes simplified order execution (no partial fills modeled in detail)
- Market impact model is simplified (square-root impact function)
- Does not account for extreme market events (flash crashes)
- Performance degrades during low liquidity periods
- Model trained on specific instruments may not generalize across asset classes

Failure Modes:
- Adverse selection during strong directional moves
- Inventory accumulation in trending markets
- Spread widening during volatility spikes
- Reduced profitability in low-volatility regimes

Hardware Requirements:
- This application runs on CPU (model weights pre-trained on GPU)
- RAM: 16GB minimum

Dependencies:
- PyTorch 2.0+
- Stable-Baselines3 (PPO implementation)
- Gymnasium (environment framework)
- Plotly (interactive visualizations)
- Gradio (web interface)

Reproducibility:
- Random seed: 42 (all libraries)
- Deterministic mode enabled
- All hyperparameters documented in MarketMakerConfig

License: MIT License
Author: Spencer Purdy
Purpose: Portfolio demonstration of HFT/RL engineering skills

Disclaimer: This is a simulation system for educational purposes. Real trading involves
            significant risks and regulatory requirements not covered in this demonstration.
"""

# ============================================================================
# INSTALLATION (uncomment for Colab)
# ============================================================================

# !pip install -q torch torchvision torchaudio stable-baselines3[extra] gymnasium pandas numpy requests plotly gradio scipy scikit-learn seaborn matplotlib

# ============================================================================
# IMPORTS
# ============================================================================

import os
import json
import time
import random
import warnings
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import requests

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import gradio as gr

# Suppress noisy warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION AND REPRODUCIBILITY
# ============================================================================

# Global random seed for all libraries
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logger.info(f"Random seed set to {RANDOM_SEED} for reproducibility")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")


@dataclass
class MarketMakerConfig:
    """
    Configuration for the RL Market Maker system.
    All hyperparameters documented with ranges and defaults.
    """
    # Random seed
    random_seed: int = RANDOM_SEED

    # Market parameters
    initial_price: float = 100.0
    tick_size: float = 0.01
    lot_size: int = 100
    max_inventory: int = 1000
    initial_cash: float = 100000.0

    # Order book parameters
    order_book_depth: int = 10       # Number of price levels per side
    base_spread_bps: float = 5.0     # Base spread in basis points
    volatility_annual: float = 0.20  # 20% annualized volatility
    arrival_rate_lambda: float = 10.0  # Orders per second (Poisson rate)

    # Market impact parameters
    permanent_impact_coef: float = 0.1
    temporary_impact_coef: float = 0.5
    adverse_selection_factor: float = 0.3

    # Transaction costs
    fee_bps: float = 1.0      # Transaction fee in basis points
    slippage_bps: float = 0.5  # Expected slippage in basis points

    # Episode parameters
    episode_length: int = 1000         # Steps per episode (1 step = 1 second)
    time_step_seconds: float = 1.0

    # PPO Hyperparameters (reference from training)
    ppo_learning_rate: float = 0.0003
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 256
    ppo_n_epochs: int = 10
    ppo_ent_coef: float = 0.03
    ppo_vf_coef: float = 0.5

    # Training parameters (reference only)
    total_timesteps: int = 500000
    eval_freq: int = 10000
    n_eval_episodes: int = 10

    # Performance thresholds
    min_sharpe_ratio: float = 1.0
    max_drawdown_pct: float = 20.0
    min_fill_rate: float = 0.5

    # Paths
    models_dir: str = "./rl_models"
    logs_dir: str = "./rl_logs"
    data_dir: str = "./market_data"

    # Pre-trained model weights (uploaded to HF Spaces repo root)
    pretrained_model_path: str = "ppo_market_maker.zip"
    training_metrics_path: str = "training_metrics.json"

    # API Configuration (loaded from environment / Hugging Face Spaces secrets)
    polygon_api_key: str = field(
        default_factory=lambda: os.environ.get("POLYGON_API_KEY", "")
    )
    alphavantage_api_key: str = field(
        default_factory=lambda: os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    )
    default_symbol: str = "AAPL"


# Initialize global configuration
config = MarketMakerConfig()

# Ensure runtime directories exist
os.makedirs(config.models_dir, exist_ok=True)
os.makedirs(config.logs_dir, exist_ok=True)
os.makedirs(config.data_dir, exist_ok=True)

# ============================================================================
# API DATA FETCHERS
# ============================================================================


class MarketDataFetcher:
    """
    Fetches historical market data from Polygon and Alpha Vantage APIs.
    Used to initialize realistic price distributions and volatility parameters.
    Falls back to synthetic GBM data when API keys are unavailable.
    """

    def __init__(self, cfg: MarketMakerConfig):
        self.config = cfg
        self.polygon_base = "https://api.polygon.io"
        self.av_base = "https://www.alphavantage.co/query"
        self.polygon_api_key = cfg.polygon_api_key
        self.alphavantage_api_key = cfg.alphavantage_api_key

        if self.polygon_api_key:
            logger.info("Polygon API key loaded")
        if self.alphavantage_api_key:
            logger.info("Alpha Vantage API key loaded")

    def fetch_polygon_quotes(
        self, symbol: str, date: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch level-1 OHLCV minute bars from Polygon.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            date: Date in YYYY-MM-DD format

        Returns:
            DataFrame with [timestamp, open, high, low, close, volume] or None
        """
        if not self.polygon_api_key:
            logger.warning("Polygon API key not set, using synthetic data")
            return None

        try:
            url = (
                f"{self.polygon_base}/v2/aggs/ticker/{symbol}"
                f"/range/1/minute/{date}/{date}"
            )
            params = {"apiKey": self.polygon_api_key, "limit": 50000}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if "results" in data and data["results"]:
                df = pd.DataFrame(data["results"])
                df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                df = df.rename(
                    columns={
                        "c": "close", "h": "high", "l": "low",
                        "o": "open", "v": "volume",
                    }
                )
                logger.info(
                    f"Fetched {len(df)} minute bars from Polygon for {symbol}"
                )
                return df[["timestamp", "open", "high", "low", "close", "volume"]]

            logger.warning(f"No data returned from Polygon for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching Polygon data: {e}")
            return None

    def fetch_alphavantage_intraday(
        self, symbol: str, interval: str = "1min"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch intraday OHLCV bars from Alpha Vantage.

        Args:
            symbol: Stock ticker
            interval: Time interval (1min, 5min, 15min, 30min, 60min)

        Returns:
            DataFrame with OHLCV data or None
        """
        if not self.alphavantage_api_key:
            logger.warning("Alpha Vantage API key not set, using synthetic data")
            return None

        try:
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": interval,
                "apikey": self.alphavantage_api_key,
                "outputsize": "full",
            }
            response = requests.get(self.av_base, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()
            time_series_key = f"Time Series ({interval})"

            if time_series_key in data:
                df = pd.DataFrame.from_dict(
                    data[time_series_key], orient="index"
                )
                df.index = pd.to_datetime(df.index)
                df = df.rename(
                    columns={
                        "1. open": "open",
                        "2. high": "high",
                        "3. low": "low",
                        "4. close": "close",
                        "5. volume": "volume",
                    }
                )
                df = df.astype(float).sort_index()
                logger.info(
                    f"Fetched {len(df)} {interval} bars from Alpha Vantage "
                    f"for {symbol}"
                )
                return df

            logger.warning(
                f"No intraday data in Alpha Vantage response for {symbol}"
            )
            return None

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data: {e}")
            return None

    def calculate_volatility(self, prices: np.ndarray) -> float:
        """
        Calculate realized annualized volatility from a price series.

        Args:
            prices: Array of prices

        Returns:
            Annualized volatility (scalar)
        """
        if len(prices) < 2:
            return self.config.volatility_annual

        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(252 * 390)

    def generate_synthetic_data(self, n_bars: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data using Geometric Brownian Motion.
        Used as a fallback when API data is unavailable.

        Args:
            n_bars: Number of bars to generate

        Returns:
            DataFrame with synthetic OHLCV data
        """
        logger.info(f"Generating {n_bars} bars of synthetic market data")

        dt = 1 / 252 / 390  # One minute expressed in years
        mu = 0.10            # 10% annual drift
        sigma = self.config.volatility_annual
        S0 = self.config.initial_price

        prices = [S0]
        for _ in range(n_bars - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            prices.append(prices[-1] * (1 + mu * dt + sigma * dW))

        prices = np.array(prices)

        opens = prices
        closes = prices * (1 + np.random.normal(0, 0.0001, n_bars))
        highs = np.maximum(opens, closes) * (
            1 + np.abs(np.random.uniform(0, 0.0005, n_bars))
        )
        lows = np.minimum(opens, closes) * (
            1 - np.abs(np.random.uniform(0, 0.0005, n_bars))
        )
        volumes = np.random.lognormal(10, 1, n_bars).astype(int)

        return pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2024-01-01 09:30:00", periods=n_bars, freq="1min"
                ),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )


# ============================================================================
# ORDER BOOK SIMULATOR
# ============================================================================


class LimitOrderBook:
    """
    Realistic limit order book simulator with multiple price levels.

    Features:
    - Multiple bid/ask levels with configurable depth
    - Poisson order arrival / cancellation dynamics
    - Square-root market impact modeling
    - Order flow toxicity tracking (VPIN-based approximation)
    - Book uncrossing logic to prevent bid-ask inversions
    """

    def __init__(self, cfg: MarketMakerConfig, initial_mid_price: float):
        self.config = cfg
        self.mid_price = initial_mid_price
        self.tick_size = cfg.tick_size

        # Internal book state: {price: size}
        self.bids: Dict[float, int] = {}
        self.asks: Dict[float, int] = {}

        self._initialize_book()

        # Cumulative tracking metrics
        self.total_volume = 0
        self.trade_count = 0
        self.aggressive_buy_volume = 0
        self.aggressive_sell_volume = 0

    def _initialize_book(self):
        """Populate order book with realistic depth around the current mid price."""
        base_spread = self.config.base_spread_bps / 10000 * self.mid_price
        half_spread = base_spread / 2

        for i in range(self.config.order_book_depth):
            bid_price = round(
                self.mid_price - half_spread - i * self.tick_size, 2
            )
            self.bids[bid_price] = int(np.random.lognormal(5, 1) * 100)

            ask_price = round(
                self.mid_price + half_spread + i * self.tick_size, 2
            )
            self.asks[ask_price] = int(np.random.lognormal(5, 1) * 100)

        # Enforce no crossing after initialization
        self._uncross_book()

    def _uncross_book(self):
        """
        Remove any crossed levels where bids >= best ask or asks <= best bid.
        This prevents the book from entering an invalid state where the
        highest bid is at or above the lowest ask.
        """
        if not self.bids or not self.asks:
            return

        best_bid = max(self.bids)
        best_ask = min(self.asks)

        if best_bid >= best_ask:
            mid = (best_bid + best_ask) / 2.0

            crossed_bids = [p for p in self.bids if p >= mid]
            for p in crossed_bids:
                del self.bids[p]

            crossed_asks = [p for p in self.asks if p <= mid]
            for p in crossed_asks:
                del self.asks[p]

    def get_best_bid(self) -> Tuple[float, int]:
        """Return (price, size) of the highest bid level."""
        if not self.bids:
            return self.mid_price - self.tick_size, 0
        best = max(self.bids)
        return best, self.bids[best]

    def get_best_ask(self) -> Tuple[float, int]:
        """Return (price, size) of the lowest ask level."""
        if not self.asks:
            return self.mid_price + self.tick_size, 0
        best = min(self.asks)
        return best, self.asks[best]

    def get_spread(self) -> float:
        """Return the current bid-ask spread (always non-negative)."""
        spread = self.get_best_ask()[0] - self.get_best_bid()[0]
        return max(spread, 0.0)

    def get_mid_price(self) -> float:
        """Return the arithmetic mid price."""
        return (self.get_best_bid()[0] + self.get_best_ask()[0]) / 2

    def get_depth(self, side: str, n_levels: int = 5) -> float:
        """
        Return total size across the top n price levels.

        Args:
            side: 'bid' or 'ask'
            n_levels: Number of levels to aggregate

        Returns:
            Aggregate size (float)
        """
        if side == "bid":
            levels = sorted(self.bids.items(), reverse=True)[:n_levels]
        else:
            levels = sorted(self.asks.items())[:n_levels]
        return sum(sz for _, sz in levels)

    def get_order_imbalance(self) -> float:
        """
        Compute order book imbalance in [-1, 1].
        Positive values indicate more bid-side pressure.
        """
        bid_depth = self.get_depth("bid")
        ask_depth = self.get_depth("ask")
        total = bid_depth + ask_depth
        return (bid_depth - ask_depth) / total if total > 0 else 0.0

    def get_order_flow_toxicity(self) -> float:
        """
        Estimate adverse selection risk via a simplified VPIN-style metric.

        Returns:
            Toxicity score in [0, 1]
        """
        if self.total_volume == 0:
            return 0.0
        imbalance = abs(
            self.aggressive_buy_volume - self.aggressive_sell_volume
        )
        return min(imbalance / max(self.total_volume, 1), 1.0)

    def add_limit_order(self, side: str, price: float, size: int):
        """
        Add or augment a resting limit order at the given price level.
        Rejects orders that would cross the book to maintain integrity.

        Args:
            side: 'bid' or 'ask'
            price: Limit price
            size: Order size in shares
        """
        price = round(price, 2)

        # Reject crossing orders to maintain book integrity
        if side == "bid" and self.asks:
            best_ask = min(self.asks)
            if price >= best_ask:
                return
        elif side == "ask" and self.bids:
            best_bid = max(self.bids)
            if price <= best_bid:
                return

        book = self.bids if side == "bid" else self.asks
        book[price] = book.get(price, 0) + size

    def remove_limit_order(self, side: str, price: float, size: int):
        """Reduce size at the given price level; remove the level if exhausted."""
        price = round(price, 2)
        book = self.bids if side == "bid" else self.asks
        if price in book:
            book[price] = max(0, book[price] - size)
            if book[price] == 0:
                del book[price]

    def execute_market_order(
        self, side: str, size: int
    ) -> Tuple[float, int]:
        """
        Walk the book and fill a market order at the best available prices.

        Args:
            side: 'buy' or 'sell'
            size: Desired fill quantity

        Returns:
            (average_execution_price, filled_size)
        """
        filled_size = 0
        total_cost = 0.0

        if side == "buy":
            for price, available in sorted(self.asks.items()):
                if filled_size >= size:
                    break
                fill = min(size - filled_size, available)
                total_cost += price * fill
                filled_size += fill
                self.remove_limit_order("ask", price, fill)
            self.aggressive_buy_volume += filled_size
        else:
            for price, available in sorted(self.bids.items(), reverse=True):
                if filled_size >= size:
                    break
                fill = min(size - filled_size, available)
                total_cost += price * fill
                filled_size += fill
                self.remove_limit_order("bid", price, fill)
            self.aggressive_sell_volume += filled_size

        self.total_volume += filled_size
        self.trade_count += 1

        avg_price = total_cost / filled_size if filled_size > 0 else 0.0
        return avg_price, filled_size

    def simulate_order_flow(self):
        """
        Drive one time-step of background order flow via a Poisson arrival process.
        Events are randomly allocated among limit order submissions,
        cancellations, and market orders.
        """
        n_events = np.random.poisson(
            self.config.arrival_rate_lambda * self.config.time_step_seconds
        )

        for _ in range(n_events):
            event = np.random.choice(
                ["limit_order", "cancellation", "market_order"],
                p=[0.6, 0.3, 0.1],
            )

            if event == "limit_order":
                side = np.random.choice(["bid", "ask"])
                if side == "bid":
                    price = self.mid_price - np.random.exponential(0.02)
                else:
                    price = self.mid_price + np.random.exponential(0.02)
                price = round(price / self.tick_size) * self.tick_size
                self.add_limit_order(
                    side, price, int(np.random.lognormal(4, 1) * 100)
                )

            elif event == "cancellation":
                side = np.random.choice(["bid", "ask"])
                book = self.bids if side == "bid" else self.asks
                if book:
                    price = np.random.choice(list(book.keys()))
                    cancel_size = int(
                        book[price] * np.random.uniform(0.1, 0.5)
                    )
                    self.remove_limit_order(side, price, cancel_size)

            elif event == "market_order":
                side = np.random.choice(["buy", "sell"])
                self.execute_market_order(
                    side, int(np.random.lognormal(3, 1) * 100)
                )

        # Re-anchor mid price to the book, then apply a small GBM drift
        self.mid_price = self.get_mid_price()
        drift = np.random.normal(
            0, self.config.volatility_annual / np.sqrt(252 * 390)
        )
        self.mid_price += self.mid_price * drift

        # Replenish depth if the book has become too thin
        if len(self.bids) < 3 or len(self.asks) < 3:
            self._initialize_book()

    def get_state_vector(self) -> np.ndarray:
        """
        Summarise the current order book state as an 8-element feature vector:
        [mid_price, spread, bid_depth, ask_depth, imbalance, toxicity,
         best_bid, best_ask]
        """
        best_bid, _ = self.get_best_bid()
        best_ask, _ = self.get_best_ask()
        return np.array(
            [
                self.mid_price,
                self.get_spread(),
                self.get_depth("bid"),
                self.get_depth("ask"),
                self.get_order_imbalance(),
                self.get_order_flow_toxicity(),
                best_bid,
                best_ask,
            ],
            dtype=np.float32,
        )


# ============================================================================
# MARKET MAKING ENVIRONMENT (OpenAI GYM)
# ============================================================================


class MarketMakingEnv(gym.Env):
    """
    OpenAI Gym environment for RL-based market making.

    State Space (12-dimensional):
        Order book features: mid_price, spread, bid_depth, ask_depth,
                             imbalance, toxicity, best_bid, best_ask
        Agent state:         normalized_inventory, cash_ratio, pnl_ratio,
                             time_remaining

    Action Space (continuous, Box):
        [bid_offset, ask_offset] -- tick offsets from mid price.
        bid_offset in [-10, 0],  ask_offset in [0, 10]

    Reward (Avellaneda-Stoikov inspired):
        Per-step PnL change (dominant signal) + quadratic inventory penalty.
        The agent learns that capturing spread and getting fills is valuable
        because it directly improves PnL, not because of extrinsic bonuses.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        cfg: MarketMakerConfig,
        market_data: Optional[pd.DataFrame] = None,
    ):
        super().__init__()

        self.config = cfg
        self.market_data = market_data

        # Continuous action space: bid and ask quote offsets in ticks
        self.action_space = spaces.Box(
            low=np.array([-10.0, 0.0], dtype=np.float32),
            high=np.array([0.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: 8 order-book features + 4 agent-state features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32,
        )

        # State variables (initialised in reset)
        self.order_book: Optional[LimitOrderBook] = None
        self.current_step = 0
        self.inventory = 0
        self.cash = cfg.initial_cash
        self.initial_cash = cfg.initial_cash
        self.pnl_history: List[float] = []
        self.fill_history: List[int] = []
        self.trades_history: List[Dict] = []
        self.step_fills: int = 0
        self.active_bid_price: Optional[float] = None
        self.active_ask_price: Optional[float] = None
        self.active_bid_size = cfg.lot_size
        self.active_ask_size = cfg.lot_size

    def reset(self, seed=None, options=None):
        """Reset environment to its initial state and return the first observation."""
        super().reset(seed=seed)

        initial_price = (
            self.market_data.iloc[0]["close"]
            if self.market_data is not None and len(self.market_data) > 0
            else self.config.initial_price
        )

        self.order_book = LimitOrderBook(self.config, initial_price)
        self.current_step = 0
        self.inventory = 0
        self.cash = self.config.initial_cash
        self.initial_cash = self.config.initial_cash
        self.pnl_history = [0.0]
        self.fill_history = []
        self.trades_history = []
        self.step_fills = 0
        self.active_bid_price = None
        self.active_ask_price = None

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Build the 12-element observation vector for the current time step."""
        ob_state = self.order_book.get_state_vector()

        mid_price = self.order_book.get_mid_price()
        pnl = (self.cash + self.inventory * mid_price) - self.initial_cash
        normalized_inventory = self.inventory / self.config.max_inventory
        time_remaining = (
            self.config.episode_length - self.current_step
        ) / self.config.episode_length

        return np.concatenate(
            [
                ob_state,
                [
                    normalized_inventory,
                    self.cash / self.initial_cash,
                    pnl / self.initial_cash,
                    time_remaining,
                ],
            ]
        ).astype(np.float32)

    def _calculate_market_impact(self, size: int) -> float:
        """
        Estimate market impact cost using a simplified square-root model.

        Args:
            size: Order size in shares

        Returns:
            Impact cost in dollars
        """
        daily_volume = 1_000_000
        volatility = self.config.volatility_annual / np.sqrt(252)
        impact = (
            self.config.permanent_impact_coef
            * volatility
            * np.sqrt(size / daily_volume)
        )
        return impact * self.order_book.mid_price

    def _execute_trade(self, side: str, price: float, size: int):
        """
        Record a trade and update inventory, cash, and trade history.

        Args:
            side: 'buy' or 'sell'
            price: Execution price
            size: Quantity traded
        """
        impact = self._calculate_market_impact(size)
        fee = (self.config.fee_bps / 10000) * price * size
        slippage = (self.config.slippage_bps / 10000) * price * size
        total_cost = impact + fee + slippage

        if side == "buy":
            self.inventory += size
            self.cash -= price * size + total_cost
        else:
            self.inventory -= size
            self.cash += price * size - total_cost

        self.trades_history.append(
            {
                "step": self.current_step,
                "side": side,
                "price": price,
                "size": size,
                "cost": total_cost,
            }
        )
        self.fill_history.append(1)
        self.step_fills += 1

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Advance the environment by one time step.

        Args:
            action: [bid_offset, ask_offset] array (in ticks from mid)

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        bid_offset, ask_offset = action
        mid_price = self.order_book.get_mid_price()

        # Set and clamp quotes
        self.active_bid_price = np.clip(
            mid_price + bid_offset * self.config.tick_size,
            mid_price - 1.0,
            mid_price,
        )
        self.active_ask_price = np.clip(
            mid_price + ask_offset * self.config.tick_size,
            mid_price,
            mid_price + 1.0,
        )

        # Reset per-step fill counter
        self.step_fills = 0

        # Advance background order flow
        self.order_book.simulate_order_flow()

        best_bid, _ = self.order_book.get_best_bid()
        best_ask, _ = self.order_book.get_best_ask()

        # Immediate (aggressive) fill check with inventory guard
        if (self.active_bid_price >= best_ask
                and abs(self.inventory + self.active_bid_size) <= self.config.max_inventory):
            self._execute_trade(
                "buy", self.active_bid_price, self.active_bid_size
            )
        if (self.active_ask_price <= best_bid
                and abs(self.inventory - self.active_ask_size) <= self.config.max_inventory):
            self._execute_trade(
                "sell", self.active_ask_price, self.active_ask_size
            )

        # Probabilistic passive fill based on proximity to mid
        spread = self.order_book.get_spread()
        if spread > 0:
            bid_prob = max(
                0, 1.0 - abs(self.active_bid_price - mid_price) / spread
            )
            ask_prob = max(
                0, 1.0 - abs(self.active_ask_price - mid_price) / spread
            )
            if (np.random.random() < bid_prob * 0.3
                    and abs(self.inventory + self.active_bid_size) <= self.config.max_inventory):
                self._execute_trade(
                    "buy", self.active_bid_price, self.active_bid_size
                )
            if (np.random.random() < ask_prob * 0.3
                    and abs(self.inventory - self.active_ask_size) <= self.config.max_inventory):
                self._execute_trade(
                    "sell", self.active_ask_price, self.active_ask_size
                )

        # Compute current mark-to-market PnL
        mid_price = self.order_book.get_mid_price()
        current_pnl = (
            self.cash + self.inventory * mid_price
        ) - self.initial_cash

        # Per-step PnL change for reward calculation
        prev_pnl = self.pnl_history[-1]
        pnl_delta = current_pnl - prev_pnl
        reward = self._calculate_reward(pnl_delta)

        self.pnl_history.append(current_pnl)
        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.config.episode_length

        # Hard inventory breach -- forced liquidation penalty
        if abs(self.inventory) > self.config.max_inventory:
            reward -= 100.0
            terminated = True

        # Terminal inventory penalty -- penalize holding positions at episode end
        if truncated and abs(self.inventory) > 0:
            liquidation_cost = abs(self.inventory) * mid_price * 5.0 / 10000
            reward -= liquidation_cost / 100.0

        info = {
            "pnl": current_pnl,
            "inventory": self.inventory,
            "cash": self.cash,
            "mid_price": mid_price,
            "spread": self.order_book.get_spread(),
            "fills": len(self.fill_history),
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _calculate_reward(self, pnl_delta: float) -> float:
        """
        Avellaneda-Stoikov inspired reward function.

        The reward is dominated by actual PnL change, with a quadratic
        inventory penalty to discourage directional risk accumulation.

        Components:
          1. pnl_delta     -- raw dollar PnL change per step (dominant signal)
          2. inv_penalty   -- quadratic cost for inventory exposure
        """
        # (1) PnL change in dollars (the primary signal)
        pnl_reward = pnl_delta

        # (2) Inventory penalty (Avellaneda-Stoikov style, quadratic)
        inv_penalty = -0.0001 * (self.inventory / self.config.lot_size) ** 2
        fill_incentive = 0.1 * self.step_fills
        return pnl_reward + inv_penalty + fill_incentive

    def render(self, mode="human"):
        """Print a brief state summary to stdout."""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Mid Price: ${self.order_book.get_mid_price():.2f}")
            print(f"Spread: ${self.order_book.get_spread():.4f}")
            print(f"Inventory: {self.inventory}")
            print(f"Cash: ${self.cash:.2f}")
            print(f"PnL: ${self.pnl_history[-1]:.2f}")
            print(f"Fills: {len(self.fill_history)}")
            print("-" * 50)


# ============================================================================
# PRE-TRAINED MODEL LOADER
# ============================================================================


def load_pretrained_model(
    cfg: MarketMakerConfig,
) -> Tuple[Optional[Any], Optional[Dict]]:
    """
    Load the pre-trained PPO model and training metrics from the repository.

    The model weights were trained on a GPU using Google Colab and
    uploaded to this Hugging Face Space alongside app.py.

    Returns:
        (model, metrics) tuple. Either may be None if the file is missing.
    """
    model = None
    metrics = None

    # Load PPO model weights
    model_path = cfg.pretrained_model_path
    if os.path.exists(model_path):
        try:
            dummy_env = DummyVecEnv(
                [lambda: MarketMakingEnv(cfg)]
            )
            model = PPO.load(
                model_path,
                env=dummy_env,
                device="cpu",
            )
            logger.info(f"Pre-trained PPO model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
    else:
        logger.warning(
            f"Model file not found at {model_path}. "
            "Upload ppo_market_maker.zip to the repository."
        )

    # Load training metrics
    metrics_path = cfg.training_metrics_path
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            logger.info(f"Training metrics loaded from {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to load metrics from {metrics_path}: {e}")
    else:
        logger.warning(
            f"Metrics file not found at {metrics_path}. "
            "Upload training_metrics.json to the repository."
        )

    return model, metrics


def normalize_metrics_drawdown(
    metrics: Dict, initial_cash: float
) -> Tuple[float, float]:
    """
    Normalize the max_drawdown field from training metrics into a consistent
    (percentage, dollar) pair regardless of which training script version
    produced the metrics file.

    Args:
        metrics: Raw training metrics dictionary
        initial_cash: Starting capital used during training

    Returns:
        (max_drawdown_pct, max_drawdown_dollars) where pct is a fraction [0, 1]
    """
    md_raw = metrics.get("max_drawdown", 0)
    md_dollars_key = metrics.get("max_drawdown_dollars", None)

    if md_dollars_key is not None:
        # Metrics file already contains both fields (v3 format)
        return md_raw, md_dollars_key

    if md_raw > 1.0:
        # Raw dollar value stored directly as max_drawdown (v2 format)
        return md_raw / initial_cash, md_raw

    # Already a fraction (v3 format without the dollars key)
    return md_raw, md_raw * initial_cash


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_agent(
    model,
    cfg: MarketMakerConfig,
    market_data: Optional[pd.DataFrame] = None,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """
    Roll out the trained agent for n_episodes and compute performance statistics.

    Args:
        model: Trained PPO model
        cfg: Configuration
        market_data: Optional historical price data
        n_episodes: Number of rollout episodes

    Returns:
        Dictionary of performance metrics
    """
    logger.info(f"Evaluating agent over {n_episodes} episodes...")
    env = MarketMakingEnv(cfg, market_data)

    all_pnls: List[float] = []
    all_inventories: List[int] = []
    all_fills: List[int] = []
    all_spreads: List[float] = []
    episode_returns: List[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reward += reward
            all_inventories.append(info["inventory"])
            all_spreads.append(info["spread"])

        all_pnls.extend(env.pnl_history)
        all_fills.append(len(env.fill_history))
        episode_returns.append(ep_reward)

    pnl_array = np.array(all_pnls)
    returns_array = np.diff(pnl_array)

    sharpe = (
        np.mean(returns_array) / np.std(returns_array) * np.sqrt(252 * 390)
        if len(returns_array) > 0 and np.std(returns_array) > 0
        else 0.0
    )

    # Drawdown as percentage of initial capital
    peak = np.maximum.accumulate(pnl_array)
    drawdown_dollars = peak - pnl_array
    max_dd_dollars = float(np.max(drawdown_dollars)) if len(drawdown_dollars) > 0 else 0.0
    max_dd_pct = max_dd_dollars / cfg.initial_cash

    total_steps = len(all_inventories)
    fill_rate = (
        sum(all_fills) / (total_steps * 2) if total_steps > 0 else 0.0
    )

    metrics = {
        "mean_pnl": float(np.mean(all_pnls)),
        "final_pnl": float(pnl_array[-1]) if len(pnl_array) > 0 else 0.0,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_dd_pct,
        "max_drawdown_dollars": max_dd_dollars,
        "fill_rate": fill_rate,
        "mean_spread": float(np.mean(all_spreads)),
        "mean_inventory": float(np.mean(np.abs(all_inventories))),
        "mean_episode_return": float(np.mean(episode_returns)),
        "std_episode_return": float(np.std(episode_returns)),
    }

    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def _empty_figure(title: str = "") -> go.Figure:
    """
    Return a blank Plotly figure with an optional title annotation.
    Used as a safe placeholder for gr.Plot outputs that have no data yet,
    preventing Gradio from rendering a broken-plot icon.
    """
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "No data yet -- run the action above first.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14, "color": "gray"},
            }
        ],
        height=400,
    )
    return fig


def plot_order_book_heatmap(order_book: LimitOrderBook) -> go.Figure:
    """
    Render a horizontal bar chart showing current order book depth.

    Args:
        order_book: Live LimitOrderBook instance

    Returns:
        Plotly figure
    """
    bid_levels = sorted(order_book.bids.items(), reverse=True)[:20]
    ask_levels = sorted(order_book.asks.items())[:20]

    prices, sizes, sides = [], [], []
    for price, size in bid_levels:
        prices.append(price)
        sizes.append(size)
        sides.append("Bid")
    for price, size in ask_levels:
        prices.append(price)
        sizes.append(size)
        sides.append("Ask")

    df = pd.DataFrame({"Price": prices, "Size": sizes, "Side": sides})

    fig = go.Figure()

    bid_df = df[df["Side"] == "Bid"]
    fig.add_trace(
        go.Bar(
            x=bid_df["Size"],
            y=bid_df["Price"],
            orientation="h",
            name="Bids",
            marker=dict(color="green", opacity=0.6),
        )
    )

    ask_df = df[df["Side"] == "Ask"]
    fig.add_trace(
        go.Bar(
            x=ask_df["Size"],
            y=ask_df["Price"],
            orientation="h",
            name="Asks",
            marker=dict(color="red", opacity=0.6),
        )
    )

    mid = order_book.get_mid_price()
    fig.add_hline(
        y=mid,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Mid: ${mid:.2f}",
    )

    fig.update_layout(
        title="Live Order Book Depth",
        xaxis_title="Size (shares)",
        yaxis_title="Price ($)",
        barmode="overlay",
        height=600,
        hovermode="y unified",
    )
    return fig


def plot_training_curves(training_history: List[Dict]) -> go.Figure:
    """
    Plot mean episode reward with standard deviation band from training history.

    Args:
        training_history: List of dicts with step, mean_reward, std_reward

    Returns:
        Plotly figure
    """
    if not training_history:
        return _empty_figure("Training Progress")

    steps = [h["step"] for h in training_history]
    means = np.array([h["mean_reward"] for h in training_history])
    stds = np.array([h["std_reward"] for h in training_history])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=means,
            mode="lines",
            name="Mean Reward",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps + steps[::-1],
            y=list(means + stds) + list((means - stds)[::-1]),
            fill="toself",
            fillcolor="rgba(0,100,255,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="+/- 1 Std Dev",
        )
    )
    fig.update_layout(
        title="Training Progress: Mean Episode Reward",
        xaxis_title="Training Steps",
        yaxis_title="Mean Reward",
        height=500,
        hovermode="x unified",
    )
    return fig


def plot_pnl_curve(pnl_history: List[float]) -> go.Figure:
    """
    Plot the cumulative mark-to-market PnL trajectory.

    Args:
        pnl_history: List of PnL values, one per time step

    Returns:
        Plotly figure
    """
    if not pnl_history:
        return _empty_figure("Cumulative Profit & Loss")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=pnl_history,
            mode="lines",
            name="PnL",
            line=dict(color="green", width=2),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Cumulative Profit & Loss",
        xaxis_title="Time Step",
        yaxis_title="PnL ($)",
        height=500,
        hovermode="x unified",
    )
    return fig


def plot_inventory_risk(inventory_history: List[int]) -> go.Figure:
    """
    Plot inventory exposure over time.

    Args:
        inventory_history: List of inventory levels, one per time step

    Returns:
        Plotly figure
    """
    if not inventory_history:
        return _empty_figure("Inventory Risk Over Time")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=inventory_history,
            mode="lines",
            name="Inventory",
            line=dict(color="orange", width=2),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Inventory Risk Over Time",
        xaxis_title="Time Step",
        yaxis_title="Inventory (shares)",
        height=500,
        hovermode="x unified",
    )
    return fig


def create_performance_dashboard(metrics: Dict[str, float]) -> go.Figure:
    """
    Render a 2x2 gauge/number dashboard for the key performance metrics.

    Args:
        metrics: Dictionary containing sharpe_ratio, max_drawdown,
                 fill_rate, and mean_pnl keys

    Returns:
        Plotly figure with four indicator subplots
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Sharpe Ratio", "Max Drawdown", "Fill Rate", "Mean PnL"
        ),
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "indicator"}, {"type": "indicator"}],
        ],
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["sharpe_ratio"],
            title={"text": "Sharpe Ratio"},
            delta={"reference": 1.0},
            gauge={
                "axis": {"range": [-3, 3]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [-3, 0], "color": "#ffcccc"},
                    {"range": [0, 1], "color": "lightgray"},
                    {"range": [1, 2], "color": "#ccffcc"},
                    {"range": [2, 3], "color": "#88ff88"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 2.0,
                },
            },
        ),
        row=1, col=1,
    )

    # Max drawdown as percentage of initial capital (0-25% gauge range)
    max_dd_display = metrics["max_drawdown"] * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=max_dd_display,
            title={"text": "Max Drawdown (%)"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 25]},
                "bar": {"color": "red"},
                "steps": [
                    {"range": [0, 5], "color": "lightgreen"},
                    {"range": [5, 10], "color": "yellow"},
                    {"range": [10, 25], "color": "#ffcccc"},
                ],
            },
        ),
        row=1, col=2,
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metrics["fill_rate"] * 100,
            title={"text": "Fill Rate (%)"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "green"},
                "steps": [
                    {"range": [0, 30], "color": "#ffcccc"},
                    {"range": [30, 60], "color": "lightgray"},
                    {"range": [60, 100], "color": "#ccffcc"},
                ],
            },
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=metrics["mean_pnl"],
            title={"text": "Mean PnL ($)"},
            number={"prefix": "$"},
            delta={"reference": 0},
        ),
        row=2, col=2,
    )

    fig.update_layout(height=600, title_text="Performance Metrics Dashboard")
    return fig


# ============================================================================
# GRADIO INTERFACE
# ============================================================================


def create_gradio_interface():
    """
    Build and return the Gradio Blocks interface for the RL Market Maker.

    Tabs:
        1. Training Summary     -- pre-loaded training metrics and curves
        2. Live Simulation      -- run a post-training rollout
        3. Order Book Analytics -- real-time order book snapshot
        4. Performance Analytics -- full evaluation dashboard
        5. Documentation        -- inline technical reference
    """
    # Load pre-trained model and metrics at interface creation time
    pretrained_model, pretrained_metrics = load_pretrained_model(config)

    # Shared in-memory state across all tab callbacks
    global_state: Dict[str, Any] = {
        "model": pretrained_model,
        "training_metrics": pretrained_metrics,
        "market_data": None,
    }

    # Generate default market data for simulations
    fetcher = MarketDataFetcher(config)
    global_state["market_data"] = fetcher.generate_synthetic_data(1000)

    # ------------------------------------------------------------------ #
    # Callback: Live Simulation                                            #
    # ------------------------------------------------------------------ #
    def run_simulation_wrapper(n_steps):
        """
        Roll out the trained agent for n_steps and produce four output values:
        summary text, PnL figure, inventory figure, order-book figure.
        """
        if global_state["model"] is None:
            return (
                "Model not loaded. Please upload ppo_market_maker.zip "
                "to the repository.",
                _empty_figure("Cumulative Profit & Loss"),
                _empty_figure("Inventory Risk Over Time"),
                _empty_figure("Live Order Book Depth"),
            )

        try:
            model = global_state["model"]
            market_data = global_state["market_data"]

            env = MarketMakingEnv(config, market_data)
            obs, _ = env.reset()

            pnl_history = [0.0]
            inventory_history = [0]

            for _ in range(int(n_steps)):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                pnl_history.append(info["pnl"])
                inventory_history.append(info["inventory"])
                if term or trunc:
                    obs, _ = env.reset()

            pnl_fig = plot_pnl_curve(pnl_history)
            inventory_fig = plot_inventory_risk(inventory_history)
            ob_fig = plot_order_book_heatmap(env.order_book)

            returns = np.diff(pnl_history)
            sharpe = (
                np.mean(returns) / np.std(returns) * np.sqrt(252 * 390)
                if np.std(returns) > 0
                else 0.0
            )

            # Count buys and sells for diagnostic
            buy_count = sum(1 for t in env.trades_history if t["side"] == "buy")
            sell_count = sum(1 for t in env.trades_history if t["side"] == "sell")

            summary = f"""
## Simulation Results

**Final PnL:** ${pnl_history[-1]:.2f}
**Sharpe Ratio:** {sharpe:.2f}
**Final Inventory:** {inventory_history[-1]} shares
**Total Steps:** {len(pnl_history) - 1}
**Fill Rate:** {len(env.fill_history) / max((len(pnl_history) - 1) * 2, 1) * 100:.1f}%
**Trades:** {buy_count} buys, {sell_count} sells
"""
            return summary, pnl_fig, inventory_fig, ob_fig

        except Exception as e:
            logger.error(f"Simulation error: {e}")
            import traceback
            return (
                f"Error during simulation: {str(e)}\n{traceback.format_exc()}",
                _empty_figure("Cumulative Profit & Loss"),
                _empty_figure("Inventory Risk Over Time"),
                _empty_figure("Live Order Book Depth"),
            )

    # ------------------------------------------------------------------ #
    # Callback: Order Book Refresh                                         #
    # ------------------------------------------------------------------ #
    def refresh_order_book():
        """Instantiate a fresh environment, warm it up, and snapshot the book."""
        try:
            market_data = global_state["market_data"]
            env = MarketMakingEnv(config, market_data)
            env.reset()

            # Run a short warm-up to populate a realistic book state
            for _ in range(10):
                env.order_book.simulate_order_flow()

            fig = plot_order_book_heatmap(env.order_book)

            metrics_text = f"""
### Current Metrics
- **Mid Price:** ${env.order_book.get_mid_price():.2f}
- **Spread:** ${env.order_book.get_spread():.4f}
- **Imbalance:** {env.order_book.get_order_imbalance():.3f}
- **Toxicity:** {env.order_book.get_order_flow_toxicity():.3f}
"""
            return metrics_text, fig

        except Exception as e:
            logger.error(f"Order book refresh error: {e}")
            return (
                f"Error: {str(e)}",
                _empty_figure("Live Order Book Depth"),
            )

    # ------------------------------------------------------------------ #
    # Callback: Evaluate Model                                             #
    # ------------------------------------------------------------------ #
    def evaluate_model_wrapper(n_episodes):
        """Run a full evaluation rollout and render the performance dashboard."""
        if global_state["model"] is None:
            return (
                "Model not loaded. Please upload ppo_market_maker.zip "
                "to the repository.",
                _empty_figure("Performance Dashboard"),
            )

        try:
            metrics = evaluate_agent(
                global_state["model"],
                config,
                global_state["market_data"],
                int(n_episodes),
            )
            dashboard_fig = create_performance_dashboard(metrics)

            dd_dollars = metrics.get(
                "max_drawdown_dollars", metrics["max_drawdown"] * config.initial_cash
            )

            summary = f"""
## Evaluation Results ({int(n_episodes)} episodes)

### Profitability
- **Mean PnL:** ${metrics['mean_pnl']:.2f}
- **Final PnL:** ${metrics['final_pnl']:.2f}
- **Sharpe Ratio:** {metrics['sharpe_ratio']:.3f}

### Risk Management
- **Max Drawdown:** {metrics['max_drawdown'] * 100:.2f}% (${dd_dollars:.2f})
- **Mean Inventory:** {metrics['mean_inventory']:.0f} shares

### Execution Quality
- **Fill Rate:** {metrics['fill_rate'] * 100:.1f}%
- **Mean Spread:** ${metrics['mean_spread']:.4f}

### Episode Performance
- **Mean Episode Return:** {metrics['mean_episode_return']:.2f}
- **Std Episode Return:** {metrics['std_episode_return']:.2f}
"""
            return summary, dashboard_fig

        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return (
                f"Error during evaluation: {str(e)}",
                _empty_figure("Performance Dashboard"),
            )

    # ------------------------------------------------------------------ #
    # UI Layout                                                            #
    # ------------------------------------------------------------------ #
    with gr.Blocks(
        title="Deep RL Market Maker", theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown(
            """
# Deep Reinforcement Learning Market Maker

**Production-Grade HFT System with Order Book Dynamics**

This system demonstrates cutting-edge Deep RL (PPO) for optimal market making
with realistic order book simulation, market microstructure modeling, and comprehensive analytics.
The model was trained on a GPU via Google Colab and deployed here for inference on CPU.

---
"""
        )

        with gr.Tabs():

            # ---------------------------------------------------------- #
            # Tab 1: Training Summary                                     #
            # ---------------------------------------------------------- #
            with gr.TabItem("Training Summary"):
                gr.Markdown(
                    """
### Pre-Trained Model Overview

The PPO agent was trained on a GPU using Google Colab Pro.
Below are the training metrics and reward curve from the training session.
"""
                )

                if pretrained_metrics:
                    ts = pretrained_metrics.get("training_timesteps", "N/A")
                    tt = pretrained_metrics.get("training_time_seconds", 0)
                    gpu = pretrained_metrics.get("gpu", "N/A")
                    sr = pretrained_metrics.get("sharpe_ratio", 0)
                    mp = pretrained_metrics.get("mean_pnl", 0)
                    fp = pretrained_metrics.get("final_pnl", 0)
                    ic = pretrained_metrics.get("initial_cash", config.initial_cash)

                    # Normalize drawdown regardless of metrics file version
                    md_pct, md_dollars = normalize_metrics_drawdown(
                        pretrained_metrics, ic
                    )

                    fr = pretrained_metrics.get("fill_rate", 0)
                    ms = pretrained_metrics.get("mean_spread", 0)
                    mi = pretrained_metrics.get("mean_inventory", 0)
                    mer = pretrained_metrics.get("mean_episode_return", 0)

                    training_summary_md = f"""
### Training Configuration
- **Total Timesteps:** {ts:,}
- **Training Time:** {tt / 60:.1f} minutes
- **Hardware:** {gpu}
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Reward Function:** Avellaneda-Stoikov inspired (PnL change + inventory penalty)
- **Environment:** Custom Market Making Gym (12-D obs, 2-D continuous action)

### Evaluation Results (10 episodes)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | {sr:.3f} |
| Mean PnL | ${mp:.2f} |
| Final PnL | ${fp:.2f} |
| Max Drawdown | {md_pct * 100:.2f}% (${md_dollars:.2f}) |
| Fill Rate | {fr * 100:.1f}% |
| Mean Spread | ${abs(ms):.4f} |
| Mean Inventory | {mi:.0f} shares |
| Mean Episode Return | {mer:.2f} |
"""
                    gr.Markdown(training_summary_md)

                    history = pretrained_metrics.get("training_history", [])
                    if history:
                        fig = plot_training_curves(history)
                        gr.Plot(value=fig, label="Training Progress")
                    else:
                        gr.Markdown("*No training curve data available.*")
                else:
                    gr.Markdown(
                        """
**training_metrics.json not found.**

Please upload the training metrics file to the repository
alongside ppo_market_maker.zip.
"""
                    )

                if pretrained_model is not None:
                    gr.Markdown(
                        "**Model Status:** Loaded and ready for inference."
                    )
                else:
                    gr.Markdown(
                        "**Model Status:** Not loaded. Upload "
                        "ppo_market_maker.zip to the repository."
                    )

            # ---------------------------------------------------------- #
            # Tab 2: Live Simulation                                      #
            # ---------------------------------------------------------- #
            with gr.TabItem("Live Simulation"):
                gr.Markdown(
                    """
### Run Live Market Making Simulation

Test the trained agent in a realistic market environment with:
- Real-time order book dynamics
- Market impact modeling
- Transaction costs
- Inventory constraints
"""
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        sim_steps = gr.Slider(
                            minimum=100,
                            maximum=5000,
                            value=1000,
                            step=100,
                            label="Simulation Steps",
                            info="Each step = 1 second of market time",
                        )
                        simulate_button = gr.Button(
                            "Run Simulation",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=2):
                        sim_summary = gr.Markdown(
                            "Waiting for simulation..."
                        )

                with gr.Row():
                    pnl_plot = gr.Plot(label="Cumulative PnL")
                    inventory_plot = gr.Plot(label="Inventory Risk")

                with gr.Row():
                    order_book_plot = gr.Plot(label="Order Book Depth")

                simulate_button.click(
                    fn=run_simulation_wrapper,
                    inputs=[sim_steps],
                    outputs=[
                        sim_summary,
                        pnl_plot,
                        inventory_plot,
                        order_book_plot,
                    ],
                )

            # ---------------------------------------------------------- #
            # Tab 3: Order Book Analytics                                 #
            # ---------------------------------------------------------- #
            with gr.TabItem("Order Book Analytics"):
                gr.Markdown(
                    """
### Real-Time Order Book Visualization

Features:
- **Bid/Ask Depth:** Volume at each price level
- **Spread Analysis:** Current bid-ask spread
- **Order Imbalance:** Buy vs. sell pressure
- **Order Flow Toxicity:** Adverse selection risk (VPIN approximation)
"""
                )

                refresh_ob = gr.Button(
                    "Refresh Order Book", variant="secondary"
                )

                with gr.Row():
                    with gr.Column():
                        ob_metrics = gr.Markdown(
                            """
### Current Metrics
- Mid Price: --
- Spread: --
- Imbalance: --
- Toxicity: --
"""
                        )
                    with gr.Column():
                        ob_heatmap = gr.Plot(label="Order Book Heatmap")

                refresh_ob.click(
                    fn=refresh_order_book,
                    outputs=[ob_metrics, ob_heatmap],
                )

            # ---------------------------------------------------------- #
            # Tab 4: Performance Analytics                                #
            # ---------------------------------------------------------- #
            with gr.TabItem("Performance Analytics"):
                gr.Markdown(
                    """
### Comprehensive Performance Evaluation

Metrics:
- **PnL:** Profit and loss over time
- **Sharpe Ratio:** Risk-adjusted returns (annualized)
- **Max Drawdown:** Worst peak-to-trough decline (% of initial capital)
- **Fill Rate:** Percentage of quotes executed
- **Inventory Risk:** Position size over time
"""
                )

                eval_episodes = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=10,
                    step=5,
                    label="Number of Evaluation Episodes",
                )
                evaluate_button = gr.Button(
                    "Evaluate Model", variant="primary", size="lg"
                )
                eval_summary = gr.Markdown("Waiting for evaluation...")
                eval_dashboard = gr.Plot(
                    label="Performance Dashboard"
                )

                evaluate_button.click(
                    fn=evaluate_model_wrapper,
                    inputs=[eval_episodes],
                    outputs=[eval_summary, eval_dashboard],
                )

            # ---------------------------------------------------------- #
            # Tab 5: Documentation                                        #
            # ---------------------------------------------------------- #
            with gr.TabItem("Documentation"):
                gr.Markdown(
                    """
## System Documentation

### Overview

This is a production-grade Deep Reinforcement Learning system for automated market making
in high-frequency trading environments. It demonstrates cutting-edge ML/RL techniques
applied to financial market microstructure.

### Key Components

#### 1. Order Book Simulator
- **Realistic limit order book** with multiple price levels per side
- **Poisson arrival process** for background order flow
- **Market impact modeling** using the square-root function
- **Order flow toxicity** tracking (VPIN-based approximation)
- **Book uncrossing logic** to prevent bid-ask inversions

#### 2. Deep RL Agent

**PPO (Proximal Policy Optimization):**
- Policy gradient method with clipped surrogate objective
- Advantage estimation via GAE (lambda = 0.95)
- Entropy regularization for continued exploration
- Trained on GPU, deployed on CPU

**Reward Function (Avellaneda-Stoikov inspired):**
- Per-step PnL change in dollars (dominant signal)
- Quadratic inventory penalty to encourage mean reversion
- Terminal inventory penalty at episode end
- The agent learns that capturing spread and getting fills is valuable
  because it directly improves PnL

#### 3. Market Microstructure Features

**Order Book Features:**
- Bid-ask spread
- Order book depth (top 5 levels per side)
- Order imbalance ratio
- Order flow toxicity (VPIN approximation)

**Agent State:**
- Normalized inventory position
- Cash ratio (current cash / initial cash)
- Mark-to-market PnL ratio
- Fraction of episode remaining

#### 4. Cost and Risk Models

**Transaction Costs:**
- Fee: 1 bps per trade
- Slippage: 0.5 bps
- Market impact: square-root model
- Adverse selection component

**Risk Management:**
- Hard inventory limit: 1,000 shares
- Quadratic inventory penalty in reward
- Inventory-aware trade execution (rejects trades that would breach limits)
- Terminal liquidation penalty
- Forced termination if limit breached

### Performance Benchmarks

**Expected Metrics (after convergence):**
- Sharpe Ratio: 0.5 -- 2.0
- Fill Rate: 40% -- 80%
- Spread Capture: 30% -- 60%
- Max Drawdown: < 5% of initial capital

### API Integration

**Polygon.io:** Real-time market data, level-1 quotes, historical minute bars
**Alpha Vantage:** Intraday data (1 min, 5 min intervals)
**Fallback:** Synthetic GBM data requires no API key

### Limitations and Assumptions

1. Simplified market impact -- square-root model, not full LOB impact
2. No partial fills -- orders execute atomically
3. Perfect execution -- no queue-priority or time-priority modeling
4. Synthetic background flow -- not calibrated to live microstructure
5. No news or event risk -- information-driven jumps not modeled
6. Single-asset scope

### Failure Modes

1. Adverse selection in strong trending regimes
2. Inventory buildup under persistent one-sided flow
3. Reduced profitability in very low-volatility environments
4. Flash-crash events outside the training distribution
5. API rate limits may affect real-time data feeds

### Hardware Requirements

| Tier          | GPU            | RAM   | Notes               |
|---------------|----------------|-------|---------------------|
| Training      | A100/H100      | 64 GB | ~10-15 minutes      |
| Inference     | CPU only       | 16 GB | This deployment     |

### Reproducibility

All random seeds are fixed to 42:
```
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```
Deterministic CUDA mode is also enabled when a GPU is present.

### References

1. Avellaneda & Stoikov (2008) -- High-frequency trading in a limit order book
2. Schulman et al. (2017) -- Proximal Policy Optimization Algorithms
3. van Hasselt et al. (2016) -- Deep RL with Double Q-learning
4. Easley et al. (2012) -- Flow toxicity and liquidity in a high-frequency world
5. Cartea et al. (2015) -- Algorithmic and High-Frequency Trading

### License and Disclaimer

**License:** MIT

**Disclaimer:** Educational and research use only. Real trading involves significant
risks and regulatory requirements not covered here. Past performance does not
guarantee future results.

### Author

**Spencer Purdy** -- portfolio demonstration of HFT systems, deep reinforcement
learning, market microstructure modeling, and production ML engineering.

---
**System Version:** 1.0.0 | **Last Updated:** February 2026
"""
                )

        gr.Markdown(
            """
---
**Deep RL Market Maker v1.0.0** | Built with PyTorch, Stable-Baselines3, Gradio | Author: Spencer Purdy

Demonstrates: Deep RL (PPO), HFT, Market Microstructure, Order Book Dynamics, Real-time Trading Simulation
"""
        )

    return interface


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info(
        "Deep Reinforcement Learning Market Maker with Order Book Dynamics"
    )
    logger.info("Author: Spencer Purdy")
    logger.info("=" * 80)

    logger.info("Creating Gradio interface...")
    interface = create_gradio_interface()

    logger.info("Launching application...")
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )