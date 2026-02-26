# Deep Reinforcement Learning Market Maker with Order Book Dynamics

A production-grade reinforcement learning system for automated market making in high-frequency trading environments. A PPO-trained agent continuously quotes bid and ask prices while managing inventory risk, capturing bid-ask spread, and adapting to real-time order book dynamics. The system demonstrates end-to-end deep RL applied to financial market microstructure.

## About

This portfolio project showcases applied deep reinforcement learning in a quantitative finance context. The agent operates within a realistic limit order book simulation featuring Poisson-driven order flow, square-root market impact, and VPIN-based toxicity estimation — core concepts from the market microstructure literature. The pre-trained model is served via Gradio with real-time visualization and performance analytics.

**Author:** Spencer Purdy  
**Development Environment:** Google Colab Pro (A100 GPU, High RAM)

## Features

- **Pre-Trained PPO Agent**: Loaded at startup, trained for 1,500,000 timesteps on GPU
- **Limit Order Book Simulation**: Realistic multi-level order book with Poisson order arrival process
- **Market Microstructure Modeling**: Spread, depth, order imbalance, and VPIN-based flow toxicity
- **Market Impact**: Square-root permanent and temporary impact components with adverse selection costs
- **Avellaneda-Stoikov Reward**: PnL change with quadratic inventory penalty for mean reversion
- **Risk Management**: Hard inventory limits, terminal liquidation penalty, forced episode termination on breach
- **API Integration**: Polygon.io and Alpha Vantage for real market data; GBM synthetic fallback requires no key
- **Interactive Interface**: Gradio web application with live trading simulation, order book heatmap, and performance dashboard
- **Comprehensive Evaluation**: PnL, Sharpe ratio, max drawdown, fill rate, and inventory risk across configurable evaluation episodes

## Environment Design

- **Observation Space:** 12-dimensional continuous vector
- **Action Space:** 2-dimensional continuous — [bid_offset, ask_offset] in ticks from mid price
- **Episode Length:** 1,000 steps (1 step = 1 simulated second)
- **Initial Capital:** $100,000
- **Tick Size:** $0.01 | **Lot Size:** 100 shares | **Max Inventory:** 1,000 shares

## Model Architecture

| Component | Details |
|-----------|---------|
| Algorithm | Proximal Policy Optimization (PPO) |
| Policy Network | MLP (64 → 64), tanh activation |
| Value Network | MLP (64 → 64), tanh activation |
| Learning Rate | 3e-4 |
| Discount Factor (γ) | 0.99 |
| GAE Lambda | 0.95 |
| Clip Range | 0.2 |
| Entropy Coefficient | 0.03 |
| Batch Size | 256 |
| N Steps | 2,048 |
| Epochs per Update | 10 |
| Training Timesteps | 1,500,000 |
| Training Time | ~68.7 minutes |
| Training Hardware | NVIDIA A100-SXM4-80GB |

## Training Progress

The agent's mean episode reward over the training run:

| Timestep | Mean Reward | Std Reward |
|----------|-------------|------------|
| 25,000 | -83.35 | 34.11 |
| 50,000 | -9.90 | 13.85 |
| 75,000 | +16.58 | 12.83 |
| 100,000 | +3.41 | 10.60 |
| 125,000 | +5.24 | 8.69 |
| 150,000 | +2.17 | 3.55 |
| 200,000 | +1.37 | 1.73 |
| 225,000+ | ~0.0 | ~0.0 |

The agent progresses from large losses at initialization, through positive reward territory by step 75k, and ultimately converges to a near-zero reward policy — consistent with a market making agent that learns to minimize inventory risk by reducing trade frequency rather than aggressively quoting.

## Expected Performance Benchmarks

Performance benchmarks on convergent policy (live simulation):

| Metric | Expected Range |
|--------|---------------|
| Sharpe Ratio | 0.5 – 2.0 |
| Fill Rate | 40% – 80% |
| Spread Capture | 30% – 60% |
| Max Drawdown | < 5% of initial capital |
| Mean Spread | ~1.23 bps |

## Market Microstructure Features

**Order Book State:**
- Bid-ask spread
- Depth across top 5 price levels per side
- Order imbalance ratio
- Order flow toxicity (VPIN approximation)

**Agent State:**
- Normalized inventory position
- Cash ratio (current / initial)
- Mark-to-market PnL ratio
- Fraction of episode remaining

## Technical Stack

- **Deep RL:** PyTorch, Stable-Baselines3 (PPO)
- **Environment:** Gymnasium (custom market making environment)
- **Market Data:** Polygon.io API, Alpha Vantage API, Synthetic GBM fallback
- **UI Framework:** Gradio
- **Visualization:** Plotly (interactive order book heatmap, PnL curves, performance dashboard)
- **Data Processing:** pandas, numpy, scipy
- **Development:** Google Colab Pro with A100 GPU

## Setup and Usage

### Running in Google Colab

1. Clone this repository or download the notebook file
2. Upload to Google Colab
3. Select Runtime > Change runtime type > A100 GPU (or T4 GPU for free tier)
4. Run all cells sequentially

The notebook will automatically:
- Install required dependencies
- Initialize the market making environment
- Train the PPO agent
- Evaluate performance across multiple episodes
- Launch a Gradio interface with a shareable link

### Running Locally

```bash
# Clone the repository
git clone https://github.com/SpencerCPurdy/Deep_RL_Market_Maker.git
cd Deep_RL_Market_Maker

# Install dependencies
pip install torch torchvision torchaudio stable-baselines3[extra] gymnasium pandas numpy requests plotly gradio scipy scikit-learn seaborn matplotlib

# Run the application
python app.py
```

**Note:** The pre-trained model weights (`ppo_market_maker.zip`) must be present in the repository root. Training from scratch takes approximately 60–70 minutes on an A100 GPU.

## Project Structure

```
├── app.py
├── ppo_market_maker.zip
├── training_metrics.json
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

The application contains the following components:

1. **Configuration & Setup**: `MarketMakerConfig` dataclass with all hyperparameters documented
2. **Market Data Fetcher**: Polygon.io, Alpha Vantage, and GBM synthetic data integrations
3. **Limit Order Book Simulator**: Multi-level LOB with Poisson order flow and book uncrossing logic
4. **Market Making Environment**: Custom Gymnasium environment with Avellaneda-Stoikov reward
5. **PPO Agent Loader**: Loads pre-trained weights at startup for CPU inference
6. **Evaluation Engine**: Multi-episode performance computation with full metrics suite
7. **Gradio Interface**: Five-tab interactive application (Live Trading, Training History, Order Book, Performance Analytics, Documentation)

## Key Implementation Details

- **Reproducibility:** All random seeds fixed to 42 across Python, NumPy, and PyTorch
- **Book Uncrossing:** Logic prevents bid-ask inversions during order placement
- **Adverse Selection:** Directional order flow component penalizes quoting into informed traders
- **Inventory Penalty:** Quadratic penalty coefficient discourages large unhedged positions
- **Terminal Penalty:** Additional inventory liquidation penalty applied at episode end
- **API Fallback:** If no API keys are configured, synthetic GBM price process is used automatically

## Limitations and Known Issues

### Model Limitations
- The trained agent converges toward low-activity policies to minimize inventory risk, which reduces PnL potential in low-volatility regimes
- No partial fills modeled — orders execute atomically
- Queue priority and time priority not modeled in order execution
- Single-asset scope; cross-asset correlations not considered

### Market Model Limitations
- Square-root market impact is a simplification; full LOB impact is not modeled
- Synthetic background order flow is not calibrated to live microstructure data
- No news or jump risk; information-driven price moves outside training distribution
- Flash crash events and circuit breaker scenarios not modeled

### Operational Limitations
- Inference runs on CPU only; real-time data requires valid Polygon.io or Alpha Vantage API keys
- API rate limits may affect data feed continuity during live simulation
- Maximum inventory hard limit of 1,000 shares enforced; trades breaching this limit are rejected

### Known Failure Modes
- Adverse selection during strong directional trending regimes
- Inventory accumulation under persistent one-sided order flow
- Spread widening during volatility spikes reduces fill rate
- Reduced profitability in very low-volatility, mean-reverting regimes

## API Integration

| Provider | Data Type | Requirement |
|----------|-----------|-------------|
| Polygon.io | Real-time quotes, Level-1 data, historical minute bars | `POLYGON_API_KEY` env variable |
| Alpha Vantage | Intraday OHLCV (1-min, 5-min) | `ALPHA_VANTAGE_API_KEY` env variable |
| Synthetic GBM | Geometric Brownian Motion simulation | No key required (automatic fallback) |

## Cost and Risk Model Parameters

| Parameter | Value |
|-----------|-------|
| Transaction Fee | 1.0 bps |
| Slippage | 0.5 bps |
| Permanent Impact Coefficient | 0.1 |
| Temporary Impact Coefficient | 0.5 |
| Adverse Selection Factor | 0.3 |
| Base Spread | 5.0 bps |
| Annualized Volatility | 20% |
| Order Arrival Rate (λ) | 10 orders/second |

## References

1. Avellaneda & Stoikov (2008) — *High-frequency trading in a limit order book*
2. Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*
3. Easley et al. (2012) — *Flow toxicity and liquidity in a high-frequency world*
4. Cartea et al. (2015) — *Algorithmic and High-Frequency Trading*
5. van Hasselt et al. (2016) — *Deep Reinforcement Learning with Double Q-learning*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Spencer Purdy**  
GitHub: [@SpencerCPurdy](https://github.com/SpencerCPurdy)

---

*This is a portfolio project developed to demonstrate deep reinforcement learning, market microstructure modeling, and production ML engineering. The system is designed for educational and demonstrational purposes only. Real trading involves significant financial risk and regulatory requirements not covered in this demonstration. Always verify with licensed financial professionals before any real-world application.*
