# Similarity-Based Candle Pattern Forecasting Trading Bot

Crypto futures trading bot using similarity-based candle pattern recognition with RAG-powered forecasting.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  React Frontend │────▶│  FastAPI Backend │────▶│  Binance API    │
│  Tailwind+Lucide│     │  (REST + WebSocket)│    │  (via CCXT)     │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ Chroma   │ │ FAISS +  │ │ vectorbt │
              │ RAG +    │ │ fastdtw  │ │ Backtest │
              │ LLM API  │ │ Patterns │ │          │
              └──────────┘ └──────────┘ └──────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Data | CCXT (Binance Futures) |
| Indicators | pandas-ta (main), TA-Lib (optional speed) |
| Backtesting | vectorbt (main), Backtesting.py (prototype) |
| Pattern Search | FAISS + fastdtw (DTW) |
| RAG/LLM | Chroma + Ollama (LLM_MODEL) |
| Optimization | Optuna |
| API | FastAPI |

## Setup

**Requires Python 3.10–3.12** (numba/vectorbt do not support 3.14 yet).

### Backend

```bash
cd backend
python3.12 -m venv venv   # or python3.11
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Create `.env`:

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
CHROMA_PERSIST_DIR=./chroma_db
LLM_API_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b-coder
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Usage

1. **Backend**: `cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python run.py`
2. **Frontend**: `cd frontend && npm install && npm run dev`
3. Open http://localhost:5173 — API is proxied to :8000

### Flow

1. **Data Collection**: Fetch OHLCV via CCXT
2. **Pattern Indexing**: Build Patterns → FAISS + Chroma
3. **Similarity Search**: DTW finds similar historical patterns
4. **RAG Forecasting**: Chroma retrieves context → LLM (Ollama) generates forecast
5. **Backtest**: vectorbt validates strategy
6. **Optimize**: Optuna tunes parameters

### PatternSimilaritySearch

Find historical moments most similar to a query using a user-chosen subset of features (RSI, long_short_ratio, taker_ratio, etc.). Supports cosine similarity and Euclidean distance.

**Example 1: Basic query with selected features**

```python
from app.patterns import PatternSimilaritySearch

searcher = PatternSimilaritySearch()
searcher.fit("data.json")  # or pass pd.DataFrame
result = searcher.query(
    current_values_dict={"rsi": 51.57, "normalize_long_short_ratio": 0.68, "taker_ratio": 1.38},
    selected_features_list=["rsi", "normalize_long_short_ratio", "taker_ratio"],
    metric="cosine",
    k=8
)
# result: DataFrame with timestamp, similarity_score, matched_row_data
```

**Example 2: Different feature set with Euclidean distance**

```python
result = searcher.query(
    current_values_dict={
        "normalize_ema_9": 0.42,
        "normalize_macd_hist": -0.1,
        "supertrend_dir": 1,
        "funding_rate": 0.0001,
    },
    selected_features_list=["normalize_ema_9", "normalize_macd_hist", "supertrend_dir", "funding_rate"],
    metric="euclidean",
    k=10
)
```

**Example 3: Similar short sequences (last 8 candles)**

```python
# current_window_df = last 8 candles from your DataFrame
result = searcher.query_window(
    current_window_df=df.tail(8),
    selected_features_list=["rsi", "macd_hist", "close"],
    metric="cosine",
    k=5,
    aggregate="mean"  # or "flatten" for full sequence
)
```

## License

MIT
