"""AI Chat Service for strategy code generation with context management."""
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime, timezone
import httpx
from app.config import get_settings
from pymongo import MongoClient
from app.services.llm1_service import MONGODB_URI, DB_NAME
from app.api.routes import STRATEGIES_COLLECTION


class AIChatService:
    """Service for AI-powered strategy code generation with conversation context."""
    
    def __init__(self):
        self.settings = get_settings()
        self.conversations: Dict[str, List[Dict[str, str]]] = {}  # session_id -> messages
        self.strategies_base_path = Path(__file__).parent.parent / "strategies"
        
    def _get_llm_client(self):
        """Get HTTP client for LLM API."""
        headers = {
            "Content-Type": "application/json",
        }
        # Only add Authorization header if API key is provided
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        
        return httpx.AsyncClient(
            base_url=self.settings.llm_api_url,
            timeout=300.0,
            headers=headers
        )
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for strategy code generation."""
        return """You are a Python code generator for trading strategies. Your ONLY job is to write working Python code based on user requirements.

MODE: CODE GENERATOR, NOT CHAT ASSISTANT
- Do NOT engage in conversation, explanations, or discussions
- Do NOT ask clarifying questions - implement what the user requests
- Do NOT provide status updates or "thinking" messages
- Generate code immediately based on the requirements
- Output ONLY executable Python code

CRITICAL REQUIREMENT: ALL VALUES MUST BE DYNAMIC PARAMETERS!
- Even if user mentions specific values (e.g., "SMA20", "RSI 30", "period 14"), make them configurable parameters
- NEVER hardcode indicator periods, thresholds, or any numeric values
- Every strategy should have parameters that can be adjusted from the UI
- Default values should match what the user requested, but must be changeable

CRITICAL: CODE MUST WORK CORRECTLY!
- The code you generate MUST be executable and work without errors
- Always check if indicators exist before using them
- If an indicator doesn't exist, calculate it dynamically using pandas
- Test your logic: ensure signals are numeric (1, -1, 0), not strings
- Handle edge cases: check data length, handle NaN values, validate inputs
- Use proper pandas operations: avoid string replacements on numeric Series
- Return Series with correct dtype (numeric, not object)

CRITICAL: SERIES-TO-SCALAR CONVERSIONS!
- When extracting values from Series/DataFrames for calculations, ALWAYS convert to scalar floats
- Use: `float(series.iloc[-1])` or `float(series.iloc[0])` to get scalar values
- Check for NaN: `pd.notna(series.iloc[-1])` before converting
- NEVER multiply Series by scalars directly - extract scalar first: `val = float(series.iloc[-1]); result = factor * val`
- Example CORRECT: `atr_val = float(df['atr'].iloc[-1]) if pd.notna(df['atr'].iloc[-1]) else 0.0`
- Example WRONG: `buffer = factor * df['atr'].iloc[-1]` (this can fail if Series)

CRITICAL: INDICATOR CALCULATIONS!
- ATR calculation REQUIRES high, low, and close - NEVER calculate ATR from close alone
- ATR formula: TR = max(high-low, abs(high-close_prev), abs(low-close_prev)); ATR = EMA(TR, period=14)
- RSI calculation: delta = close.diff(); gain = delta.where(delta>0,0); loss = (-delta).where(delta<0,0); avg_gain = gain.ewm(alpha=1/period).mean(); avg_loss = loss.ewm(alpha=1/period).mean(); RSI = 100 - (100/(1+avg_gain/avg_loss))
- Volume SMA: df['volume'].rolling(window=period).mean()
- Always validate calculated indicators have valid values before using them

CRITICAL: NaN HANDLING IN ROLLING WINDOWS!
- Rolling windows (rolling(), ewm()) produce NaN for the first N bars where N = window size
- ALWAYS fill NaN values after rolling calculations: series.bfill().ffill() or series.fillna(method='bfill').fillna(method='ffill')
- Example CORRECT:
  ```python
  volume_sma = df['volume'].rolling(window=20).mean()
  volume_sma = volume_sma.bfill().ffill()  # Fill NaN from rolling window
  volume_sma = volume_sma.fillna(0)  # Replace any remaining NaN with 0
  ```
- For comparisons, use Series operations: `df['volume'] >= (multiplier * volume_sma)` (both are Series)
- Only extract scalars when you need a single value for a calculation: `val = float(series.iloc[-1])`
- NEVER compare Series with NaN values directly - fill NaN first!

CRITICAL: BOOLEAN OPERATOR PRECEDENCE!
- ALWAYS use parentheses around boolean conditions when combining with & or |
- Example CORRECT: `condition = (df['close'] > level) & (df['volume'] > threshold) & (rsi > 50)`
- Example WRONG: `condition = df['close'] > level & df['volume'] > threshold & rsi > 50` (will fail!)
- Python requires parentheses: `(condition1) & (condition2) & (condition3)`

CRITICAL: HANDLING LARGE DATASETS (500+ candles)!
- Strategies MUST work correctly with ANY dataset size (100, 500, 2000+ candles)
- Rolling windows produce NaN for first N bars - ALWAYS fill these NaN values
- Use forward/backward fill: `series.bfill().ffill()` or `series.fillna(method='bfill').fillna(method='ffill')`
- After filling, replace remaining NaN with 0: `series.fillna(0)`
- Check data length BEFORE calculations: `if len(df) < min_period: return signals`
- Ensure signals are generated for ALL valid rows, not just first few hundred
- Example CORRECT pattern:
  ```python
  # Calculate rolling indicator
  rolling_high = df['high'].rolling(window=20).max()
  # Fill NaN from rolling window
  rolling_high = rolling_high.bfill().ffill().fillna(0)
  # Use in condition (works for all rows)
  condition = (df['close'] > rolling_high) & (rolling_high > 0)
  ```

OUTPUT FORMAT: Respond with ONLY Python code wrapped in ```python code blocks. 
- NO explanations before or after the code
- NO comments outside the code block
- NO JSON responses
- NO conversational text
- Just the code block with working Python code

Strategy Code Requirements:
- Must inherit from BaseStrategy (from app.strategies.base import BaseStrategy)
- Must implement generate_signals(self, df: pd.DataFrame) -> pd.Series
- Use pandas DataFrame with OHLCV columns: 'open', 'high', 'low', 'close', 'volume'
- Return Series with values: 1 (long), -1 (short/close), 0 (flat) - MUST be numeric!
- ALL parameters must be configurable via __init__ kwargs with sensible defaults
- Use kwargs.get() pattern for all parameters with defaults

Available Pre-calculated Indicators (may exist in DataFrame):
- RSI: df['rsi'] (period 9 or 14, depending on timeframe)
- MACD: df['macd'], df['macd_signal'], df['macd_hist']
- EMAs: df['ema_9'], df['ema_12'], df['ema_21'], df['ema_26'] (NOT ema_5, ema_35, etc.)
- SMAs: df['sma_20'], df['sma_50'], df['sma_200']
- Bollinger Bands: df['bb_upper'], df['bb_mid'], df['bb_lower']
- Stochastic: df['stoch_k'], df['stoch_d']
- CCI, ATR, PSAR, SuperTrend, VWAP, etc.

IMPORTANT: Indicator Availability Rules:
- Only ema_9, ema_12, ema_21, ema_26 are pre-calculated
- Only sma_20, sma_50, sma_200 are pre-calculated
- If you need other periods (e.g., ema_5, ema_35, sma_100), YOU MUST CALCULATE THEM:
  - EMA: df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
  - SMA: df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
- Always check if indicator exists: if f"ema_{period}" not in df.columns: calculate it
- Check data length: if len(df) < period: return empty signals

Example: If user says "SMA20 cross SMA100", create parameters:
- fast_sma_period (default: 20)
- slow_sma_period (default: 100)
- Calculate sma_100 if it doesn't exist (sma_20 might exist, sma_100 won't)

Example structure (CORRECT - shows all critical patterns):
```python
from app.strategies.base import BaseStrategy
import pandas as pd
import numpy as np

class MyStrategy(BaseStrategy):
    def __init__(self, fast_period=20, slow_period=100, rsi_threshold=30, atr_period=14, **kwargs):
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_threshold = rsi_threshold
        self.atr_period = atr_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # Check data length
        if len(df) < max(self.slow_period, self.atr_period) + 10:
            return signals
        
        # Calculate indicators if they don't exist
        fast_col = f"sma_{self.fast_period}"
        slow_col = f"sma_{self.slow_period}"
        
        if fast_col not in df.columns:
            df[fast_col] = df["close"].rolling(window=self.fast_period).mean()
        if slow_col not in df.columns:
            df[slow_col] = df["close"].rolling(window=self.slow_period).mean()
        
        # Calculate ATR correctly (requires high, low, close)
        if "atr" not in df.columns:
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr"] = tr.ewm(span=self.atr_period, adjust=False).mean()
        
        # Extract scalar values for calculations (CRITICAL!)
        atr_val = float(df["atr"].iloc[-1]) if pd.notna(df["atr"].iloc[-1]) else 0.0
        if atr_val == 0:
            return signals  # Can't proceed without valid ATR
        
        fast_sma = df[fast_col]
        slow_sma = df[slow_col]
        
        # Fill NaN values from rolling calculations (CRITICAL for large datasets!)
        fast_sma = fast_sma.bfill().ffill().fillna(0)
        slow_sma = slow_sma.bfill().ffill().fillna(0)
        
        # Generate signals (numeric: 1, -1, 0)
        # ALWAYS use parentheses around boolean conditions!
        # Crossover detection: current condition true AND previous condition false
        long_condition = (
            (fast_sma > slow_sma) & 
            (fast_sma.shift(1) <= slow_sma.shift(1)) &
            (fast_sma > 0) &  # Ensure valid values
            (slow_sma > 0)
        )
        signals[long_condition] = 1
        
        short_condition = (
            (fast_sma < slow_sma) & 
            (fast_sma.shift(1) >= slow_sma.shift(1)) &
            (fast_sma > 0) &  # Ensure valid values
            (slow_sma > 0)
        )
        signals[short_condition] = -1
        
        # Handle NaN values
        signals = signals.fillna(0)
        
        return signals
```

CRITICAL REMINDERS:
1. Make EVERY value a parameter, even if user specifies exact numbers
2. ALWAYS check if indicators exist before using them - calculate if missing
3. Return numeric signals (1, -1, 0) - NEVER strings like 'buy' or 'sell'
4. Handle edge cases: check data length, handle NaN values
5. Test your logic: ensure the code will execute without errors
6. Generate complete, working code that can be executed directly
7. ALWAYS convert Series to scalar floats before calculations: `val = float(series.iloc[-1])`
8. ALWAYS use parentheses around boolean conditions: `(cond1) & (cond2) & (cond3)`
9. ATR MUST use high, low, close - never calculate from close alone
10. Validate scalar values before using: check for NaN, check for zero if needed
11. ALWAYS fill NaN values from rolling windows: `series.bfill().ffill().fillna(0)`
12. Strategies MUST work with ANY dataset size (100, 500, 2000+ candles)
13. Use Series comparisons for volume thresholds: `df['volume'] >= (multiplier * volume_sma)` (both Series)
14. Only extract scalars when you need a single value, otherwise use Series operations

COMMON MISTAKES TO AVOID:
- DON'T: `buffer = factor * df['atr'].iloc[-1]` (Series multiplication issue)
- DO: `atr_val = float(df['atr'].iloc[-1]); buffer = factor * atr_val`
- DON'T: `condition = df['close'] > level & df['volume'] > threshold` (operator precedence)
- DO: `condition = (df['close'] > level) & (df['volume'] > threshold)`
- DON'T: `atr = df['close'].ewm(span=14).mean()` (ATR needs high, low, close)
- DO: Calculate True Range first, then EMA of TR
- DON'T: `volume_sma = df['volume'].rolling(20).mean(); condition = df['volume'] > threshold` (threshold is scalar)
- DO: `volume_sma = df['volume'].rolling(20).mean().bfill().ffill().fillna(0); condition = df['volume'] >= (multiplier * volume_sma)` (both Series)
- DON'T: `rolling_high = df['high'].rolling(20).max(); condition = df['close'] > rolling_high` (NaN values cause issues)
- DO: `rolling_high = df['high'].rolling(20).max().bfill().ffill().fillna(0); condition = (df['close'] > rolling_high) & (rolling_high > 0)`
- DON'T: Ignore NaN values from rolling windows - they will cause 0 signals with large datasets
- DO: Always fill NaN: `series.bfill().ffill().fillna(0)` after any rolling calculation

REMEMBER: You are a CODE GENERATOR. Generate code immediately. Do NOT chat, explain, or ask questions.
Just output the Python code block when you receive requirements.
"""
    
    def _build_code_review_prompt(self, code_content: str) -> str:
        """Build system prompt for code review and refactoring."""
        return f"""You are an expert Python code reviewer specializing in trading strategy backtesting code. Your task is to review and refactor the provided Python code.

CRITICAL TASKS:
1. VALIDATE: Check if the code is valid Python code for backtesting
   - Must inherit from BaseStrategy (from app.strategies.base import BaseStrategy)
   - Must implement generate_signals(self, df: pd.DataFrame) -> pd.Series
   - Must return Series with values: 1 (long), -1 (short/close), 0 (flat) - MUST be numeric!
   - Must use pandas DataFrame with OHLCV columns
   - Code MUST work correctly - check for errors, missing indicators, incorrect logic

2. IDENTIFY STATIC VALUES: Find ALL hardcoded/static values in the code:
   - Numeric literals (e.g., 20, 100, 0.75, 2.5, 30, 70)
   - String literals used as thresholds or parameters
   - Boolean values that should be configurable
   - List/tuple literals with fixed values
   - Magic numbers in calculations

3. REFACTOR TO DYNAMIC PARAMETERS:
   - Move ALL static values to __init__ method parameters
   - Use kwargs.get() pattern for parameters with defaults
   - Replace hardcoded values with self.parameter_name throughout the code
   - Keep the original values as default parameter values
   - Ensure all parameters have sensible type hints

4. CODE STRUCTURE REQUIREMENTS:
   - All parameters must be in __init__ method signature
   - Parameters must be stored as instance variables (self.param_name)
   - generate_signals method must use self.param_name, never hardcoded values
   - Preserve all original logic and functionality
   - Maintain code comments and docstrings

5. ENSURE CODE WORKS CORRECTLY:
   - Check if indicators exist before using them - calculate if missing
   - Only pre-calculated indicators: ema_9, ema_12, ema_21, ema_26, sma_20, sma_50, sma_200
   - For other periods (e.g., ema_5, ema_35, sma_100), MUST calculate them:
     * EMA: df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
     * SMA: df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
   - Always check: if f"ema_{period}" not in df.columns: calculate it
   - Check data length: if len(df) < period: return empty signals
   - Return numeric signals (1, -1, 0) - NEVER strings like 'buy' or 'sell'
   - Handle NaN values: signals = signals.fillna(0)
   - Validate logic: ensure signals are numeric Series, not object dtype
   
6. CRITICAL: NaN HANDLING IN ROLLING WINDOWS:
   - Rolling windows (rolling(), ewm()) produce NaN for the first N bars where N = window size
   - ALWAYS fill NaN values after rolling calculations: series.bfill().ffill().fillna(0)
   - Example: volume_sma = df['volume'].rolling(20).mean().bfill().ffill().fillna(0)
   - This is CRITICAL for large datasets (2000+ candles) - without it, strategies return 0 signals
   - Use Series comparisons for volume thresholds: df['volume'] >= (multiplier * volume_sma) (both Series)
   - Only extract scalars when you need a single value: val = float(series.iloc[-1])
   
7. CRITICAL: HANDLING LARGE DATASETS:
   - Strategies MUST work correctly with ANY dataset size (100, 500, 2000+ candles)
   - After ANY rolling calculation, fill NaN: series.bfill().ffill().fillna(0)
   - Ensure signals are generated for ALL valid rows, not just first few hundred
   - Check data length BEFORE calculations: if len(df) < min_period: return signals
   
8. CRITICAL: SERIES-TO-SCALAR CONVERSIONS:
   - When extracting values from Series for calculations, ALWAYS convert to scalar floats
   - Use: float(series.iloc[-1]) with pd.notna() check before converting
   - NEVER multiply Series by scalars directly - extract scalar first
   - Example CORRECT: atr_val = float(df['atr'].iloc[-1]); buffer = factor * atr_val
   - Example WRONG: buffer = factor * df['atr'].iloc[-1] (can fail if Series)
   
9. CRITICAL: BOOLEAN OPERATOR PRECEDENCE:
   - ALWAYS use parentheses around boolean conditions: (cond1) & (cond2) & (cond3)
   - Example CORRECT: condition = (df['close'] > level) & (df['volume'] > threshold)
   - Example WRONG: condition = df['close'] > level & df['volume'] > threshold (will fail!)
   
10. CRITICAL: INDICATOR CALCULATIONS:
    - ATR MUST use high, low, close - NEVER calculate from close alone
    - ATR formula: TR = max(high-low, abs(high-close_prev), abs(low-close_prev)); ATR = EMA(TR, period=14)
    - RSI calculation: delta = close.diff(); gain = delta.where(delta>0,0); loss = (-delta).where(delta<0,0); avg_gain = gain.ewm(alpha=1/period).mean(); avg_loss = loss.ewm(alpha=1/period).mean(); RSI = 100 - (100/(1+avg_gain/avg_loss))

AVAILABLE INDICATORS (already in DataFrame):
- RSI: df['rsi']
- MACD: df['macd'], df['macd_signal'], df['macd_hist']
- EMAs: df['ema_9'], df['ema_12'], df['ema_21'], df['ema_26']
- SMAs: df['sma_20'], df['sma_50'], df['sma_200']
- Bollinger Bands: df['bb_upper'], df['bb_mid'], df['bb_lower']
- Stochastic: df['stoch_k'], df['stoch_d']
- CCI, ATR, PSAR, SuperTrend, VWAP, etc.

CODE TO REVIEW:
```python
{code_content}
```

RESPONSE FORMAT:
Respond with ONLY the refactored Python code wrapped in ```python code blocks. Do not include explanations, comments outside code blocks, or JSON.

The refactored code must:
- Be valid, executable Python code
- Have ALL static values converted to __init__ parameters
- Use self.parameter_name instead of hardcoded values
- Preserve all original functionality
- Be ready for backtesting use
- Handle NaN values properly: fill NaN after rolling calculations (series.bfill().ffill().fillna(0))
- Work correctly with ANY dataset size (100, 500, 2000+ candles)
- Use Series comparisons for volume thresholds, not scalar comparisons
- Always use parentheses around boolean conditions: (cond1) & (cond2)
- Convert Series to scalars only when needed: float(series.iloc[-1])

COMMON ISSUES TO FIX:
- Missing NaN handling after rolling calculations (causes 0 signals with large datasets)
- Using scalar thresholds instead of Series comparisons for volume
- Missing parentheses in boolean conditions
- Not converting Series to scalars before calculations
- Calculating ATR from close alone instead of high, low, close

EXAMPLE TRANSFORMATION:
Before:
```python
def generate_signals(self, df: pd.DataFrame) -> pd.Series:
    signals = pd.Series(0, index=df.index)
    if df['rsi'].iloc[-1] < 30:  # Hardcoded 30
        signals.iloc[-1] = 1
    return signals
```

After:
```python
def __init__(self, rsi_oversold=30, **kwargs):
    super().__init__(**kwargs)
    self.rsi_oversold = rsi_oversold

def generate_signals(self, df: pd.DataFrame) -> pd.Series:
    signals = pd.Series(0, index=df.index)
    if df['rsi'].iloc[-1] < self.rsi_oversold:  # Dynamic parameter
        signals.iloc[-1] = 1
    return signals
```

REMEMBER: Every numeric literal, string threshold, boolean flag, and magic number must become a parameter!
"""
    
    def _extract_code_and_params(self, text: str) -> Dict[str, Any]:
        """Extract code and parameters from LLM response. Returns ONLY Python code, no explanations."""
        # Try to find JSON in the response (if LLM returns JSON)
        json_match = re.search(r'\{[^{}]*"code"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                code = data.get("code", "").strip()
                if code:
                    # Clean code - remove any markdown if present
                    code = self._clean_python_code(code)
                    params = self._extract_params_from_code(code)
                    return {
                        "code": code,
                        "params": params,
                        "message": "Strategy code extracted from JSON"
                    }
            except json.JSONDecodeError:
                pass
        
        # Try to find code blocks - multiple patterns (prioritize python blocks)
        code_patterns = [
            r'```python\s*\n(.*?)\n```',  # Standard markdown with python
            r'```python(.*?)```',  # No newline after python
            r'```\s*\n(.*?)\n```',  # Code block without language
            r'```(.*?)```',  # Any code block
        ]
        
        for pattern in code_patterns:
            code_block_match = re.search(pattern, text, re.DOTALL)
            if code_block_match:
                code = code_block_match.group(1).strip()
                # Clean the code
                code = self._clean_python_code(code)
                # Validate it looks like Python code
                if 'class' in code and 'BaseStrategy' in code and 'generate_signals' in code:
                    # Try to extract parameters from code
                    params = self._extract_params_from_code(code)
                    return {
                        "code": code,
                        "params": params,
                        "message": "Strategy code extracted from code block"
                    }
        
        # Try to find code without markdown blocks (if LLM returns raw code)
        # Look for class definition followed by methods - more comprehensive pattern
        raw_code_match = re.search(
            r'(from\s+app\.strategies\.base\s+import\s+BaseStrategy.*?class\s+\w+\s*\(BaseStrategy\).*?def\s+generate_signals.*?return\s+signals)',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if raw_code_match:
            code = raw_code_match.group(1).strip()
            code = self._clean_python_code(code)
            params = self._extract_params_from_code(code)
            return {
                "code": code,
                "params": params,
                "message": "Strategy code extracted from raw text"
            }
        
        # Fallback: return text as message (no code found)
        return {
            "code": None,
            "params": [],
            "message": f"Could not extract code from response. Response: {text[:500]}..."
        }
    
    def _clean_python_code(self, code: str) -> str:
        """Clean Python code: remove markdown, explanations, and ensure it's pure Python."""
        # Remove any markdown code block markers if still present
        code = re.sub(r'^```python\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
        
        # Remove common explanation patterns that might be mixed in
        # Remove lines that look like explanations (not Python code)
        lines = code.split('\n')
        cleaned_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            # Skip empty lines at the start
            if not in_code and not stripped:
                continue
            
            # Start collecting when we see imports or class definition
            if 'import' in stripped.lower() or stripped.startswith('class ') or stripped.startswith('from '):
                in_code = True
            
            # Stop if we see markdown or explanation markers
            if stripped.startswith('#') and ('explanation' in stripped.lower() or 'note:' in stripped.lower()):
                continue
            
            # Skip markdown-style headers
            if stripped.startswith('#') and len(stripped) > 0 and stripped[1:].strip().isupper():
                continue
            
            if in_code:
                cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines).strip()
        
        # Ensure code starts with import or class
        if not code.startswith(('from ', 'import ', 'class ')):
            # Try to find the actual start
            import_match = re.search(r'(from\s+.*?import.*?\n.*?class)', code, re.DOTALL)
            if import_match:
                code = code[import_match.start():]
        
        return code
    
    def _generate_strategy_name(self, description: str) -> str:
        """Generate a unique, understandable strategy name from description."""
        # Clean description: lowercase, remove special chars, keep alphanumeric and spaces
        clean = re.sub(r'[^a-z0-9\s]', '', description.lower())
        # Take first few words
        words = clean.split()[:4]
        if not words:
            words = ["strategy"]
        
        # Create base name
        base_name = "_".join(words)
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{base_name}_{timestamp}"
    
    def _save_strategy_code(self, code: str, strategy_name: str) -> str:
        """Save strategy code to file and return the file path. Ensures ONLY Python code is saved."""
        # Clean the code before saving - remove any explanations or markdown
        clean_code = self._clean_python_code(code)
        
        # Validate we have actual Python code
        if not clean_code or len(clean_code.strip()) < 50:
            raise ValueError("Invalid or empty code after cleaning")
        
        if 'class' not in clean_code or 'BaseStrategy' not in clean_code:
            raise ValueError("Code does not contain a valid BaseStrategy class")
        
        # Ensure strategies directory exists
        self.strategies_base_path.mkdir(parents=True, exist_ok=True)
        
        # Create strategy directory
        strategy_dir = self.strategies_base_path / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ONLY clean Python code to file
        strategy_file = strategy_dir / "strategy.py"
        strategy_file.write_text(clean_code, encoding="utf-8")
        
        # Return relative path from app directory (used by loader)
        relative_path = f"app/strategies/{strategy_name}/strategy.py"
        return relative_path
    
    def _extract_params_from_code(self, code: str) -> List[Dict[str, Any]]:
        """Extract parameter definitions from strategy code."""
        params = []
        
        # Find __init__ method signature and body
        init_match = re.search(r'def __init__\(self[^)]*\):\s*(.*?)(?=\n    def|\nclass|\Z)', code, re.DOTALL)
        if not init_match:
            return params
        
        init_body = init_match.group(1)
        
        # Extract function signature parameters first
        init_sig_match = re.search(r'def __init__\(self[^)]*\)', code)
        if init_sig_match:
            sig = init_sig_match.group(0)
            # Extract parameters from signature: param1=value1, param2=value2
            sig_params = re.findall(r'(\w+)\s*=\s*([0-9.]+|True|False|[\'"][^\'"]*[\'"])', sig)
            for param_name, default_str in sig_params:
                # Skip kwargs
                if param_name == "kwargs":
                    continue
                
                # Parse default value
                default_value = default_str.strip()
                param_type = "float"
                
                if default_value.lower() in ("true", "false"):
                    param_type = "bool"
                    default_value = default_value.lower() == "true"
                elif default_value.startswith(('"', "'")):
                    param_type = "string"
                    default_value = default_value.strip('"\'')
                else:
                    try:
                        if '.' in default_value:
                            default_value = float(default_value)
                        else:
                            default_value = int(default_value)
                            param_type = "int"
                    except ValueError:
                        param_type = "string"
                
                # Check if already added
                if not any(p["name"] == param_name for p in params):
                    params.append({
                        "name": param_name,
                        "type": param_type,
                        "default": default_value,
                        "description": f"Parameter {param_name}"
                    })
        
        # Also find parameter assignments in body (for kwargs.get pattern)
        param_patterns = [
            r'self\.(\w+)\s*=\s*kwargs\.get\([\'"]([^\'"]+)[\'"],\s*([^\)]+)\)',
            r'self\.(\w+)\s*=\s*([0-9.]+|True|False|[\'"][^\'"]*[\'"])',
        ]
        
        for pattern in param_patterns:
            for match in re.finditer(pattern, init_body):
                param_name = match.group(1)
                
                # Skip if already added
                if any(p["name"] == param_name for p in params):
                    continue
                
                # Extract default value
                if len(match.groups()) >= 3:
                    # kwargs.get pattern
                    default_value = match.group(3).strip()
                else:
                    default_value = match.group(2).strip()
                
                # Determine type
                param_type = "float"
                if isinstance(default_value, str):
                    default_value = default_value.strip()
                    if default_value.lower() in ("true", "false"):
                        param_type = "bool"
                        default_value = default_value.lower() == "true"
                    elif default_value.startswith(('"', "'")):
                        param_type = "string"
                        default_value = default_value.strip('"\'')
                    else:
                        try:
                            if '.' in default_value:
                                default_value = float(default_value)
                            else:
                                default_value = int(default_value)
                                param_type = "int"
                        except ValueError:
                            param_type = "string"
                
                params.append({
                    "name": param_name,
                    "type": param_type,
                    "default": default_value,
                    "description": f"Parameter {param_name}"
                })
        
        return params
    
    async def stream_chat(
        self,
        message: str,
        session_id: str,
        strategy_context: Optional[str] = None,
        strategy_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream chat response with status updates."""
        # Get or create conversation history
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        # Add user message
        self.conversations[session_id].append({
            "role": "user",
            "content": message
        })
        
        # Build messages for LLM
        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        
        # Add strategy context if provided
        if strategy_context:
            messages.append({
                "role": "system",
                "content": f"Current strategy context: {strategy_context}"
            })
        
        # Add conversation history
        messages.extend(self.conversations[session_id][-10:])  # Last 10 messages
        
        # Stream response
        async with self._get_llm_client() as client:
            try:
                # Send status update
                yield {
                    "type": "status",
                    "status": "generating",
                    "message": "Generating strategy code..."
                }
                
                # Prepare request - Ollama uses /api/chat endpoint
                request_data = {
                    "model": self.settings.llm_model,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.2,  # Lower temperature for more deterministic code generation
                }
                
                # Determine endpoint based on API URL
                endpoint = "/api/chat" if "ollama" in self.settings.llm_api_url.lower() else "/chat/completions"
                
                # Stream response
                async with client.stream(
                    "POST",
                    endpoint,
                    json=request_data
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        error_msg = error_text.decode() if error_text else "Unknown error"
                        
                        # Try to parse error JSON for better error message
                        try:
                            error_json = json.loads(error_msg)
                            if "error" in error_json and "message" in error_json["error"]:
                                error_msg = error_json["error"]["message"]
                        except:
                            pass
                        
                        yield {
                            "type": "error",
                            "message": f"LLM API error ({response.status_code}): {error_msg}\n\nTip: Make sure the model '{self.settings.llm_model}' is available in Ollama. Run 'ollama pull {self.settings.llm_model}' to download it."
                        }
                        return
                    
                    buffer = ""
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        
                        # Handle both SSE format (data: {...}) and direct JSON
                        data_str = line
                        if line.startswith("data: "):
                            data_str = line[6:]
                        
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            # Ollama format: {"message": {"content": "..."}, "done": false}
                            # OpenAI format: {"choices": [{"delta": {"content": "..."}}]}
                            content = None
                            
                            if "message" in data and isinstance(data["message"], dict):
                                # Ollama format
                                content = data["message"].get("content", "")
                            elif "choices" in data:
                                # OpenAI format
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                            
                            if content:
                                buffer += content
                                yield {
                                    "type": "content",
                                    "content": content
                                }
                            
                            # Check if done (Ollama)
                            if data.get("done", False):
                                break
                                
                        except json.JSONDecodeError:
                            # If it's not JSON, might be raw text - add to buffer
                            if line and not line.startswith("data:"):
                                buffer += line + "\n"
                            continue
                    
                    # Debug: log buffer if empty or short
                    if not buffer or len(buffer.strip()) < 50:
                        import logging
                        logging.warning(f"Short or empty buffer received. Length: {len(buffer)}, Preview: {buffer[:200]}")
                    
                    # Process final response
                    yield {
                        "type": "status",
                        "status": "generating",
                        "message": "Extracting code and parameters..."
                    }
                    
                    # Extract code and params
                    result = self._extract_code_and_params(buffer)
                    
                    # If no code found, provide helpful message
                    if not result.get("code"):
                        yield {
                            "type": "error",
                            "message": f"Could not extract Python code from LLM response. Please ensure the model generates complete Python code.\n\nResponse preview: {buffer[:300]}..."
                        }
                        return
                    
                    # Save code to file and database if available
                    strategy_file_path = None
                    strategy_name = None
                    saved_strategy_id = strategy_id  # Use provided strategy_id or None
                    if result.get("code"):
                        try:
                            from bson import ObjectId
                            
                            # Save to database
                            client = MongoClient(MONGODB_URI)
                            db = client[DB_NAME]
                            
                            # Prepare params dict from extracted params
                            params_dict = {}
                            for param in result.get("params", []):
                                params_dict[param["name"]] = param.get("default", 0)
                            
                            if strategy_id:
                                # Update existing strategy
                                existing = db[STRATEGIES_COLLECTION].find_one({"_id": ObjectId(strategy_id)})
                                if existing:
                                    strategy_name = existing.get("name", self._generate_strategy_name(message))
                                    
                                    # Use existing file path or create new one
                                    existing_file = existing.get("strategy_file")
                                    if existing_file:
                                        # Update existing file
                                        from pathlib import Path
                                        backend_dir = Path(__file__).parent.parent.parent
                                        file_path = backend_dir / existing_file
                                        if file_path.exists():
                                            # Clean code before saving
                                            clean_code = self._clean_python_code(result["code"])
                                            file_path.write_text(clean_code, encoding="utf-8")
                                            strategy_file_path = existing_file
                                        else:
                                            # File doesn't exist, create new one
                                            strategy_file_path = self._save_strategy_code(result["code"], strategy_name)
                                    else:
                                        # No existing file, create new one
                                        strategy_file_path = self._save_strategy_code(result["code"], strategy_name)
                                    
                                    # Update strategy
                                    db[STRATEGIES_COLLECTION].update_one(
                                        {"_id": ObjectId(strategy_id)},
                                        {
                                            "$set": {
                                                "strategy_file": strategy_file_path,
                                                "params": params_dict,
                                                "updated_at": datetime.now(timezone.utc),
                                            }
                                        }
                                    )
                                    saved_strategy_id = strategy_id
                                else:
                                    # Strategy not found, create new one
                                    strategy_id = None
                            else:
                                # Create new strategy
                                strategy_name = self._generate_strategy_name(message)
                                strategy_file_path = self._save_strategy_code(result["code"], strategy_name)
                                
                                # Check if strategy name already exists, append number if needed
                                final_name = strategy_name
                                counter = 1
                                while db[STRATEGIES_COLLECTION].find_one({"name": final_name}):
                                    final_name = f"{strategy_name}_{counter}"
                                    counter += 1
                                
                                doc = {
                                    "name": final_name,
                                    "strategy_file": strategy_file_path,
                                    "params": params_dict,
                                    "created_at": datetime.now(timezone.utc),
                                    "updated_at": datetime.now(timezone.utc),
                                }
                                
                                r = db[STRATEGIES_COLLECTION].insert_one(doc)
                                saved_strategy_id = str(r.inserted_id)
                                strategy_name = final_name  # Use final name (with counter if needed)
                            
                        except Exception as e:
                            # Log error but don't fail the request
                            import traceback
                            print(f"Error saving strategy: {e}\n{traceback.format_exc()}")
                    
                    # Add assistant message to history
                    self.conversations[session_id].append({
                        "role": "assistant",
                        "content": buffer
                    })
                    
                    # Send final result
                    yield {
                        "type": "complete",
                        "message": result["message"],
                        "code": result.get("code"),
                        "params": result.get("params", []),
                        "strategy_file": strategy_file_path,
                        "strategy_name": strategy_name,
                        "strategy_id": saved_strategy_id
                    }
                    
            except Exception as e:
                import traceback
                yield {
                    "type": "error",
                    "message": f"Error: {str(e)}\n{traceback.format_exc()}"
                }
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversations:
            del self.conversations[session_id]
    
    def get_conversation(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        return self.conversations.get(session_id, [])
    
    async def review_and_refactor_code(
        self,
        code_content: str,
        strategy_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Review strategy code and refactor static values to dynamic parameters.
        
        Args:
            code_content: The Python code to review and refactor
            strategy_id: Optional strategy ID to update after refactoring
            
        Yields:
            Dict with type, status, content, code, params, etc.
        """
        try:
            # Build review prompt
            review_prompt = self._build_code_review_prompt(code_content)
            
            # Stream response from LLM
            async with self._get_llm_client() as client:
                yield {
                    "type": "status",
                    "status": "reviewing",
                    "message": "Reviewing code and identifying static values..."
                }
                
                # Prepare request
                request_data = {
                    "model": self.settings.llm_model,
                    "messages": [
                        {"role": "system", "content": review_prompt}
                    ],
                    "stream": True,
                    "temperature": 0.3,  # Lower temperature for code review
                }
                
                # Determine endpoint
                endpoint = "/api/chat" if "ollama" in self.settings.llm_api_url.lower() else "/chat/completions"
                
                # Stream response
                async with client.stream(
                    "POST",
                    endpoint,
                    json=request_data
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(f"LLM API error: {response.status_code} - {error_text.decode()}")
                    
                    buffer = ""
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if not data_str.strip() or data_str.strip() == "[DONE]":
                                continue
                            
                            try:
                                data = json.loads(data_str)
                                
                                # Handle Ollama format
                                if "ollama" in self.settings.llm_api_url.lower():
                                    if "message" in data and "content" in data["message"]:
                                        content = data["message"]["content"]
                                        buffer += content
                                        yield {
                                            "type": "content",
                                            "content": content
                                        }
                                    
                                    if data.get("done", False):
                                        break
                                else:
                                    # Handle OpenAI format
                                    choices = data.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            buffer += content
                                            yield {
                                                "type": "content",
                                                "content": content
                                            }
                                    
                                    if choices and choices[0].get("finish_reason"):
                                        break
                                        
                            except json.JSONDecodeError:
                                if line and not line.startswith("data:"):
                                    buffer += line + "\n"
                                continue
                
                # Extract refactored code
                yield {
                    "type": "status",
                    "status": "extracting",
                    "message": "Extracting refactored code and parameters..."
                }
                
                result = self._extract_code_and_params(buffer)
                
                if not result.get("code"):
                    yield {
                        "type": "error",
                        "message": f"Could not extract Python code from review response. Response preview: {buffer[:300]}..."
                    }
                    return
                
                # If strategy_id provided, update the strategy file
                strategy_file_path = None
                if strategy_id and result.get("code"):
                    try:
                        from bson import ObjectId
                        client = MongoClient(MONGODB_URI)
                        db = client[DB_NAME]
                        existing = db[STRATEGIES_COLLECTION].find_one({"_id": ObjectId(strategy_id)})
                        
                        if existing:
                            strategy_name = existing.get("name", f"strategy_{strategy_id}")
                            existing_file = existing.get("strategy_file")
                            
                            if existing_file:
                                # Update existing file
                                backend_dir = Path(__file__).parent.parent.parent
                                file_path = backend_dir / existing_file
                                if file_path.exists():
                                    # Clean code before saving - ensure ONLY Python code
                                    clean_code = self._clean_python_code(result["code"])
                                    file_path.write_text(clean_code, encoding="utf-8")
                                    strategy_file_path = existing_file
                                else:
                                    strategy_file_path = self._save_strategy_code(result["code"], strategy_name)
                            else:
                                strategy_file_path = self._save_strategy_code(result["code"], strategy_name)
                            
                            # Update params in database
                            params_dict = {}
                            for param in result.get("params", []):
                                params_dict[param["name"]] = param.get("default", 0)
                            
                            db[STRATEGIES_COLLECTION].update_one(
                                {"_id": ObjectId(strategy_id)},
                                {
                                    "$set": {
                                        "strategy_file": strategy_file_path,
                                        "params": params_dict,
                                        "updated_at": datetime.now(timezone.utc),
                                    }
                                }
                            )
                    except Exception as e:
                        import traceback
                        print(f"Error updating strategy after review: {e}\n{traceback.format_exc()}")
                
                # Send final result
                yield {
                    "type": "complete",
                    "message": "Code review completed. All static values have been converted to dynamic parameters.",
                    "code": result.get("code"),
                    "params": result.get("params", []),
                    "strategy_file": strategy_file_path,
                    "strategy_id": strategy_id
                }
                
        except Exception as e:
            import traceback
            yield {
                "type": "error",
                "message": f"Error during code review: {str(e)}\n{traceback.format_exc()}"
            }


# Singleton instance
_chat_service: Optional[AIChatService] = None

def get_chat_service() -> AIChatService:
    """Get singleton chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = AIChatService()
    return _chat_service
