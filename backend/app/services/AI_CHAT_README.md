# AI Chat Service for Strategy Code Generation

## Overview

The AI Chat Service provides a conversational interface for generating trading strategy code. It uses Server-Sent Events (SSE) for real-time streaming responses and maintains conversation context for better understanding.

## Features

1. **Conversation Context Management**: Remembers previous messages in a session for better understanding
2. **SSE Streaming**: Real-time status updates and content streaming
3. **Code Generation**: Generates Python strategy code that inherits from `BaseStrategy`
4. **Parameter Extraction**: Automatically extracts configurable parameters from generated code
5. **Status Updates**: Shows what the AI is working on (thinking, generating, etc.)

## API Endpoints

### POST `/api/ai/chat/stream`

Stream AI chat response with SSE.

**Request Body:**
```json
{
  "message": "Create a strategy that buys when RSI < 30",
  "session_id": "optional-session-id",
  "strategy_context": "Optional context about current strategy"
}
```

**Response:** Server-Sent Events stream with:
- `type: "status"` - Status updates (thinking, generating)
- `type: "content"` - Partial content chunks
- `type: "complete"` - Final response with code and params
- `type: "error"` - Error messages
- `type: "session"` - Session ID

### POST `/api/ai/chat/clear`

Clear conversation history for a session.

**Query Parameters:**
- `session_id`: Session ID to clear

### GET `/api/ai/chat/history/{session_id}`

Get conversation history for a session.

## Configuration

Set in `.env`:
```
LLM_API_URL=http://localhost:11434/v1  # Ollama endpoint
LLM_API_KEY=                            # Optional API key
LLM_MODEL=qwen2.5:7b-coder              # Model name
```

## How It Works

1. **User sends message** → Added to conversation history
2. **System prompt** → Instructs AI on code generation requirements
3. **LLM streams response** → Content chunks sent via SSE
4. **Code extraction** → Parses Python code from response
5. **Parameter extraction** → Finds `__init__` parameters and their types
6. **Final response** → Returns code and structured parameters

## Parameter Structure

Extracted parameters follow this format:
```json
{
  "name": "rsi_threshold",
  "type": "float",
  "default": 30.0,
  "min": 0,
  "max": 100,
  "step": 1.0,
  "description": "RSI threshold for buy signal"
}
```

Types supported: `int`, `float`, `bool`, `string`

## Frontend Integration

The frontend `ChatUI` component:
- Connects to `/api/ai/chat/stream` endpoint
- Handles SSE stream parsing
- Updates UI with status messages and content
- Calls `onParamsReceived` callback when code and params are ready
- Maintains session ID for context

## Example Usage

1. User: "Create a breakout strategy with RSI filter"
2. AI: [Streams status] "Understanding your strategy requirements..."
3. AI: [Streams content] "I'll create a breakout strategy..."
4. AI: [Streams complete] Returns code and parameters
5. Frontend: Updates ParamPanel with extracted parameters
