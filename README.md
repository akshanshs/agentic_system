Building AI agents

## LangGraph Flight Booking Example (RAG + Tools)

This repo now includes `flight_booking_langgraph.py`, a LangGraph workflow that:

- Uses a **RAG step** (`rag_node`) to retrieve airline policy context from local docs.
- Uses a **tool-calling agent node** (`agent_node`) powered by `gpt-4o-mini`.
- Uses **tools** for booking operations:
  - `search_flights`
  - `get_price_breakdown`
  - `create_booking`

### Run

```bash
uv run python flight_booking_langgraph.py
```

Set your API key first:

```bash
export OPENAI_API_KEY="your-key"
```
