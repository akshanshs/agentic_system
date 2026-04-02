"""LangGraph workflow for flight booking with RAG + tools.

Run:
    uv run python flight_booking_langgraph.py

Environment:
    OPENAI_API_KEY=<your key>
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# -----------------------------
# Lightweight local RAG corpus
# -----------------------------
RAG_DOCS = [
    {
        "id": "baggage_policy",
        "text": (
            "Carry-on baggage: one cabin bag up to 8kg and one personal item. "
            "Checked baggage allowance is fare-dependent: Economy Saver includes 0 checked bags, "
            "Economy Flex includes 1 x 23kg bag, and Business includes 2 x 32kg bags."
        ),
    },
    {
        "id": "change_policy",
        "text": (
            "Flight changes are allowed up to 3 hours before departure. "
            "Economy Saver has a $120 change fee, Economy Flex has a $40 fee, and Business has no change fee."
        ),
    },
    {
        "id": "refund_policy",
        "text": (
            "Refund rules: Economy Saver is non-refundable, Economy Flex is refundable with a $75 processing fee, "
            "Business is fully refundable before departure."
        ),
    },
    {
        "id": "checkin_policy",
        "text": (
            "Online check-in opens 24 hours before departure and closes 90 minutes before departure. "
            "Airport check-in closes 60 minutes before domestic and 75 minutes before international flights."
        ),
    },
]


def simple_retrieve(query: str, k: int = 2) -> str:
    """Very small keyword-overlap retriever for local policy docs."""
    query_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    scored: list[tuple[int, dict]] = []

    for doc in RAG_DOCS:
        terms = set(re.findall(r"[a-zA-Z0-9]+", doc["text"].lower()))
        score = len(query_terms & terms)
        scored.append((score, doc))

    top = [d for score, d in sorted(scored, key=lambda x: x[0], reverse=True)[:k] if score > 0]
    if not top:
        return "No matching policy documents found."

    return "\n\n".join(f"[{doc['id']}] {doc['text']}" for doc in top)


# -----------------------------
# Tools (mocked backend actions)
# -----------------------------
@tool

def search_flights(origin: str, destination: str, depart_date: str, passengers: int = 1) -> str:
    """Search available flights by route/date and return JSON list of options."""
    # Mock inventory
    flights = [
        {
            "flight_id": "AX102",
            "origin": origin.upper(),
            "destination": destination.upper(),
            "depart_date": depart_date,
            "depart_time": "08:30",
            "arrive_time": "11:25",
            "fare": "Economy Saver",
            "price_usd": 249,
            "seats_left": 5,
        },
        {
            "flight_id": "AX220",
            "origin": origin.upper(),
            "destination": destination.upper(),
            "depart_date": depart_date,
            "depart_time": "13:50",
            "arrive_time": "16:40",
            "fare": "Economy Flex",
            "price_usd": 319,
            "seats_left": 9,
        },
        {
            "flight_id": "AX908",
            "origin": origin.upper(),
            "destination": destination.upper(),
            "depart_date": depart_date,
            "depart_time": "19:10",
            "arrive_time": "21:58",
            "fare": "Business",
            "price_usd": 689,
            "seats_left": 2,
        },
    ]
    available = [f for f in flights if f["seats_left"] >= passengers]
    return json.dumps(available)


@tool

def get_price_breakdown(base_price_usd: int, passengers: int = 1) -> str:
    """Return total price with taxes/fees in JSON."""
    taxes = int(base_price_usd * 0.14)
    service_fee = 18
    total = (base_price_usd + taxes + service_fee) * passengers
    return json.dumps(
        {
            "base_price_usd": base_price_usd,
            "taxes_usd": taxes,
            "service_fee_usd": service_fee,
            "passengers": passengers,
            "grand_total_usd": total,
        }
    )


@tool

def create_booking(
    flight_id: str,
    traveler_name: str,
    traveler_email: str,
    passengers: int = 1,
) -> str:
    """Create a booking and return a confirmation number."""
    booking_id = f"BK-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    return json.dumps(
        {
            "booking_id": booking_id,
            "flight_id": flight_id,
            "traveler_name": traveler_name,
            "traveler_email": traveler_email,
            "passengers": passengers,
            "status": "CONFIRMED",
        }
    )


tools = [search_flights, get_price_breakdown, create_booking]


# -----------------------------
# LangGraph state + nodes
# -----------------------------
class BookingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    rag_context: str


def rag_node(state: BookingState) -> dict:
    """Retrieve policy context from local docs using latest user message."""
    latest_user = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    query = latest_user.content if latest_user else "flight booking policies"
    context = simple_retrieve(str(query), k=2)
    return {"rag_context": context}


def agent_node(state: BookingState) -> dict:
    """LLM planner node. It can answer directly or call tools."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    system = SystemMessage(
        content=(
            "You are a flight booking assistant. Use tools for live booking actions. "
            "Use the provided RAG context for policy questions (baggage/changes/refunds/check-in). "
            "If user asks to book, collect required details first: origin, destination, date, passengers, "
            "traveler_name, traveler_email. Then search flights, quote total, and ask confirmation before booking. "
            "When giving policy answers, cite the policy section ID in brackets.\n\n"
            f"RAG context:\n{state.get('rag_context', 'No context')}"
        )
    )

    response = llm.invoke([system, *state["messages"]])
    return {"messages": [response]}


def route_after_agent(state: BookingState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


builder = StateGraph(BookingState)
builder.add_node("rag", rag_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("rag")
builder.add_edge("rag", "agent")
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "end": END,
    },
)
builder.add_edge("tools", "agent")

flight_booking_graph = builder.compile()


def run_demo() -> None:
    """Simple CLI demo for the graph."""
    print("Flight Booking Assistant (LangGraph + RAG + Tools). Type 'exit' to quit.")
    state: BookingState = {"messages": [], "rag_context": ""}

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        state_update = flight_booking_graph.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "rag_context": state.get("rag_context", ""),
            }
        )
        state = {
            "messages": state.get("messages", []) + state_update["messages"],
            "rag_context": state_update.get("rag_context", state.get("rag_context", "")),
        }

        ai_msg = next(
            (m for m in reversed(state_update["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        if ai_msg:
            print(f"Assistant: {ai_msg.content}")


if __name__ == "__main__":
    missing = [k for k in ["OPENAI_API_KEY"] if not os.getenv(k)]
    if missing:
        print(f"Missing env vars: {missing}. Please set them before running.")
    else:
        run_demo()
