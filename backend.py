# langgraph_backend.py – FITNESS & NUTRITION COACH EDITION (Gemini 2.5 Pro – Nov 22, 2025)
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")

# Gemini 2.5 Pro – best stable model as of Nov 22, 2025
MODEL_NAME = "gemini-2.5-pro"
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.7,
    google_api_key=api_key,
    convert_system_message_to_human=True
)

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# ── CLASSIFIER: Is this about Fitness / Nutrition / Workouts / Diet? ──
def classifier_node(state: ChatState):
    last_msg = state["messages"][-1].content
    prompt = (
        "Does this message relate to fitness, workouts, exercise, nutrition, diet, supplements, "
        "weight loss, muscle gain, sports performance, or healthy eating? "
        "Answer only 'yes' or 'no'.\n\nQuery: " + last_msg
    )
    response = llm.invoke(prompt)
    decision = "fitness_nutrition" if "yes" in response.content.lower() else "off_topic"
    return {"classification": decision}

# ── MAIN COACH NODE ──
def coach_node(state: ChatState):
    messages_with_system = [
        SystemMessage(content=(
            "You are an expert Fitness & Nutrition Coach. "
            "You give personalized, science-based advice on training programs, workout plans, "
            "nutrition, macros, meal ideas, supplements, recovery, and lifestyle habits. "
            "Always ask follow-up questions when needed (age, goals, experience, dietary restrictions, etc.). "
            "Be motivating, clear, and practical. Never give medical diagnoses."
        ))
    ] + state["messages"]

    try:
        response = llm.invoke(messages_with_system)
    except Exception as e:
        if "NotFound" in str(e) or "404" in str(e):
            print("Falling back to gemini-2.5-flash...")
            fallback = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=api_key)
            response = fallback.invoke(messages_with_system)
        else:
            raise e
    return {"messages": [response]}

# ── OFF-TOPIC NODE ──
def off_topic_node(state: ChatState):
    msg = (
        "I'm your dedicated Fitness & Nutrition Coach! "
        "I can help with workout plans, meal ideas, macros, supplements, and reaching your goals 💪 "
        "Ask me anything about training or nutrition!"
    )
    return {"messages": [AIMessage(content=msg)]}

# ── BUILD GRAPH ──
graph = StateGraph(ChatState)
graph.add_node("classifier", classifier_node)
graph.add_node("coach", coach_node)
graph.add_node("off_topic", off_topic_node)

graph.add_edge(START, "classifier")
graph.add_conditional_edges(
    "classifier",
    lambda s: s.get("classification", "off_topic"),
    {"fitness_nutrition": "coach", "off_topic": "off_topic"}
)
graph.add_edge("coach", END)
graph.add_edge("off_topic", END)

memory = MemorySaver()
chatbot = graph.compile(checkpointer=memory)

# ── PRETTY TERMINAL GRAPH ──31
try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console(force_terminal=True)
    def print_graph_in_terminal():
        console.print(Panel(
            """
[bold cyan]START[/] ───► [bold yellow on blue]classifier[/] ───┬─► [bold green]Fitness & Nutrition Coach[/] (gemini-2.5-pro) ───► [bold cyan]END[/]
                                         └─► [bold red]off_topic[/] ───► [bold cyan]END[/]

    Persistent memory • Full history • Personalized fitness & nutrition advice
            """,
            title="[bold magenta]FITNESS COACH LANGGRAPH – ACTIVE 💪[/]",
            border_style="bright_magenta",
        ))
except ImportError:
    def print_graph_in_terminal():
        print("\n" + "="*80)
        print("FITNESS & NUTRITIONCOACH: START → classifier → coach / off_topic → END")
        print("Full conversation history ✓ | Powered by Gemini 2.5 Pro")
        print("="*80 + "\n")

# ── QUICK TEST (run python langgraph_backend.py) ──
if __name__ == "__main__":
    print_graph_in_terminal()
    config = {"configurable": {"thread_id": "test-fit-001"}}
    
    # On-topic test
    resp = chatbot.invoke({"messages": [HumanMessage(content="I want to build muscle and eat in a surplus. Help me plan meals.")]}, config=config)
    print("Coach response:", resp["messages"][-1].content[:200] + "...")
    
    # Off-topic test (new thread)
    resp2 = chatbot.invoke({"messages": [HumanMessage(content="What's the capital of France?")]}, config={"configurable": {"thread_id": "test-002"}})
    print("\nOff-topic response:", resp2["messages"][-1].content)