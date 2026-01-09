"""
Restaurant QnA Flow using LangGraph
Conversational interface to collect user requirements
"""

from typing import TypedDict, Annotated, Optional, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import os

from dotenv import load_dotenv
load_dotenv()
# ============================================================================
# STATE DEFINITION
# ============================================================================

class UserRequirements(TypedDict):
    """User's restaurant requirements"""
    # Mandatory
    budget: Optional[int]           # Price per person
    timing: Optional[str]           # "dinner" or "brunch"
    num_guests: Optional[int]       # Number of people
    
    # Optional (can skip)
    cuisine: Optional[str]          # e.g., "Italian", "Japanese"
    vibe: Optional[str]             # e.g., "romantic", "casual"
    occasion: Optional[str]         # e.g., "date night", "business"
    dietary: Optional[str]          # e.g., "vegetarian", "vegan"
    specific_items: Optional[str]   # e.g., "duck", "pasta"


class ConversationState(TypedDict):
    """Overall conversation state"""
    conversation_history: List[dict]  # List of {"role": "user/assistant", "content": "..."}
    user_requirements: UserRequirements
    questions_asked: List[str]        # Track which fields we've asked about
    is_complete: bool                 # All mandatory fields collected
    current_question: Optional[str]   # The question we just asked


# ============================================================================
# LLM SETUP
# ============================================================================

def get_llm():
    """Get OpenAI LLM instance"""
    # You can also use local models with Ollama
    return ChatOpenAI(
        model="gpt-4o-mini",  # or "gpt-3.5-turbo" for cheaper
        temperature=0.7
    )


# ============================================================================
# GRAPH NODES
# ============================================================================

def check_initial_state(state: ConversationState) -> ConversationState:
    """
    START node: Initialize conversation with open-ended question
    """
    # If first time, add greeting with open question
    if not state["conversation_history"]:
        greeting = "Hi! 👋 I'm here to help you find the perfect restaurant.\n\nTell me what you're looking for! You can share anything like number of people, budget, cuisine preference, occasion, or vibe you want. 😊"
        
        state["conversation_history"].append({
            "role": "assistant",
            "content": greeting
        })
    
    return state


def generate_next_question(state: ConversationState) -> ConversationState:
    """
    QUESTION_GENERATOR node: Use LLM to generate next question dynamically
    """
    llm = get_llm()
    
    # Get current requirements
    reqs = state["user_requirements"]
    asked = state["questions_asked"]
    
    # Count what we have
    mandatory_complete = (
        reqs.get("budget") is not None and
        reqs.get("timing") is not None and
        reqs.get("num_guests") is not None
    )
    
    # Determine what's missing
    mandatory_missing = []
    if reqs.get("budget") is None:
        mandatory_missing.append("budget")
    if reqs.get("timing") is None:
        mandatory_missing.append("meal timing (dinner/brunch)")
    if reqs.get("num_guests") is None:
        mandatory_missing.append("number of guests")
    
    # Build context for LLM based on what's needed
    if mandatory_missing:
        # Still need mandatory info
        context = f"""
You are a friendly, conversational restaurant assistant.

What we know so far:
- Budget: {reqs.get('budget', '❌ Missing')}
- Timing: {reqs.get('timing', '❌ Missing')}
- Guests: {reqs.get('num_guests', '❌ Missing')}
- Cuisine: {reqs.get('cuisine', 'Not specified')}
- Vibe: {reqs.get('vibe', 'Not specified')}
- Occasion: {reqs.get('occasion', 'Not specified')}

MISSING: {', '.join(mandatory_missing)}

Ask for ONE missing field in a natural, friendly way.
Keep it short and conversational.

Examples:
- "What's your budget per person?" 
- "How many people are joining?"
- "Is this for dinner or brunch?"

Generate just the question, nothing else.
"""
    elif "optional_asked" not in asked:
        # Mandatory done, ask ONE open question for optional preferences
        context = f"""
You are a friendly restaurant assistant.

Great! We have the essentials:
✓ Budget: ${reqs['budget']} per person
✓ Timing: {reqs['timing'].title()}
✓ Guests: {reqs['num_guests']}

Now ask ONE open-ended question about their preferences for cuisine, vibe, occasion, or dietary needs.
Make it conversational and let them share what matters to them.

Example:
"Perfect! Do you have any preferences for cuisine, vibe, or occasion? Or any dietary needs? (Feel free to share what matters most to you, or say 'no preference'!)"

Generate just the question, nothing else.
"""
        # Mark that we asked about optional
        state["questions_asked"].append("optional_asked")
    else:
        # Everything collected
        context = """
Perfect! We have everything needed. 
Generate a friendly message:
"Great! I have everything I need. Ready to find the perfect restaurants for you! 🔍"
"""
    
    # Call LLM
    response = llm.invoke([
        SystemMessage(content="You are a helpful, conversational restaurant assistant. Keep responses short and friendly."),
        HumanMessage(content=context)
    ])
    
    question = response.content.strip()
    
    # Update state
    state["current_question"] = question
    state["conversation_history"].append({
        "role": "assistant",
        "content": question
    })
    
    return state


def process_user_answer(state: ConversationState, user_message: str) -> ConversationState:
    """
    ANSWER_PROCESSOR node: Extract structured data from user's answer using LLM
    Handles both new information and updates to existing information
    """
    llm = get_llm()
    
    # Add user message to history
    state["conversation_history"].append({
        "role": "user",
        "content": user_message
    })
    
    # Check if user wants to skip
    if user_message.lower().strip() in ["skip", "skip this", "pass", "next", "no preference"]:
        # Mark last asked question as skipped
        return state
    
    # Extract information using LLM (handles updates too)
    extraction_prompt = f"""
Extract restaurant preference information from the user's message.

User said: "{user_message}"

IMPORTANT: User might be UPDATING previous info. Look for words like "change", "actually", "update", "make it".

Current info:
- Budget: {state["user_requirements"].get("budget")}
- Timing: {state["user_requirements"].get("timing")}
- Guests: {state["user_requirements"].get("num_guests")}
- Cuisine: {state["user_requirements"].get("cuisine")}
- Vibe: {state["user_requirements"].get("vibe")}
- Occasion: {state["user_requirements"].get("occasion")}
- Dietary: {state["user_requirements"].get("dietary")}
- Specific items: {state["user_requirements"].get("specific_items")}

Extract ANY relevant information and return ONLY a JSON object:
{{
  "budget": number or null,
  "timing": "dinner" or "brunch" or null,
  "num_guests": number or null,
  "cuisine": string or null,
  "vibe": string or null,
  "occasion": string or null,
  "dietary": string or null,
  "specific_items": string or null
}}

Examples:
"Italian for 4 people" → {{"cuisine": "italian", "num_guests": 4}}
"Around $80 per person" → {{"budget": 80}}
"Romantic dinner" → {{"vibe": "romantic", "timing": "dinner"}}
"We're 6 people" → {{"num_guests": 6}}
"Change budget to 600" → {{"budget": 600}}
"Actually make it $100" → {{"budget": 100}}
"Update: we're now 8 people" → {{"num_guests": 8}}

Return ONLY the JSON, no other text.
"""
    
    response = llm.invoke([
        SystemMessage(content="You extract structured data from text. Return only valid JSON. Handle updates by returning the NEW value."),
        HumanMessage(content=extraction_prompt)
    ])
    
    # Parse extracted data
    try:
        # Clean response - remove markdown code blocks if present
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        extracted = json.loads(content)
        
        # Update user requirements with extracted data
        for key, value in extracted.items():
            if value is not None and key in state["user_requirements"]:
                # Update the value (handles both new and updates)
                state["user_requirements"][key] = value
                
                # Track that we got info about this field
                if key not in state["questions_asked"]:
                    state["questions_asked"].append(key)
    
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse LLM response: {response.content}")
        print(f"Error: {e}")
    
    return state


def check_completion(state: ConversationState) -> ConversationState:
    """
    COMPLETION_CHECKER node: Check if we have minimum info to search
    We're "ready to search" when all mandatory fields are filled
    But user can still provide more info or make changes
    """
    reqs = state["user_requirements"]
    
    # Check mandatory fields
    mandatory_complete = (
        reqs.get("budget") is not None and
        reqs.get("timing") is not None and
        reqs.get("num_guests") is not None
    )
    
    state["is_complete"] = mandatory_complete
    
    return state


def route_next(state: ConversationState) -> str:
    """
    Router: Decide next node based on completion status
    """
    if state["is_complete"]:
        return "complete"
    else:
        return "continue"


# ============================================================================
# BUILD GRAPH
# ============================================================================

def create_qna_graph():
    """
    Create the LangGraph workflow
    
    Note: This graph is designed to be called ONCE per interaction:
    1. Call once at start → generates first question
    2. User answers (in Streamlit)
    3. Process answer → Call again → generates next question
    4. Repeat until complete
    """
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("start", check_initial_state)
    workflow.add_node("generate_question", generate_next_question)
    workflow.add_node("check_completion", check_completion)
    
    # Set entry point
    workflow.set_entry_point("start")
    
    # Add edges
    workflow.add_edge("start", "check_completion")
    
    # Conditional edge based on completion
    workflow.add_conditional_edges(
        "check_completion",
        route_next,
        {
            "continue": "generate_question",
            "complete": END
        }
    )
    
    # After generating question, END (wait for user input)
    workflow.add_edge("generate_question", END)
    
    return workflow.compile()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_state() -> ConversationState:
    """Create initial empty state"""
    return {
        "conversation_history": [],
        "user_requirements": {
            "budget": None,
            "timing": None,
            "num_guests": None,
            "cuisine": None,
            "vibe": None,
            "occasion": None,
            "dietary": None,
            "specific_items": None
        },
        "questions_asked": [],
        "is_complete": False,
        "current_question": None
    }


def format_requirements(reqs: UserRequirements) -> str:
    """Format requirements for display"""
    output = []
    
    if reqs.get("num_guests"):
        output.append(f"👥 Guests: {reqs['num_guests']}")
    
    if reqs.get("budget"):
        output.append(f"💰 Budget: ${reqs['budget']} per person")
    
    if reqs.get("timing"):
        output.append(f"🕐 Timing: {reqs['timing'].title()}")
    
    if reqs.get("cuisine"):
        output.append(f"🍽️ Cuisine: {reqs['cuisine'].title()}")
    
    if reqs.get("vibe"):
        output.append(f"✨ Vibe: {reqs['vibe'].title()}")
    
    if reqs.get("occasion"):
        output.append(f"🎉 Occasion: {reqs['occasion'].title()}")
    
    if reqs.get("dietary"):
        output.append(f"🥗 Dietary: {reqs['dietary'].title()}")
    
    if reqs.get("specific_items"):
        output.append(f"🍴 Items: {reqs['specific_items'].title()}")
    
    return "\n".join(output) if output else "No information collected yet"


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    # Test the graph
    import os
    
    # Set OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    print("🍽️  Restaurant QnA System Test\n")
    
    # Initialize
    state = initialize_state()
    graph = create_qna_graph()
    
    # Start conversation
    state = graph.invoke(state)
    
    print(f"Bot: {state['conversation_history'][-1]['content']}\n")
    
    # Simulate conversation
    test_inputs = [
        "We're a group of 4 people",
        "Around $100 per person",
        "Dinner time",
        "Italian food",
        "skip"  # Skip remaining
    ]
    
    for user_input in test_inputs:
        print(f"You: {user_input}")
        
        # Process answer
        state = process_user_answer(state, user_input)
        
        # Generate next question or complete
        state = graph.invoke(state)
        
        print(f"Bot: {state['conversation_history'][-1]['content']}\n")
        
        print("📊 Collected so far:")
        print(format_requirements(state["user_requirements"]))
        print("-" * 50)
        
        if state["is_complete"]:
            print("\n✅ All mandatory information collected!")
            break
    
    print("\n📋 Final Requirements:")
    print(format_requirements(state["user_requirements"]))