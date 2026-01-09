"""
Streamlit App for Restaurant Finder QnA Flow
Two-column layout with editable requirements panel
"""

import streamlit as st
from qna_graph import (
    create_qna_graph,
    initialize_state,
    process_user_answer,
)

try:
    from search_engine import search_restaurants
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False
    print("⚠️  search_engine.py not found - search disabled")


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Restaurant Finder",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none}
    .stChatMessage {padding: 1rem; border-radius: 0.5rem;}
    h1 {text-align: center; margin-bottom: 0.5rem;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================

if "state" not in st.session_state:
    st.session_state.state = initialize_state()
    st.session_state.graph = create_qna_graph()
    st.session_state.state = st.session_state.graph.invoke(st.session_state.state)

if "ready_to_search" not in st.session_state:
    st.session_state.ready_to_search = False

if "search_results" not in st.session_state:
    st.session_state.search_results = None

if "searching" not in st.session_state:
    st.session_state.searching = False


# ============================================================================
# FUNCTIONS
# ============================================================================

def send_message():
    """Handle user message"""
    user_msg = st.session_state.user_input_widget
    if user_msg.strip():
        st.session_state.state = process_user_answer(st.session_state.state, user_msg)
        st.session_state.state = st.session_state.graph.invoke(st.session_state.state)


def update_field(field, value):
    """Update a requirement field"""
    st.session_state.state["user_requirements"][field] = value
    st.session_state.state["conversation_history"].append({
        "role": "assistant",
        "content": f"✅ Updated {field.replace('_', ' ')} to: {value}"
    })


def reset_all():
    """Reset conversation"""
    st.session_state.state = initialize_state()
    st.session_state.graph = create_qna_graph()
    st.session_state.state = st.session_state.graph.invoke(st.session_state.state)
    st.session_state.ready_to_search = False
    st.session_state.search_results = None


def ready_to_search():
    """Mark ready for search"""
    st.session_state.ready_to_search = True


def perform_search():
    """Actually perform the restaurant search"""
    if not SEARCH_AVAILABLE:
        st.error("Search engine not available. Please ensure search_engine.py and restaurants_search_index.json are in the same directory.")
        return
    
    st.session_state.searching = True
    
    try:
        # Get requirements
        reqs = st.session_state.state["user_requirements"]
        
        # Perform search
        with st.spinner("🔍 Searching for perfect restaurants..."):
            results = search_restaurants(reqs)
        
        st.session_state.search_results = results
        st.session_state.searching = False
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        st.session_state.searching = False


def ready_to_search():
    """Mark ready for search"""
    st.session_state.ready_to_search = True


# ============================================================================
# MAIN UI
# ============================================================================

# Title
st.title("🍽️ Restaurant Finder")

# Status
reqs = st.session_state.state["user_requirements"]
mandatory_count = sum([
    reqs.get("num_guests") is not None,
    reqs.get("budget") is not None,
    reqs.get("timing") is not None
])

if st.session_state.ready_to_search:
    st.success("✅ Ready to search for restaurants!")
elif st.session_state.state["is_complete"]:
    st.info("💬 You can add more details or click 'Find Restaurants' when ready!")
else:
    st.info(f"💬 Collecting information... ({mandatory_count}/3 required)")

st.divider()

# Two Columns
col1, col2 = st.columns([2, 1])

# ============================================================================
# LEFT: CHAT
# ============================================================================

with col1:
    st.subheader("💬 Conversation")
    
    chat_container = st.container(height=450)
    with chat_container:
        for msg in st.session_state.state["conversation_history"]:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                st.write(msg["content"])
    
    if not st.session_state.ready_to_search:
        st.chat_input("Type your message...", key="user_input_widget", on_submit=send_message)
        st.caption("💡 Share multiple details at once, or edit using the panel on the right →")
        
        btn_col1, btn_col2 = st.columns([3, 1])
        with btn_col1:
            if st.session_state.state["is_complete"]:
                if st.button("🔍 Find Restaurants", type="primary", use_container_width=True):
                    ready_to_search()
                    st.rerun()
        with btn_col2:
            if st.button("🔄 Reset", use_container_width=True):
                reset_all()
                st.rerun()
    else:
        # Show results if available
        if st.session_state.search_results:
            st.success("✅ Found your perfect restaurants!")
            
            # Display each recommendation
            for i, rec in enumerate(st.session_state.search_results, 1):
                with st.container(border=True):
                    st.markdown(f"### 🏆 #{i}: {rec['restaurant_name']}")
                    st.caption(f"{', '.join(rec['cuisines'])} • ${rec['price_range']['min']}-${rec['price_range']['max']}/person")
                    
                    st.markdown(f"**🍽️ Recommended: {rec['recommended_menu']}** (${rec['menu_price']}/person)")
                    
                    st.markdown(f"**✨ Why it's perfect:**")
                    st.write(rec['reasoning'])
                    
                    st.markdown(f"**🎯 {rec['perfect_for']}**")
                    
                    st.markdown("**🍴 Must-try dishes:**")
                    for dish in rec['dish_highlights']:
                        st.markdown(f"• {dish}")
                    
                    with st.expander("More Details"):
                        st.write(f"**Vibes:** {', '.join(rec['vibes'][:4])}")
                        st.write(f"**Perfect for:** {', '.join(rec['occasions'][:4])}")
            
            # Action buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 New Search", use_container_width=True):
                    reset_all()
                    st.rerun()
            with col_b:
                if st.button("✏️ Refine", use_container_width=True):
                    st.session_state.ready_to_search = False
                    st.session_state.search_results = None
                    st.rerun()
        
        else:
            # No results yet, show search button
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔍 Search Now", type="primary", use_container_width=True, disabled=st.session_state.searching):
                    perform_search()
                    st.rerun()
            with col_b:
                if st.button("🔄 Start Over", use_container_width=True):
                    reset_all()
                    st.rerun()

# ============================================================================
# RIGHT: EDITABLE REQUIREMENTS
# ============================================================================

with col2:
    st.subheader("📋 Your Requirements")
    
    # ALL REQUIREMENTS IN ONE SECTION (no labels)
    with st.container(border=True):
        # Guests
        if reqs.get("num_guests"):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.success(f"👥 **{reqs['num_guests']}** guests")
            with col_b:
                if st.button("✏️", key="edit_guests"):
                    st.session_state.editing = "guests"
        else:
            st.warning("👥 Guests: Not set")
        
        if st.session_state.get("editing") == "guests":
            new_val = st.number_input("Guests:", 1, 50, reqs.get("num_guests", 2), key="new_guests")
            if st.button("💾 Save", key="save_guests"):
                update_field("num_guests", new_val)
                del st.session_state.editing
                st.rerun()
        
        st.divider()
        
        # Budget
        if reqs.get("budget"):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.success(f"💰 **${reqs['budget']}**/person")
            with col_b:
                if st.button("✏️", key="edit_budget"):
                    st.session_state.editing = "budget"
        else:
            st.warning("💰 Budget: Not set")
        
        if st.session_state.get("editing") == "budget":
            new_val = st.number_input("Budget ($):", 10, 1000, reqs.get("budget", 50), key="new_budget")
            if st.button("💾 Save", key="save_budget"):
                update_field("budget", new_val)
                del st.session_state.editing
                st.rerun()
        
        st.divider()
        
        # Timing
        if reqs.get("timing"):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.success(f"🕐 **{reqs['timing'].title()}**")
            with col_b:
                if st.button("✏️", key="edit_timing"):
                    st.session_state.editing = "timing"
        else:
            st.warning("🕐 Timing: Not set")
        
        if st.session_state.get("editing") == "timing":
            new_val = st.selectbox("Timing:", ["dinner", "brunch"], key="new_timing")
            if st.button("💾 Save", key="save_timing"):
                update_field("timing", new_val)
                del st.session_state.editing
                st.rerun()
        
        st.divider()
        
        # Cuisine
        col_a, col_b = st.columns([3, 1])
        with col_a:
            if reqs.get("cuisine"):
                st.info(f"🍽️ {reqs['cuisine'].title()}")
            else:
                st.text("🍽️ Cuisine: -")
        with col_b:
            if st.button("✏️", key="edit_cuisine"):
                st.session_state.editing = "cuisine"
        
        if st.session_state.get("editing") == "cuisine":
            new_val = st.text_input("Cuisine:", reqs.get("cuisine", ""), key="new_cuisine")
            if st.button("💾", key="save_cuisine"):
                update_field("cuisine", new_val or None)
                del st.session_state.editing
                st.rerun()
        
        st.divider()
        
        # Vibe
        col_a, col_b = st.columns([3, 1])
        with col_a:
            if reqs.get("vibe"):
                st.info(f"✨ {reqs['vibe'].title()}")
            else:
                st.text("✨ Vibe: -")
        with col_b:
            if st.button("✏️", key="edit_vibe"):
                st.session_state.editing = "vibe"
        
        if st.session_state.get("editing") == "vibe":
            new_val = st.text_input("Vibe:", reqs.get("vibe", ""), key="new_vibe")
            if st.button("💾", key="save_vibe"):
                update_field("vibe", new_val or None)
                del st.session_state.editing
                st.rerun()
        
        st.divider()
        
        # Occasion
        col_a, col_b = st.columns([3, 1])
        with col_a:
            if reqs.get("occasion"):
                st.info(f"🎉 {reqs['occasion'].title()}")
            else:
                st.text("🎉 Occasion: -")
        with col_b:
            if st.button("✏️", key="edit_occasion"):
                st.session_state.editing = "occasion"
        
        if st.session_state.get("editing") == "occasion":
            new_val = st.text_input("Occasion:", reqs.get("occasion", ""), key="new_occasion")
            if st.button("💾", key="save_occasion"):
                update_field("occasion", new_val or None)
                del st.session_state.editing
                st.rerun()
        
        st.divider()
        
        # Dietary
        col_a, col_b = st.columns([3, 1])
        with col_a:
            if reqs.get("dietary"):
                st.info(f"🥗 {reqs['dietary'].title()}")
            else:
                st.text("🥗 Dietary: -")
        with col_b:
            if st.button("✏️", key="edit_dietary"):
                st.session_state.editing = "dietary"
        
        if st.session_state.get("editing") == "dietary":
            new_val = st.text_input("Dietary:", reqs.get("dietary", ""), key="new_dietary")
            if st.button("💾", key="save_dietary"):
                update_field("dietary", new_val or None)
                del st.session_state.editing
                st.rerun()

st.divider()