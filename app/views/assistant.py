"""
AI Assistant Page
=================
Chat interface with Databricks Genie API integration.

API Reference: https://docs.databricks.com/api/azure/workspace/genie
"""

import streamlit as st
from config import genie_config, databricks_config
from data.genie_client import get_genie_client, GenieResponse


def render_assistant_page():
    """Render the AI Assistant page."""
    
    st.markdown("""
    <h1 style="margin-bottom: 8px;">🤖 AI Assistant</h1>
    <p style="font-size: 1rem; color: #6b6b69; margin-bottom: 16px;">
        Chat with Databricks Genie to analyze scrap data using natural language
    </p>
    """, unsafe_allow_html=True)
    
    render_chat_interface()


def format_genie_response(response: GenieResponse) -> None:
    """Format and display a Genie response with proper styling."""
    
    # Main content/description
    if response.content:
        st.markdown(response.content)
    
    # Data table
    if response.data_table:
        st.markdown(f"**Results** ({response.row_count} rows):")
        st.markdown(response.data_table)
    
    # SQL query in expander (hidden by default like Databricks UI)
    if response.sql_query:
        with st.expander("📝 View SQL Query", expanded=False):
            st.code(response.sql_query, language="sql")
    
    # Suggested follow-up questions
    if response.suggested_questions:
        st.markdown("---")
        st.markdown("**💡 Suggested follow-ups:**")
        for q in response.suggested_questions:
            if st.button(f"→ {q}", key=f"suggest_{hash(q)}", use_container_width=True):
                st.session_state.pending_question = q
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()


def render_chat_interface():
    """Render the chat interface with Genie API integration."""
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"""👋 Hello! I'm connected to **{genie_config.space_name}** via the Databricks Genie API.

I can query your scrap data directly. Try asking:

• "What's our total scrap cost this year?"
• "Show me the top 5 plants by scrap cost"
• "Which suppliers have quality scores below 0.85?"
• "Compare defect rates across all shifts"

Ask me anything about your scrap data!""",
                "type": "intro"
            }
        ]
    
    if "genie_client" not in st.session_state:
        st.session_state.genie_client = get_genie_client()
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    
    # Check for pending question from quick buttons
    pending_prompt = st.session_state.pending_question
    if pending_prompt:
        st.session_state.pending_question = None  # Clear immediately
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "genie_response" and "response_obj" in msg:
                # Render structured Genie response
                format_genie_response(msg["response_obj"])
            else:
                # Render plain text
                st.markdown(msg["content"])
    
    # Quick Questions - show above chat input if no conversation yet
    if len(st.session_state.messages) <= 1:
        st.markdown("##### ⚡ Quick Questions")
        
        quick_questions = [
            ("📊 YTD Summary", "What's our total scrap cost year to date?"),
            ("🏭 Plant Ranking", "Show the top 5 plants by scrap cost this year"),
            ("📦 Top Defects", "What are the top 5 defect types by cost?"),
            ("👥 Supplier Issues", "Which suppliers have the highest scrap rates?"),
            ("📋 8D Reports", "How many 8D reports are currently open?"),
        ]
        
        cols = st.columns(5)
        for i, (label, question) in enumerate(quick_questions):
            with cols[i]:
                if st.button(label, key=f"quick_{label}", use_container_width=True):
                    st.session_state.pending_question = question
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.rerun()
    
    # Get prompt from either chat input or pending question
    chat_prompt = st.chat_input("Ask about scrap data...", disabled=st.session_state.processing)
    prompt = pending_prompt or chat_prompt
    
    # Chat input
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get Genie response
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_placeholder.markdown("🔄 *Sending request to Genie...*")
            
            st.session_state.processing = True
            
            try:
                client = st.session_state.genie_client
                
                status_placeholder.markdown("⏳ *Analyzing your question (this may take up to 30 seconds)...*")
                
                message_id, response = client.send_message(prompt)
                
                status_placeholder.empty()
                
                if response:
                    # Display the formatted response
                    format_genie_response(response)
                    
                    # Store in messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": client.format_response_markdown(response),
                        "type": "genie_response",
                        "response_obj": response
                    })
                else:
                    error_msg = "Sorry, I couldn't get a response. Please try again."
                    st.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            except Exception as e:
                status_placeholder.empty()
                error_msg = f"Error connecting to Genie: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ {error_msg}"
                })
            finally:
                st.session_state.processing = False
    
    # Action buttons at bottom
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🗑️ Clear Chat", key="clear_chat_btn", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]
            st.session_state.genie_client.reset_conversation()
            st.rerun()
    
    with col2:
        if st.button("🔄 New Conversation", key="new_convo_btn", use_container_width=True):
            st.session_state.genie_client.reset_conversation()
            st.toast("Started new conversation", icon="✅")
    
    with col3:
        # Tips expander inline
        with st.expander("💡 Tips for Better Results"):
            st.markdown("""
            **Be specific:** "Show total scrap cost by plant for 2024" ✅  
            **Use time periods:** "this month", "last quarter", "YTD"  
            **Ask for comparisons:** "Compare scrap rates between shifts"
            """)


def is_genie_connected() -> bool:
    """Check if Genie API is configured and accessible."""
    return genie_config.is_configured()


def get_genie_status_html() -> str:
    """Get HTML for Genie connection status (for sidebar)."""
    if genie_config.is_configured():
        return f"""
        <div style="margin-top: 8px;">
            <span style="color: #10b981; font-size: 0.8rem;">✅ Genie Connected</span>
            <p style="color: #6b6b69; font-size: 0.7rem; margin: 4px 0 0 0;">
                Space: {genie_config.space_name}
            </p>
        </div>
        """
    else:
        return '<span style="color: #f59e0b; font-size: 0.8rem;">⚠️ Genie Not Configured</span>'
