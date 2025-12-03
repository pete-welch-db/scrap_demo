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
    <h1>🤖 AI Assistant</h1>
    <p style="font-size: 1.1rem; color: #64748b; margin-bottom: 24px;">
        Chat with Databricks Genie to analyze scrap data using natural language
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_chat_interface()
    
    with col2:
        render_ai_integrations()


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
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()


def render_chat_interface():
    """Render the chat interface with Genie API integration."""
    
    st.markdown("### 💬 Chat with Genie")
    
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
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "genie_response" and "response_obj" in msg:
                # Render structured Genie response
                format_genie_response(msg["response_obj"])
            else:
                # Render plain text
                st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about scrap data...", disabled=st.session_state.processing):
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
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", key="clear_chat_btn", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]
            st.session_state.genie_client.reset_conversation()
            st.rerun()
    
    with col2:
        if st.button("🔄 New Conversation", key="new_convo_btn", use_container_width=True):
            st.session_state.genie_client.reset_conversation()
            st.toast("Started new conversation", icon="✅")


def render_ai_integrations():
    """Render the AI integrations panel."""
    
    st.markdown("### 🔌 Genie Connection")
    
    # Genie status
    genie_configured = genie_config.is_configured()
    genie_url = f"https://{databricks_config.host}/genie/rooms/{genie_config.space_id}"
    
    if genie_configured:
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <span style="font-size: 1.5rem;">🧞</span>
                <h4 style="margin: 0; color: #f1f5f9;">{genie_config.space_name}</h4>
            </div>
            <div style="margin-bottom: 12px; padding: 8px 12px; background: #1a1a2e; border-radius: 8px;">
                <span style="color: #10b981;">✅ API Connected</span>
            </div>
            <p style="color: #64748b; font-size: 0.75rem; margin: 0;">
                Space ID: <code style="color: #a855f7;">{genie_config.space_id[:16]}...</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.link_button(
            "🔗 Open in Databricks",
            genie_url,
            use_container_width=True
        )
    else:
        st.warning("Genie not configured. Set GENIE_SPACE_ID in .env")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### ⚡ Quick Questions")
    
    quick_questions = [
        ("📊 YTD Summary", "What's our total scrap cost year to date?"),
        ("🏭 Plant Ranking", "Show the top 5 plants by scrap cost this year"),
        ("📦 Top Defects", "What are the top 5 defect types by cost?"),
        ("👥 Supplier Issues", "Which suppliers have the highest scrap rates?"),
        ("📋 8D Reports", "How many 8D reports are currently open?"),
    ]
    
    for label, question in quick_questions:
        if st.button(label, key=f"quick_{label}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    st.markdown("---")
    
    # Tips
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        **Be specific with your questions:**
        - ✅ "Show total scrap cost by plant for 2024"
        - ❌ "Show me data"
        
        **Use time periods:**
        - "this month", "last quarter", "YTD", "2024"
        
        **Ask for comparisons:**
        - "Compare scrap rates between Shift 1 and Shift 2"
        
        **Request rankings:**
        - "Top 10 parts by scrap cost"
        - "Worst performing suppliers"
        """)
    
    # Debug mode toggle
    with st.expander("🔧 Advanced"):
        debug_mode = st.checkbox("Enable Debug Mode", value=False, key="genie_debug")
        if "genie_client" in st.session_state:
            st.session_state.genie_client.debug = debug_mode
        
        if st.button("🔄 Reset Connection", key="reset_genie"):
            if "genie_client" in st.session_state:
                st.session_state.genie_client.reset_conversation()
                st.toast("Genie connection reset", icon="✅")
