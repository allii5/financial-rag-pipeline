
import uuid
import streamlit as st


from assistant import CarFinanceAssistant

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistani Car Finance Assistant",
    page_icon="🚙",
    layout="centered",
    initial_sidebar_state="expanded"
)


if "assistant" not in st.session_state:
    with st.spinner("Initializing AI Brain and connecting to ChromaDB..."):
        try:
            st.session_state.assistant = CarFinanceAssistant.build()
        except RuntimeError as e:
            st.error(f"Failed to start assistant: {e}")
            st.stop()

# Generate a unique session ID for this user's browser tab to isolate LangChain memory
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize the UI chat history with a welcoming default message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Assalam o Alaikum! I am your dedicated Car Finance Consultant. Are you looking for conventional or Islamic (Shariah-compliant) financing today?"
        }
    ]


# ─── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🚙 Car Finance Bot")
    st.markdown(
        "Welcome to the Pakistani Car Finance Consultant. "
        "I am trained on the 2025 Car Financing Guide covering 22 major banks."
    )
    
    st.markdown("### Features")
    st.markdown(
        """
        - 🕌 **Islamic & Conventional** rates
        - 📊 **KIBOR** tracking & SBP guidelines
        - 💰 **Down Payment** & tenure comparisons
        """
    )
    
    st.divider()
    
    # Clear conversation resets both the UI and the backend LangChain memory
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.assistant.clear_session(st.session_state.session_id)
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Conversation cleared. How can I help you today?"
            }
        ]
        st.rerun()


# ─── Main UI Header ───────────────────────────────────────────────────────────
st.title("🚙 Pakistani Car Finance Consultant")
st.caption("Expert RAG guidance on auto loans, KIBOR rates, and Islamic financing (2025 SBP Data).")
st.divider()


# ─── Render Existing Chat History ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─── Chat Input & Streaming Execution ─────────────────────────────────────────
if prompt := st.chat_input("Ask about Meezan Bank, used car limits, or down payments..."):
    
    # 1. Display user message in UI instantly
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Stream the assistant's response
    with st.chat_message("assistant"):
        # Fetch the generator from the backend RAG chain
        stream_generator = st.session_state.assistant.stream(
            user_input=prompt, 
            session_id=st.session_state.session_id
        )
        
        # st.write_stream automatically handles the loop and typewriter effect!
        # It returns the fully assembled string once the stream is complete.
        full_response = st.write_stream(stream_generator)



    # 3. Save the final response to UI memory
    # We use str() to satisfy Pylance's strict type checking
    st.session_state.messages.append({"role": "assistant", "content": str(full_response)})