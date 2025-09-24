import streamlit as st
from openai import OpenAI

# Load API key securely from Streamlit secrets
client = OpenAI(api_key=st.secrets["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"])

st.set_page_config(page_title="Health Chatbot", page_icon="🩺", layout="centered")

st.title("🩺 Health Chatbot")
st.write("Ask me anything about general health, fitness, diet, and well-being. (I am not a doctor, just an AI assistant!)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful health assistant. Provide safe, general advice only."}
    ]

# Display past messages
for msg in st.session_state.messages[1:]:  # skip system
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your health question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # lightweight & cheaper
        messages=st.session_state.messages
    )

    reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply)
