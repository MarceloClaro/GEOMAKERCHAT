import streamlit as st
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, ChatMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def upload_json_data():
    """ Permite o upload de arquivos JSON e os armazena como DataFrame no estado da sessão. """
    uploaded_files = st.file_uploader("Upload JSON files", type='json', accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            try:
                data = pd.read_json(file)
                st.session_state.setdefault('uploaded_data', []).append(data)
                st.write("Preview of the uploaded JSON file:")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Failed to read the file {file.name}: {e}")

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Advanced Chat Interface with RAG")
    st.title("Welcome to the Advanced RAG Chat Maker!")
    st.write("This chatbot leverages advanced language generation and retrieval techniques.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    upload_json_data()

    groq_api_key = st.secrets.get('GROQ_API_KEY', 'Default_API_Key')
    model_choice = st.sidebar.selectbox("Choose a model", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")

    user_question = st.text_input("Ask a question:")
    if user_question:
        # Preparing chat history for the prompt
        chat_messages = [ChatMessage(role='user', content=msg) for msg in st.session_state.chat_history]
        chat_messages.append(ChatMessage(role='user', content=user_question))

        prompt = ChatPromptTemplate(messages=[
            SystemMessage(content="How can I assist you today?"),
            MessagesPlaceholder(messages=chat_messages),
            HumanMessagePromptTemplate(template="{human_input}")
        ])

        groq_chat = ChatGroq(api_key=groq_api_key, model_name=model_choice)
        conversation = LLMChain(llm=groq_chat, prompt=prompt, memory=memory)
        response = conversation.predict(human_input=user_question)
        
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response)
        
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
