import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Corrected import paths assuming these are the correct ones.
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, ChatMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def upload_and_process_files():
    uploaded_files = st.file_uploader("Upload your files (JSON, CSV, XLSX up to 200MB each)", 
                                      type=['json', 'csv', 'xlsx'], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            try:
                if file.type == "application/json":
                    data = pd.read_json(file)
                elif file.type == "text/csv":
                    data = pd.read_csv(file)
                elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    data = pd.read_excel(file)
                
                st.session_state.setdefault('data_frames', []).append(data)
                st.write(f"Preview of uploaded file {file.name}:")
                st.dataframe(data.head())
                visualize_data(data)
            except Exception as e:
                st.error(f"Error processing file {file.name}: {e}")

def visualize_data(data):
    if not data.empty:
        st.write("Graphical visualization of data:")
        sns.pairplot(data.select_dtypes(include=[np.number]).dropna())
        plt.show()

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Advanced Chat Interface with RAG")
    st.image("Untitled.png", width=100)
    st.title("Welcome to the Advanced Chat with RAG!")

    groq_api_key = os.getenv('GROQ_API_KEY', 'Default_API_Key')
    model_choice = st.sidebar.selectbox("Choose a model", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")

    upload_and_process_files()

    user_question = st.text_input("Ask a question:")
    if user_question:
        chat_messages = [
            SystemMessage(content="How can I assist you today?"),
            *[ChatMessage(content=msg) for msg in st.session_state.get('chat_history', [])],
            HumanMessagePromptTemplate(template="{human_input}")
        ]

        prompt = ChatPromptTemplate(messages=chat_messages)
        groq_chat = ChatGroq(api_key=groq_api_key, model_name=model_choice)
        conversation = LLMChain(llm=groq_chat, prompt=prompt, memory=memory)
        response = conversation.predict(human_input=user_question)
        st.session_state['chat_history'].append(user_question)
        st.session_state['chat_history'].append(response)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
