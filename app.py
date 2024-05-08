import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, ChatMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def upload_and_visualize_data():
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos (JSON, CSV, XLSX até 200MB cada)", 
                                      type=['json', 'csv', 'xlsx'], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            try:
                if file.type == "application/json":
                    data = pd.read_json(file)
                elif file.type == "text/csv":
                    data = pd.read_csv(file)
                else:
                    data = pd.read_excel(file)
                st.session_state.setdefault('data_frames', []).append(data)
                st.write(f"Pré-visualização do arquivo {file.name} carregado:")
                st.dataframe(data.head())
                visualize_data(data)
            except Exception as e:
                st.error(f"Erro ao processar o arquivo {file.name}: {e}")

def visualize_data(data):
    if not data.empty:
        st.write("Visualização gráfica dos dados:")
        sns.pairplot(data.select_dtypes(include=[np.number]).dropna())
        st.pyplot()

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Advanced Chat Interface with RAG")
    st.image("Untitled.png", width=100)
    st.title("Bem-vindo ao Chat Avançado com RAG!")

    groq_api_key = os.getenv('GROQ_API_KEY', 'Chave_API_Padrão')
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")

    upload_and_visualize_data()

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        chat_messages = [
            SystemMessage(content="Como posso ajudar você hoje?"),
            *[ChatMessage(content=msg) for msg in st.session_state.get('chat_history', [])],
            HumanMessagePromptTemplate(template="{human_input}")
        ]

        prompt = ChatPromptTemplate(messages=chat_messages)
        groq_chat = ChatGroq(api_key=groq_api_key, model_name=model_choice)
        conversation = LLMChain(llm=groq_chat, prompt=prompt, memory=memory)
        response = conversation.predict(human_input=user_question)
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
