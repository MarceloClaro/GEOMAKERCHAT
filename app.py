import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import toml
import time  # Para adicionar um pequeno atraso entre as solicitações

# Carregar a chave de API do Groq do arquivo secrets.toml
secrets = toml.load("secrets.toml")
groq_api_key = secrets["GROQ_API_KEY"]

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG")
    st.image("Untitled.png", width=100)
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.")

    st.sidebar.title('Customização')
    primary_prompt = st.sidebar.text_input("Prompt do sistema principal", "Como posso ajudar você hoje?")
    secondary_prompt = st.sidebar.text_input("Prompt do sistema secundário", "Há algo mais em que posso ajudar?")
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    conversational_memory_length = st.sidebar.slider('Tamanho da memória conversacional', 1, 50, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    model_kwargs = {}
    groq_chat = ChatGroq(api_key=groq_api_key, model_name=model_choice, **model_kwargs)

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        prompt = f"{primary_prompt} {user_question} {secondary_prompt}"
        conversation = ChatGroq(api_key=groq_api_key, model_name=model_choice, **model_kwargs)
        response = conversation.predict(prompt)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
