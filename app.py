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
    """
    Permite aos usuários fazer upload de arquivos JSON, CSV e XLSX que podem ser usados como fonte de dados.
    Os arquivos são carregados através de um widget de upload no Streamlit, lidos como DataFrames e visualizados.
    """
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos (JSON, CSV, XLSX até 200MB cada)", 
                                      type=['json', 'csv', 'xlsx'], accept_multiple_files=True, key="file_upload")
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
    """
    Gera visualizações automáticas para os DataFrames carregados, utilizando Matplotlib e Seaborn.
    """
    if not data.empty:
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            st.write("Visualização gráfica dos dados:")
            plt.figure(figsize=(10, 6))
            sns.pairplot(data[numeric_columns].dropna())
            st.pyplot(plt)
        if 'date' in data.columns or 'Date' in data.columns:
            date_col = 'date' if 'date' in data.columns else 'Date'
            data[date_col] = pd.to_datetime(data[date_col])
            plt.figure(figsize=(10, 6))
            plt.plot(data[date_col], data[numeric_columns[0]].dropna(), marker='o')
            plt.title('Time Series Plot')
            plt.xlabel('Date')
            plt.ylabel('Values')
            st.pyplot(plt)

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG")
    st.image("Untitled.png", width=100)
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.")

    groq_api_key = os.getenv('GROQ_API_KEY', 'Chave_API_Padrão')
    st.sidebar.title('Customização')
    primary_prompt = st.sidebar.text_input("Prompt do sistema principal", "Como posso ajudar você hoje?")
    secondary_prompt = st.sidebar.text_input("Prompt do sistema secundário", "Há algo mais em que posso ajudar?")
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    upload_and_visualize_data()

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        current_prompt = secondary_prompt if 'last_prompt' in st.session_state and st.session_state.last_prompt == primary_prompt else primary_prompt
        st.session_state.last_prompt = current_prompt

        # Preparing chat messages properly
        chat_messages = [SystemMessage(content=current_prompt)]
        chat_messages.extend([ChatMessage(content=msg) for msg in st.session_state.get('chat_history', [])])
        chat_messages.append(HumanMessagePromptTemplate(template="{human_input}"))

        prompt = ChatPromptTemplate(messages=chat_messages)

        groq_chat = ChatGroq(api_key=groq_api_key, model_name=model_choice)
        conversation = LLMChain(llm=groq_chat, prompt=prompt, memory=memory)
        response = conversation.predict(human_input=user_question)
        st.session_state.chat_history.append(user_question)  # Track user questions
        st.session_state.chat_history.append(response)  # Save response
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
