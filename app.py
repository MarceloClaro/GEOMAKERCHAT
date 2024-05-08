import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def upload_and_visualize_data():
    """
    Permite aos usuários fazer upload de arquivos JSON, CSV ou XLSX e os visualiza automaticamente.
    """
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos (JSON, CSV, XLSX até 300MB cada)", type=['json', 'csv', 'xlsx'], accept_multiple_files=True)
    data_frames = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                if file.type == "application/json":
                    data = pd.read_json(file)
                elif file.type == "application/vnd.ms-excel":
                    data = pd.read_excel(file)
                else:
                    data = pd.read_csv(file)
                data_frames.append(data)
                st.write(f"Pré-visualização do arquivo {file.name} carregado:")
                st.dataframe(data.head())
                visualize_data(data)
            except Exception as e:
                st.error(f"Erro ao processar o arquivo {file.name}: {e}")

def visualize_data(data):
    """
    Gera visualizações de dados para o DataFrame fornecido.
    """
    if not data.empty:
        fig, ax = plt.subplots()
        if len(data.columns) > 1 and is_numeric_dtype(data[data.columns[1]]):
            sns.barplot(data=data, x=data.columns[0], y=data.columns[1], ax=ax)
            st.pyplot(fig)
        else:
            sns.countplot(x=data[data.columns[0]], ax=ax)
            st.pyplot(fig)

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG")
    st.image("Untitled.png", width=100)
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.")

    groq_api_key = os.getenv('GROQ_API_KEY', 'Chave_API_Padrão')
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    conversational_memory_length = st.sidebar.slider('Tamanho da memória conversacional', 1, 50, value=5)
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    groq_chat = ChatGroq(api_key=groq_api_key, model_name=model_choice)
    upload_and_visualize_data()

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        current_prompt = secondary_prompt if 'last_prompt' in st.session_state and st.session_state.last_prompt == primary_prompt else primary_prompt
        st.session_state.last_prompt = current_prompt

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=current_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        conversation = LLMChain(llm=groq_chat, prompt=prompt, memory=memory)
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)
        st.image("eu.ico", width=100)
        st.write("""
        Projeto Geomaker + IA 
        - Professor: Marcelo Claro.
        Contatos: marceloclaro@gmail.com
        Whatsapp: (88)981587145
        Instagram: https://www.instagram.com/marceloclaro.geomaker/
        """)    

if __name__ == "__main__":
    main()
