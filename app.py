import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.models.rag import RAG
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import pandas as pd
import toml

# Carregar a chave de API do Groq do arquivo secrets.toml
secrets = toml.load("secrets.toml")
groq_api_key = secrets["GROQ_API_KEY"]

def get_tokens_per_minute(model_name):
    # Defina os limites de tokens por minuto para cada modelo
    model_limits = {
        "rag-7b-it": 3000,  # Ajuste conforme necessário
    }
    return model_limits.get(model_name, 3000)  # Padrão para 3000 se o modelo não estiver na lista

def upload_json_data():
    """
    Permite aos usuários fazer upload de arquivos JSON que podem ser usados como fonte de dados.
    Os arquivos são carregados através de um widget de upload no Streamlit e lidos como DataFrames.
    """
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos JSON (até 2 arquivos, 200MB cada)", type='json', accept_multiple_files=True, key="json_upload")
    if uploaded_files:
        data_frames = []
        for file in uploaded_files:
            try:
                # Lê o arquivo JSON e tenta convertê-lo em DataFrame
                data = pd.read_json(file)
                data_frames.append(data)
                st.write(f"Pré-visualização do arquivo JSON carregado:")
                st.dataframe(data.head())
            except ValueError as e:
                # Caso ocorra um erro na leitura do JSON, mostra uma mensagem de erro
                st.error(f"Erro ao ler o arquivo {file.name}: {e}")
        st.session_state['uploaded_data'] = data_frames

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG")
    st.image("Untitled.png", width=100)
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo RAG em português para responder às suas perguntas.")

    groq_api_key = os.getenv('GROQ_API_KEY', 'Chave_API_Padrão')

    st.sidebar.title('Customização')
    primary_prompt = st.sidebar.text_input("Prompt do sistema principal", "Como posso ajudar você hoje?")
    secondary_prompt = st.sidebar.text_input("Prompt do sistema secundário", "Há algo mais em que posso ajudar?")
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["rag-7b-it"])
    conversational_memory_length = st.sidebar.slider('Tamanho da memória conversacional', 1, 50, value=5)

    tokens_per_minute = get_tokens_per_minute(model_choice)
    st.sidebar.slider('Tokens por minuto', 100, 5000, value=tokens_per_minute)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    rag_chat = RAG(api_key=groq_api_key, model_name=model_choice, tokens_per_minute=tokens_per_minute)

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        current_prompt = secondary_prompt if 'last_prompt' in st.session_state and st.session_state.last_prompt == primary_prompt else primary_prompt
        st.session_state.last_prompt = current_prompt

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=current_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        response = rag_chat.predict(human_input=user_question, prompt=prompt)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
