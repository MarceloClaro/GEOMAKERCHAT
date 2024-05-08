import streamlit as st
import os
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, ChatMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq  # Verificar importação

def upload_json_data():
    """Carrega dados JSON e os transforma em DataFrame para uso posterior."""
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos JSON (até 2 arquivos, 200MB cada)", type='json', accept_multiple_files=True)
    data_frames = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                data = pd.read_json(file)
                data_frames.append(data)
                st.write("Pré-visualização do arquivo JSON carregado:")
                st.dataframe(data.head())
            except ValueError as e:
                st.error(f"Erro ao ler o arquivo {file.name}: {e}")
    return data_frames

def format_chat_history(history):
    """Formata o histórico de chat para o formato esperado pela LangChain."""
    return [ChatMessage(role='user', content=msg) for msg in history]

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG")
    st.image("Untitled.png", width=100)
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.")

    groq_api_key = os.getenv('GROQ_API_KEY', 'Chave_API_Padrão')
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")
    data_frames = upload_json_data()

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append(user_question)

        formatted_history = format_chat_history(st.session_state.chat_history)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Como posso ajudar você hoje?"),
            MessagesPlaceholder(variable_name="formatted_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        conversation = LLMChain(llm=ChatGroq(api_key=groq_api_key, model_name=model_choice), prompt=prompt, memory=memory)
        response = conversation.predict(human_input=user_question, formatted_history=formatted_history)
        message = {'role': 'user', 'content': user_question}
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
