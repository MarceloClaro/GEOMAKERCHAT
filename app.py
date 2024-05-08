import streamlit as st
import os
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq  # Verifique se esta importação está correta

def upload_json_data():
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos JSON (até 2 arquivos, 200MB cada)", type='json', accept_multiple_files=True)
    data_frames = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                data = pd.read_json(file)
                data_frames.append(data)
                st.write(f"Pré-visualização do arquivo JSON carregado:")
                st.dataframe(data.head())
            except ValueError as e:
                st.error(f"Erro ao ler o arquivo {file.name}: {e}")
        st.session_state['uploaded_data'] = data_frames

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
    data_frames = upload_json_data()

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({'role': 'user', 'content': user_question})

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=primary_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        conversation = LLMChain(llm=ChatGroq(api_key=groq_api_key, model_name=model_choice), prompt=prompt, memory=memory)
        response = conversation.predict(human_input=user_question)
        message = {'role': 'user', 'content': user_question}
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
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
