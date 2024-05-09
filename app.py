import streamlit as st
import os
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def upload_data():
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos (até 2 arquivos, 200MB cada)", type=['json', 'xlsx', 'csv'], accept_multiple_files=True, key="data_upload")
    if uploaded_files:
        data_frames = []
        for file in uploaded_files:
            try:
                if file.type == 'application/json':
                    data = pd.read_json(file)
                elif file.type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
                    data = pd.read_excel(file)
                elif file.type == 'text/csv':
                    data = pd.read_csv(file)
                else:
                    raise ValueError("Tipo de arquivo não suportado")
                data_frames.append(data)
                st.write(f"Pré-visualização do arquivo {file.type.split('/')[-1]} carregado:")
                st.dataframe(data.head())
            except ValueError as e:
                st.error(f"Erro ao ler o arquivo {file.name}: {e}")
        st.session_state['uploaded_data'] = data_frames

def analyze_data(question, data_frames):
    # Implemente a lógica para analisar os dados carregados com base na pergunta
    # Esta função deve retornar os resultados da análise que serão usados para gerar a resposta
    # Exemplo: análise de tendências, cálculos estatísticos, etc.
    # Para fins de exemplo, vamos apenas retornar uma mensagem indicando que a análise foi realizada
    return f"Analisando dados para a pergunta '{question}'..."

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
    conversational_memory_length = st.sidebar.slider('Tamanho da memória conversacional', 1, 50, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    groq_chat = ChatGroq(api_key=groq_api_key, model_name=model_choice)
    upload_data()

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        current_prompt = secondary_prompt if 'last_prompt' in st.session_state and st.session_state.last_prompt == primary_prompt else primary_prompt
        st.session_state.last_prompt = current_prompt

        # Verifica se a pergunta é complexa e requer análise profunda dos dados
        if is_complex_question(user_question):  # Função hipotética para determinar se a pergunta é complexa
            analysis_result = analyze_data(user_question, st.session_state['uploaded_data'])
            response = f"{analysis_result} Por favor, tente novamente com uma pergunta mais específica."
        else:
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
