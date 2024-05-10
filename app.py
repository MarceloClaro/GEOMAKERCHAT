import streamlit as st 
import os
import pandas as pd
from crewai import Agent, Task, Crew
from crewai_tools import PDFSearchTool
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import toml

# Função para upload de dados
def upload_data(uploaded_files):
    data_frames = []
    for file in uploaded_files:
        try:
            if file.type == 'application/json':
                data = pd.read_json(file)
            elif file.type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
                data = pd.read_excel(file)
            elif file.type == 'text/csv':
                data = pd.read_csv(file)
            elif file.type == 'application/pdf':
                pdf_search_tool = PDFSearchTool(pdf=file)
                data = pdf_search_tool.search("sua consulta aqui")
            else:
                raise ValueError("Tipo de arquivo não suportado")
            data_frames.append(data)
            st.write(f"Pré-visualização do arquivo {file.type.split('/')[-1]} carregado:")
            st.dataframe(data.head())
        except ValueError as e:
            st.error(f"Erro ao ler o arquivo {file.name}: {e}")
    return data_frames

# Função principal
def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG")
    st.image("Untitled.png", width=100)
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.")

    # Configurações da barra lateral
    st.sidebar.title('Customização')
    primary_prompt = st.sidebar.text_input("Prompt do sistema principal", "Como posso ajudar você hoje?")
    secondary_prompt = st.sidebar.text_input("Prompt do sistema secundário", "Há algo mais em que posso ajudar?")
    conversational_memory_length = st.sidebar.slider('Tamanho da memória conversacional', 1, 50, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Carregar a chave GROQ_API_KEY do arquivo secrets.toml
    secrets = toml.load("secrets.toml")
    GROQ_API_KEY = secrets["GROQ_API_KEY"]

    llama3_model = LLMChain(model_name="llama3-70b-8192", llm="texto", prompt="sua consulta aqui", api_key=GROQ_API_KEY)

    uploaded_files = st.file_uploader("Faça upload dos seus arquivos (até 2 arquivos, 200MB cada)", type=['json', 'xlsx', 'csv', 'pdf'], accept_multiple_files=True, key="data_upload")
    if uploaded_files:
        data_frames = upload_data(uploaded_files)
        st.session_state['uploaded_data'] = data_frames

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        current_prompt = secondary_prompt if 'last_prompt' in st.session_state and st.session_state.last_prompt == primary_prompt else primary_prompt
        st.session_state.last_prompt = current_prompt

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=current_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        researcher = Agent(
            role='Senior Research Analyst',
            goal='Descobrir desenvolvimentos de ponta em IA e ciência de dados',
            backstory='Eu sou um Analista de Pesquisa Sênior em um think tank de tecnologia líder. Minha expertise está em identificar tendências emergentes e tecnologias inovadoras em IA e ciência de dados. Eu tenho habilidade em dissecar dados complexos e apresentar insights acionáveis.',
            allow_delegation=False,
            tools=[llama3_model],
            max_rpm=100
        )

        writer = Agent(
            role='Tech Content Strategist',
            goal='Criar conteúdo envolvente sobre avanços tecnológicos',
            backstory='Eu sou um renomado Estrategista de Conteúdo de Tecnologia, conhecido por meus artigos perspicazes e envolventes sobre tecnologia e inovação. Com um profundo entendimento da indústria de tecnologia, eu transformo conceitos complexos em narrativas cativantes.',
            allow_delegation=True,
            tools=[llama3_model],
            cache=False,
            max_rpm=100
        )

        data_scientist = Agent(
            role='Data Scientist',
            goal='Analisar dados e fornecer insights',
            backstory='Eu sou um Cientista de Dados com expertise em analisar conjuntos de dados complexos e extrair insights valiosos. Meu objetivo é ajudá-lo a tomar decisões informadas com base em análises orientadas por dados.',
            allow_delegation=False,
            tools=[],
            cache=False,
            max_rpm=100
        )

        data_analysis_task = Task(
            description='Analisar dados para fornecer insights sobre tendências de pesquisa em IA e ciência de dados.',
            agents=[data_scientist],
            human_input=False
        )

        task1 = Task(
            description='Pesquisar e fornecer informações sobre os últimos avanços em IA e ciência de dados.',
            agents=[researcher],
            human_input=True,
        )

        task2 = Task(
            description='Escrever um artigo sobre as aplicações práticas da IA na ciência de dados.',
            agents=[writer],
            human_input=True,
        )

        crew = Crew(
            agents=[researcher, writer, data_scientist],
            tasks=[task1, task2, data_analysis_task],
            verbose=2
        )

        result = crew.kickoff()
        response = result
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
