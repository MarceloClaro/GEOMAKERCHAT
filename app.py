import streamlit as st
import os
import pandas as pd
from crewai import Agent, Task, Crew
from crewai_tools import PDFSearchTool
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def upload_data():
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos (até 2 arquivos, 200MB cada)", type=['json', 'xlsx', 'csv', 'pdf'], accept_multiple_files=True, key="data_upload")
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
                elif file.type == 'application/pdf':
                    # Initialize PDFSearchTool with the uploaded PDF file
                    pdf_search_tool = PDFSearchTool(pdf=file)
                    # Use the tool to search within the PDF
                    data = pdf_search_tool.search("your_query_here")
                else:
                    raise ValueError("Tipo de arquivo não suportado")
                data_frames.append(data)
                st.write(f"Pré-visualização do arquivo {file.type.split('/')[-1]} carregado:")
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

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=current_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        # Define your CrewAI agents with roles, goals, and tools
        researcher = Agent(
            role='Senior Research Analyst',
            goal='Uncover cutting-edge developments in AI and data science',
            backstory='I am a Senior Research Analyst at a leading tech think tank. My expertise lies in identifying emerging trends and technologies in AI and data science. I have a knack for dissecting complex data and presenting actionable insights.',
            verbose=True,
            allow_delegation=False,
            tools=[groq_chat],
            max_rpm=100
        )

        writer = Agent(
            role='Tech Content Strategist',
            goal='Craft compelling content on tech advancements',
            backstory='I am a renowned Tech Content Strategist, known for my insightful and engaging articles on technology and innovation. With a deep understanding of the tech industry, I transform complex concepts into compelling narratives.',
            verbose=True,
            allow_delegation=True,
            tools=[groq_chat],
            cache=False, # Disable cache for this agent
            max_rpm=100
        )

        data_scientist = Agent(
            role='Data Scientist',
            goal='Analyze data and provide insights',
            backstory='I am a Data Scientist with expertise in analyzing complex data sets and extracting valuable insights. My goal is to help you make informed decisions based on data-driven analysis.',
            verbose=True,
            allow_delegation=True,
            tools=[groq_chat],
            cache=False, # Disable cache for this agent
            max_rpm=100
        )

        task = Task(
            description='Answer user inquiries',
            expected_output='A response to the user inquiry',
            agent=researcher,
            human_input=True,
        )

        data_analysis_task = Task(
            description='Analyze uploaded data',
            expected_output='Insights from the data analysis',
            agent=data_scientist,
            human_input=False,  # No human input required for this task
        )

        crew = Crew(
            agents=[researcher, writer, data_scientist],
            tasks=[task, data_analysis_task],
            verbose=2
        )

        response = crew.kickoff(human_input=user_question)
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
