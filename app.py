import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, YoutubeVideoSearchTool
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Carregue as variáveis de ambiente
load_dotenv()

# Defina as chaves de API
os.environ["SERPER_API_KEY"] = "4fe66ce9a539c645b83c11188f3410f2e82d8c18"
GROQ_API_KEY = "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"

# Crie o objeto da API Langchain Groq
groq_api_key = GROQ_API_KEY
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

# Crie a interface Streamlit
st.title("Pesquisador de Vídeos do YouTube")

# Adicione um campo de entrada de texto
entrada = st.text_input("Pesquisar vídeos do YouTube:", placeholder="Digite sua pesquisa")

# Adicione um botão de pesquisa
if st.button("Pesquisar"):
    # Crie o agente de pesquisa e a tarefa
    research_agent = Agent(
        role='Pesquisador de Vídeos do YouTube',
        goal='Encontrar os 10 vídeos mais relevantes sobre ' + entrada,
        backstory="""Você é um pesquisador de vídeos do YouTube. Você é responsável por encontrar os vídeos mais relevantes sobre um determinado tópico.""",
        verbose=True
    )

    research_task = Task(
        description='Encontre os 10 vídeos mais relevantes no YouTube sobre ' + entrada,
        expected_output='Uma lista com os 10 vídeos no seguinte modelo: Título, contagem de visualizações, link do vídeo',
        agent=research_agent,
        tools=[SerperDevTool(), YoutubeVideoSearchTool()],
    )

    # Crie a equipe e execute a tarefa
    crew = Crew(agents=[research_agent], tasks=[research_task], verbose=2)
    result = crew.kickoff()

    # Exiba os resultados
    st.write("Top 10 vídeos mais relevantes do YouTube:")
    for video in result:
        st.write(f"**{video['title']}** - {video['view_count']} views - [Assistir no YouTube]({video['link']})")
