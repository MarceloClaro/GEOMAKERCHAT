import os
import streamlit as st
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, YoutubeVideoSearchTool
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Set API keys
os.environ["SERPER_API_KEY"] = "4fe66ce9a539c645b83c11188f3410f2e82d8c18"


# Create Langchain Groq API object
groq_api_key = GROQ_API_KEY
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

# Create Streamlit interface
st.title("Pesquisa de Vídeos do YouTube")

entrada = st.text_input("Pesquise vídeos do YouTube:")

if st.button("Pesquisar"):
    # Create research agent and task
    research_agent = Agent(
        name="research_agent",
        role='Pesquisador de Vídeos do YouTube',
        goal='Encontrar os 10 vídeos mais relevantes sobre ' + entrada,
        backstory="""Você é um pesquisador de vídeo experiente.
            Você é responsável por encontrar os vídeos mais relevantes no YouTube.""",
        verbose=True,
        tools=[SerperDevTool(), YoutubeVideoSearchTool()]
    )

    research_task = Task(
        description='Encontre os 10 vídeos mais relevantes no YouTube sobre ' + entrada,
        expected_output='Uma lista com os 10 vídeos no seguinte modelo: Título, contagem de visualizações, link do vídeo',
        agent="research_agent",
    )

    # Create crew and execute task
    crew = Crew(agents=[research_agent], tasks=[research_task], verbose=2)
    result = crew.kickoff()

    # Display results
    st.write("Top 10 vídeos mais relevantes do YouTube:")
    for video in result:
        st.write(f"**{video['title']}** - {video['view_count']} views - [Assistir no YouTube]({video['link']})")
