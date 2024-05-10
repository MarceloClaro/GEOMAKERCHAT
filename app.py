import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import toml
import time  # Para adicionar um pequeno atraso entre as solicitações

# Carregar a chave de API do Groq do arquivo secrets.toml
secrets = toml.load("secrets.toml")
groq_api_key = secrets["GROQ_API_KEY"]

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG")
    st.image("Untitled.png", width=100)
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.")

    st.sidebar.title('Customização')
    primary_prompt = st.sidebar.text_input("Prompt do sistema principal", "Como posso ajudar você hoje?")
    secondary_prompt = st.sidebar.text_input("Prompt do sistema secundário", "Há algo mais em que posso ajudar?")
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    conversational_memory_length = st.sidebar.slider('Tamanho da memória conversacional', 1, 50, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    search_tool = DuckDuckGoSearchRun()
    researcher = Agent(
        role="Pesquisador Sênior",
        goal="Descubra as três principais notícias de renderização em {topic}",
        verbose=True,
        memory=True,
        backstory=(
            "Como assistente de pesquisa dedicado a descobrir as tendências mais impactantes, você é movido por uma curiosidade implacável e um compromisso com a inovação. Sua função envolve aprofundar-se nos desenvolvimentos mais recentes em vários setores para identificar e analisar as principais notícias de tendência em qualquer campo. Essa busca não apenas satisfaz sua sede de conhecimento, mas também permite que você contribua com insights valiosos que podem potencialmente remodelar entendimentos e expectativas em escala global."
        ),
        tools=[search_tool],
        allow_delegation=True,
        llm=ChatGroq(api_key=groq_api_key, model_name=model_choice)
    )

    blog_writer = Agent(
        role="Escritor Especialista",
        goal="Escreva conteúdos envolventes sobre {topic}",
        verbose=True,
        memory=True,
        backstory=(
            "Armed with the knack for distilling complex subjects into digestible, compelling stories, you, as a blog writer, masterfully weave narratives that both enlighten and engage your audience. Your writing illuminates fresh insights and discoveries, making them approachable for everyone. Through your craft, you bring to the forefront the essence of new developments across various topics, making the intricate world of news a fascinating journey for your readers."
        ),
        tools=[search_tool],
        allow_delegation=False,
        llm=ChatGroq(api_key=groq_api_key, model_name=model_choice)
    )

    research_task = Task(
        description=(
            "Identify the next big trend in {topic}. Focus on identifying pros and cons and the overall narrative. Your final report should clearly articulate the key points its market opportunities, and potential risks."
        ),
        expected_output="A comprehensive 3 paragraphs long report on the {topic}",
        tools=[search_tool],
        agent=researcher
    )

    write_task = Task(
        description=(
            "Compose an insightful article on {topic}. Focus on the latest trends and how it's impacting the industry. This article should be easy to understand, engaging, and positive."
        ),
        expected_output="A 4 paragraph article on {topic} advancements formatted as markdown traduzido em portugues.",
        tools=[search_tool],
        agent=blog_writer,
        aync_execution=False,
        output_file="blog-post.md"
    )

    crew = Crew(
        agents=[researcher, blog_writer],
        tasks=[research_task, write_task],
        process=Process.sequential
    )

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        current_prompt = secondary_prompt if 'last_prompt' in st.session_state and st.session_state.last_prompt == primary_prompt else primary_prompt
        st.session_state.last_prompt = current_prompt

        prompt = f"{current_prompt} {user_question}"
        result = crew.kickoff(inputs={"topic": user_question})
        st.write("Chatbot:", result)

if __name__ == "__main__":
    main()
