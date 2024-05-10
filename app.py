import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from crewai_tools import DuckDuckGoSearchRun

import toml
import time

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
    groq_chat = ChatGroq(api_key=groq_api_key, model_name=model_choice)

    researcher = Agent(
        role="Pesquisador Sênior",
        goal="Descubra as três principais notícias sobre {topic}",
        verbose=True,
        memory=True,
        backstory=(
            "Como assistente de pesquisa dedicado a descobrir as tendências mais impactantes, você é movido por uma curiosidade implacável e um compromisso com a inovação. Sua função envolve aprofundar-se nos desenvolvimentos mais recentes em vários setores para identificar e analisar as principais notícias de tendência em qualquer campo. Essa busca não apenas satisfaz sua sede de conhecimento, mas também permite que você contribua com insights valiosos que podem potencialmente remodelar entendimentos e expectativas em escala global."
        ),
        tools=[search_tool],
        allow_delegation=True,
        llm=groq_chat
    )

    write_task = Task(
        description=(
            "Componha um artigo informativo sobre {topic}."
            "Concentre-se nas últimas tendências e em como elas estão impactando a indústria."
            "Este artigo deve ser fácil de entender, envolvente e positivo."
        ),
        expected_output="Um artigo de 4 parágrafos sobre os avanços de {topic} formatado como markdown traduzido para o português.",
        tools=[search_tool],
        agent=researcher,
        async_execution=False,
        output_file="blog-post.md"
    )

    crew = Crew(
        agents=[researcher],
        tasks=[write_task],
        process=Process.sequential
    )

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        current_prompt = secondary_prompt if 'last_prompt' in st.session_state and st.session_state.last_prompt == primary_prompt else primary_prompt
        st.session_state.last_prompt = current_prompt

        prompt = f"{current_prompt} {user_question}"
        conversation = ChatGroq(api_key=groq_api_key, model_name=model_choice)
        response = conversation.predict(prompt)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
