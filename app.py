from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
import groq  # Adicione esta linha
import toml
import time


# Carregar a chave de API do Groq do arquivo secrets.toml
secrets = toml.load("secrets.toml")
groq_api_key = secrets["GROQ_API_KEY"]

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG+CreWAI")
    st.image("Untitled.png", width=100)
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAGRAG+CreWAI!")
    st.write("""Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.
    Com 1 Agente: 
    "Pesquisador Acadêmico": Encontre informações confiáveis e atuais sobre {topic} seguindo as normas científicas e da ABNT.
    "Como pesquisador acadêmico, seu objetivo é contribuir para o avanço do conhecimento científico em sua área. Você segue rigorosamente as normas e metodologias científicas e da ABNT para garantir a qualidade e confiabilidade de suas pesquisas. Sua busca por informações é guiada pela busca da verdade e pela contribuição para a comunidade acadêmica."
    "Pesquise e compile informações relevantes e atualizadas sobre seguindo as normas científicas e da formatação ABNT. Certifique-se de incluir referências bibliográficas adequadas."
    "Um resumo detalhado e bem estruturado sobre {topic} seguindo as normas científicas e da ABNT.",""")

    st.sidebar.title('Customização')
    primary_prompt = st.sidebar.text_input("Prompt do sistema principal", "Como posso ajudar você hoje?")
    secondary_prompt = st.sidebar.text_input("Prompt do sistema secundário", "Há algo mais em que posso ajudar?")
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    conversational_memory_length = st.sidebar.slider('Tamanho da memória conversacional', 1, 50, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    search_tool = DuckDuckGoSearchRun()
    academic_researcher = Agent(
        role="Pesquisador Acadêmico",
        goal="Encontre informações confiáveis e atuais sobre {topic} seguindo as normas científicas e da ABNT",
        verbose=True,
        memory=True,
        backstory=(
            "Como pesquisador acadêmico, seu objetivo é contribuir para o avanço do conhecimento científico em sua área. Você segue rigorosamente as normas e metodologias científicas e da ABNT para garantir a qualidade e confiabilidade de suas pesquisas. Sua busca por informações é guiada pela busca da verdade e pela contribuição para a comunidade acadêmica."
        ),
        tools=[search_tool],
        allow_delegation=True,
        llm=ChatGroq(api_key=groq_api_key, model_name=model_choice)
    )

    research_task = Task(
        description=(
            "Pesquise e compile informações relevantes e atualizadas sobre {topic} seguindo as normas científicas e da formatação ABNT. Certifique-se de incluir referências bibliográficas adequadas."
        ),
        expected_output="Um resumo detalhado e bem estruturado sobre {topic} seguindo as normas científicas e da ABNT.",
        tools=[search_tool],
        agent=academic_researcher
    )

    crew = Crew(
        agents=[academic_researcher],
        tasks=[research_task],
        process=Process.sequential
    )

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        current_prompt = secondary_prompt if 'last_prompt' in st.session_state and st.session_state.last_prompt == primary_prompt else primary_prompt
        st.session_state.last_prompt = current_prompt

        prompt = f"{current_prompt} {user_question}"
        while True:
            try:
                result = crew.kickoff(inputs={"topic": user_question})
                st.write("Chatbot:", result)
                break
            except groq.RateLimitError as e:
                st.warning(f"Rate limit exceeded. Waiting for {e.wait_time} seconds before trying again...")
                time.sleep(e.wait_time)

if __name__ == "__main__":
    main()

