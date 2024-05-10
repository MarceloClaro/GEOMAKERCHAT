import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from crewai_tools import DuckDuckGoSearchRun

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
            """
            Como assistente de pesquisa dedicado a descobrir as tendências mais impactantes, você é movido por uma curiosidade implacável e um compromisso com a inovação. Sua função envolve aprofundar-se nos desenvolvimentos mais recentes em vários setores para identificar e analisar as principais notícias de tendência em qualquer campo. Essa busca não apenas satisfaz sua sede de conhecimento, mas também permite que você contribua com insights valiosos que podem potencialmente remodelar entendimentos e expectativas em escala global.
            """
        ),
        tools=[search_tool],
        allow_delegation=True,
        llm=ChatGroq(groq_api_key=groq_api_key, model_name=model_choice)
    )

    blog_writer = Agent(
        role="Escritor Especialista",
        goal="Escrever conteúdos envolventes sobre {topic}",
        verbose=True,
        memory=True,
        backstory=(
            """
            Armado com o talento para destilar assuntos complexos em histórias digeríveis e envolventes, você, como escritor de blog, tece habilmente narrativas que tanto iluminam quanto envolvem seu público. Sua escrita ilumina novas percepções e descobertas, tornando-as acessíveis a todos. Através de sua arte, você traz à tona a essência de novos desenvolvimentos em diversos tópicos, tornando o intricado mundo das notícias uma jornada fascinante para seus leitores.
            """
        ),
        tools=[search_tool],
        allow_delegation=False,
        llm=ChatGroq(groq_api_key=groq_api_key, model_name=model_choice)
    )

    research_task = Task(
        description=(
            """
            Identificar a próxima grande tendência em {topic}.
            Concentre-se em identificar prós e contras e a narrativa geral.
            Seu relatório final deve articular claramente os principais pontos, suas oportunidades de mercado e riscos potenciais.
            """
        ),
        expected_output="Um relatório abrangente de 3 parágrafos sobre o {topic}",
        tools=[search_tool],
        agent=researcher
    )

    write_task = Task(
        description=(
            """
            Compor um artigo perspicaz sobre {topic}.
            Concentre-se nas últimas tendências e como elas estão impactando a indústria.
            Este artigo deve ser de fácil compreensão, envolvente e positivo.
            """
        ),
        expected_output="Um artigo de 4 parágrafos sobre os avanços {topic} formatado como markdown traduzido em português.",
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
