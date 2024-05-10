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
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAG+CreWAI!")
    st.write("""Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.
    
    Com 3 Agentes: 
    
    Pesquisador Acadêmico": Encontre informações confiáveis e atuais sobre tópico, seguindo as normas científicas e da ABNT.
    Como pesquisador acadêmico, seu objetivo é contribuir para o avanço do conhecimento científico em sua área. 
    Você segue rigorosamente as normas e metodologias científicas e da ABNT para garantir a qualidade e confiabilidade de suas pesquisas.
    Sua busca por informações é guiada pela busca da verdade e pela contribuição para a comunidade acadêmica.
    Pesquise e compile informações relevantes e atualizadas sobre o assunto, seguindo as normas científicas e da formatação ABNT. 
    Certifique-se de incluir referências bibliográficas adequadas.
    Um resumo detalhado e bem estruturado sobre o tema, seguindo as normas científicas e nas normais da ABNT.
    
    Escritor de Artigos": Escreva conteúdos envolventes sobre {topic}.
    Como escritor de artigos, sua habilidade em transformar assuntos complexos em narrativas envolventes é excepcional. 
    Sua escrita ilumina novas perspectivas e descobertas, tornando-as acessíveis para todos. 
    Através do seu trabalho, você destaca os aspectos mais importantes das últimas tendências em diversos temas, tornando o mundo intricado das notícias uma jornada fascinante para seus leitores.

    Avaliador de Artigos": Avalie criticamente artigos acadêmicos sobre {topic}.
    Como avaliador de artigos, você possui habilidades analíticas aguçadas e um profundo entendimento do processo de pesquisa acadêmica. 
    Sua análise crítica destaca não apenas os pontos fortes, mas também as falhas potenciais nos artigos, contribuindo para a melhoria contínua da qualidade da pesquisa acadêmica.
    
    """)

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
        goal="Encontre informações confiáveis e atuais sobre {topic} seguindo as normas científicas e das normais da ABNT",
        verbose=True,
        memory=True,
        backstory=(
            "Como pesquisador acadêmico, seu objetivo é contribuir para o avanço do conhecimento científico em sua área. Você segue rigorosamente as normas e metodologias científicas e da ABNT para garantir a qualidade e confiabilidade de suas pesquisas. Sua busca por informações é guiada pela busca da verdade e pela contribuição para a comunidade acadêmica."
        ),
        tools=[search_tool],
        allow_delegation=True,
        llm=ChatGroq(api_key=groq_api_key, model_name=model_choice)
    )

    blog_writer = Agent(
        role="Escritor de Artigos",
        goal="Escreva conteúdos envolventes sobre {topic}",
        verbose=True,
        memory=True,
        backstory=(
            "Como escritor de artigos, sua habilidade em transformar assuntos complexos em narrativas envolventes é excepcional. Sua escrita ilumina novas perspectivas e descobertas, tornando-as acessíveis para todos. Através do seu trabalho, você destaca os aspectos mais importantes das últimas tendências em diversos temas, tornando o mundo intricado das notícias uma jornada fascinante para seus leitores."
        ),
        tools=[search_tool],
        allow_delegation=False,
        llm=ChatGroq(api_key=groq_api_key, model_name=model_choice)
    )

    article_evaluator = Agent(
        role="Avaliador de Artigos",
        goal="Avalie criticamente artigos acadêmicos sobre {topic}",
        verbose=True,
        memory=True,
        backstory=(
            "Como avaliador de artigos, você possui habilidades analíticas aguçadas e um profundo entendimento do processo de pesquisa acadêmica. Sua análise crítica destaca não apenas os pontos fortes, mas também as falhas potenciais nos artigos, contribuindo para a melhoria contínua da qualidade da pesquisa acadêmica."
        ),
        tools=[search_tool],
        allow_delegation=False,
        llm=ChatGroq(api_key=groq_api_key, model_name=model_choice)
    )

    research_task = Task(
        description=(
            "Pesquise e compile informações relevantes e atualizadas sobre {topic} seguindo as normas científicas e da formatação ABNT. Certifique-se de incluir referências bibliográficas adequadas."
        ),
        expected_output="Um resumo detalhado e bem estruturado sobre {topic} seguindo as normas científicas e nas normais da ABNT.",
        tools=[search_tool],
        agent=academic_researcher
    )

    write_task = Task(
        description=(
            "Escreva um artigo envolvente sobre {topic}, focando nas últimas tendências e como elas estão impactando a indústria. Este artigo deve ser fácil de entender, envolvente e positivo."
        ),
        expected_output="Um artigo de 4 parágrafos sobre os avanços em {topic}, formatado em markdown.",
        tools=[search_tool],
        agent=blog_writer,
        async_execution=False,
        output_file="blog-post.md"
    )

    evaluate_task = Task(
        description=(
            "Avalie criticamente artigos acadêmicos sobre {topic}. Destaque os pontos fortes e as falhas potenciais nos artigos."
        ),
        expected_output="Uma avaliação crítica dos artigos acadêmicos sobre {topic}, destacando os pontos fortes e as falhas potenciais.",
        tools=[search_tool],
        agent=article_evaluator
    )

    crew = Crew(
        agents=[academic_researcher, blog_writer, article_evaluator],
        tasks=[research_task, write_task, evaluate_task],
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
                st.warning(f"Rate limit exceeded. Waiting for {e.retry_after} seconds before trying again...")
                time.sleep(e.retry_after)

if __name__ == "__main__":
    main()
