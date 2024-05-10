import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
import toml
import time  # Para adicionar um pequeno atraso entre as solicitações

# Carregar a chave de API do Groq do arquivo secrets.toml
secrets = toml.load("secrets.toml")
groq_api_key = secrets["GROQ_API_KEY"]

# Interface do chat Streamlit
st.title("Chat de Pesquisa Científica")
topic = st.text_input("Digite o tópico da pesquisa:", "avanços científicos")

# Controle deslizante para o usuário controlar o número de tokens por minuto (TPM)
tokens_per_minute = st.slider('Tokens por minuto', 100, 5000, value=3000)

# Inicializar o modelo Groq com o número de tokens por minuto configurado
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192", tokens_per_minute=tokens_per_minute)

# Definir agentes
researcher = Agent(
    role="Pesquisador Científico",
    goal="descubra os três principais avanços científicos em {topic}",
    verbose=True,
    memory=True,
    backstory=(
        """
    Como pesquisador científico, sua missão é explorar e descobrir os mais recentes avanços científicos em seu campo de estudo. Sua paixão pela ciência e sua curiosidade insaciável o impulsionam a buscar constantemente novos conhecimentos e descobertas que possam revolucionar nossa compreensão do mundo. Seu trabalho é crucial para avançar o conhecimento humano e contribuir para o progresso da ciência.
    """
    ),
    llm=llm
)

report_writer = Agent(
    role="Escritor de Relatórios Científicos",
    goal="escrever um relatório detalhado sobre os avanços científicos em {topic}",
    verbose=True,
    memory=True,
    backstory=(
    """
    Como escritor de relatórios científicos, sua habilidade em comunicar descobertas complexas de forma clara e precisa é fundamental. Seu relatório deve fornecer uma visão abrangente dos avanços científicos mais recentes em um campo específico, destacando sua importância e impacto potencial. Sua escrita é essencial para compartilhar descobertas científicas com a comunidade acadêmica e o público em geral.
    """
    ),
    llm=llm
)

# Definir tarefas
research_task = Task(
    description=(
    """
    Identifique os próximos grandes avanços em {topic}.
    Concentre-se em identificar os prós e os contras e na narrativa geral.
    Seu relatório final deve articular claramente os principais pontos,
    suas oportunidades de mercado e riscos potenciais.
    """
    ),
    expected_output="Um relatório abrangente de 3 parágrafos sobre os avanços em {topic}",
    agent=researcher
)

write_task = Task(
    description=(
    """
    Escreva um artigo informativo sobre os avanços em {topic}.
    Concentre-se nas últimas tendências e em como isso está impactando a indústria.
    Este artigo deve ser fácil de entender, envolvente e positivo.
    """
    ),
    expected_output="Um artigo de 4 parágrafos sobre os avanços em {topic}, formatado como markdown traduzido em português.",
    agent=report_writer
)

# Inicializar a equipe
crew = Crew(
    agents=[researcher, report_writer],
    tasks=[research_task, write_task],
    process=Process.sequential
)

if st.button("Iniciar Pesquisa"):
    result = crew.kickoff(inputs={"topic": topic})
    st.write(result)

    # Adicionar um pequeno atraso entre as solicitações para evitar problemas de taxa
    time.sleep(1)
