from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import streamlit as st
from crewai import Agent, Crew
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
import toml

# Carregar a chave de API do Groq do arquivo secrets.toml
secrets = toml.load("secrets.toml")
groq_api_key = secrets["GROQ_API_KEY"]

# Definir os limites de taxa para cada modelo
rate_limits = {
    "llama3-70b-8192": 6000,
    "llama3-8b-8192": 30000,
    "gemma-7b-it": 15000,
    "mixtral-8x7b-32768": 5000
}

# Inicializar um dicionário para rastrear os tokens usados por cada modelo
tokens_used = {model: 0 for model in rate_limits}

def create_agent(role, goal, backstory, tools, model_name):
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools,
        llm=ChatGroq(api_key=groq_api_key, model_name=model_name),
        max_iter=15,  # Limite de iterações
        verbose=True,
        cache=True  # Cache habilitado
    )

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG+CreWAI")
    st.image("Untitled.png", width=100)
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAG+CreWAI!")

    st.write("""
    Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.
    
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

    # Sidebar para customização
    st.sidebar.title("Customização")
    st.sidebar.subheader("Configuração do Sistema")
    primary_prompt = st.sidebar.text_input("Prompt do sistema principal", "Como posso ajudar você hoje?")
    secondary_prompt = st.sidebar.text_input("Prompt do sistema secundário", "Há algo mais em que posso ajudar?")
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    conversational_memory_length = st.sidebar.slider('Tamanho da memória conversacional', 1, 50, value=5)
    search_topic = st.sidebar.text_input("Tema da pesquisa", "Insira o tema que deseja pesquisar")

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    search_tool = DuckDuckGoSearchRun()

    agents = [
        create_agent(
            "学术研究员",
            f"在符合科学规范和ABNT规范的前提下，找到关于{search_topic}的可靠和最新信息",
            "作为学术研究员，您的目标是为推进您领域的知识发展做出贡献。您严格遵循科学和ABNT的规范和方法，以确保您的研究质量和可靠性。您对信息的搜索是由对真理的追求和对学术界的贡献驱动的。",
            [search_tool],
            model_choice
        ),
        create_agent(
            "文章撰写者",
            f"撰写有关{search_topic}的引人入胜的内容",
            "作为文章撰写者，您将复杂的主题转化为引人入胜的叙述的能力是异常的。您的写作为新视角和发现提供了光明，使它们对所有人都更易于理解。",
            [search_tool],
            model_choice
        ),
        create_agent(
            "文章评估者",
            f"对{search_topic}的学术文章进行批判性评价",
            "作为文章评估者，您具有敏锐的分析能力和对学术研究过程的深刻理解。您的批判性分析不仅突出了文章的优点，还指出了文章可能存在的缺陷，为持续改进学术研究质量做出了贡献。",
            [search_tool],
            model_choice
        )
    ]

    agent_names = ["Pesquisador Acadêmico", "Escritor de Artigos", "Avaliador de Artigos"]
    selected_agent = st.sidebar.selectbox("Escolha um agente", agent_names)

    for agent in agents:
        if agent.role == selected_agent:
            current_agent = agent

    # Exibir a contagem de tokens na barra lateral
    st.sidebar.write(f"Tokens usados ({model_choice}): {tokens_used[model_choice]} de {rate_limits[model_choice]}")

    # Obter a mensagem do usuário
    user_input = st.text_input("Você:", value="")

    if st.button("Enviar"):
        # Executar a interação com o agente
        try:
            response = current_agent.interact(user_input)
            st.write(f"{current_agent.role}:", response)
        except Exception as e:
            retry_time_str = None
            if isinstance(e.args[0], dict) and "error" in e.args[0]:
                retry_time_str = e.args[0]["error"]

            if retry_time_str is not None:
                st.warning(f"Você atingiu o limite de taxa para o modelo {model_choice}. Tente novamente em {retry_time_str}")
            else:
                st.error("Ocorreu um erro ao interagir com o agente. Por favor, tente novamente.")

    st.write("Exemplo de conversa:")
    for message in st.session_state.chat_history:
        st.write(message["agent"], ":", message["text"])

if __name__ == "__main__":
    main()
