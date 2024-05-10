from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
import groq
import toml
import time

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

def main():
    # Configurações da página do Streamlit
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

    # Sidebar para customização
    primary_prompt = st.sidebar.text_input("Prompt do sistema principal", "Como posso ajudar você hoje?")
    secondary_prompt = st.sidebar.text_input("Prompt do sistema secundário", "Há algo mais em que posso ajudar?")
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    conversational_memory_length = st.sidebar.slider('Tamanho da memória conversacional', 1, 50, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    search_tool = DuckDuckGoSearchRun()
    academic_researcher = Agent(
        role="学术研究员",
        goal="在符合科学规范和ABNT规范的前提下，找到关于{topic}的可靠和最新信息",
        verbose=True,
        memory=True,
        backstory=(
            "作为学术研究员，您的目标是为推进您领域的知识发展做出贡献。您严格遵循科学和ABNT的规范和方法，以确保您的研究质量和可靠性。您对信息的搜索是由对真理的追求和对学术界的贡献驱动的。"
        ),
        tools=[search_tool],
        allow_delegation=True,
        llm=ChatGroq(api_key=groq_api_key, model_name=model_choice)
    )

    blog_writer = Agent(
        role="文章撰写者",
        goal="撰写有关{topic}的引人入胜的内容",
        verbose=True,
        memory=True,
        backstory=(
            "作为文章撰写者，您将复杂的主题转化为引人入胜的叙述的能力是异常的。您的写作为新视角和发现提供了光明，使它们对所有人都更易于理解。通过您的工作，您突出了各种主题最新趋势的最重要方面，使新闻复杂的世界对您的读者来说是一次迷人的旅程。"
        ),
        tools=[search_tool],
        allow_delegation=False,
        llm=ChatGroq(api_key=groq_api_key, model_name=model_choice)
    )

    article_evaluator = Agent(
        role="文章评估者",
        goal="对{topic}的学术文章进行批判性评价",
        verbose=True,
        memory=True,
        backstory=(
            "作为文章评估者，您具有敏锐的分析能力和对学术研究过程的深刻理解。您的批判性分析不仅突出了文章的优点，还指出了文章可能存在的缺陷，为持续改进学术研究质量做出了贡献。"
        ),
        tools=[search_tool],
        allow_delegation=False,
        llm=ChatGroq(api_key=groq_api_key, model_name=model_choice)
    )

    research_task = Task(
        description=(
            "在符合科学规范和ABNT格式的前提下，搜索和整理有关{topic}的相关和最新信息。确保包含适当的参考文献。"
        ),
        expected_output="一份详细和结构良好的关于{topic}的摘要，遵循科学规范和ABNT规范。",
        tools=[search_tool],
        agent=academic_researcher
    )

    write_task = Task(
        description=(
            "撰写有关{topic}的引人入胜的内容，着重介绍最新趋势及其对行业的影响。这篇文章应易于理解、引人入胜且积极向上。"
        ),
        expected_output="一篇关于{topic}进展的四段文章，采用markdown格式。",
        tools=[search_tool],
        agent=blog_writer,
        async_execution=False,
        output_file="blog-post.md"
    )

    evaluate_task = Task(
        description=(
            "对{topic}的学术文章进行批判性评价。突出文章的优点和可能存在的缺陷。"
        ),
        expected_output="一份对{topic}的学术文章进行批判性评价，突出文章的优点和可能存在的缺陷。",
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
        model = crew.agents[0].llm.model_name  # Assume que o modelo do primeiro agente é o modelo escolhido
        while True:
            # Verificar se excedeu o limite de tokens
            if tokens_used[model] >= rate_limits[model]:
                st.warning(f"Limite de tokens excedido para o modelo {model}. Aguardando antes de tentar novamente...")
                time.sleep(60)
                tokens_used[model] = 0
            else:
                try:
                    result = crew.kickoff(inputs={"topic": user_question})
                    st.write("Chatbot:", result)
                    tokens_used[model] += 1  # Incrementar o contador de tokens
                    break
                except groq.RateLimitError as e:
                    # Extrair o tempo de espera da mensagem de erro
                    retry_time_str = e.args[0]["error"]["message"].split("Please try again in ")[1].split(".")[0]
                    retry_time_seconds = int(retry_time_str.split("m")[0]) * 60 + float(retry_time_str.split("m")[1][:-1])
                    st.warning(f"Limite de taxa excedido. Aguardando {retry_time_seconds} segundos antes de tentar novamente...")
                    time.sleep(retry_time_seconds)

        # Exibir a contagem de tokens na barra lateral
        st.sidebar.write("### Limites de Taxa")
        st.sidebar.write("Aqui estão alguns pontos-chave sobre os limites de taxa:")
        st.sidebar.write("Os seguintes cabeçalhos são definidos (os valores são ilustrativos):")
        st.sidebar.write("```")
        st.sidebar.write("Cabeçalho\tValor\tAnotações")
        st.sidebar.write("retry-after\t2\tEm segundos")
        st.sidebar.write("x-ratelimit-limit-requests\t14400\tSempre se refere a Solicitações por Dia (RPD)")
        st.sidebar.write("x-ratelimit-limit-tokens\t18000\tSempre se refere a Tokens por Minuto (TPM)")
        st.sidebar.write("x-ratelimit-remaining-requests\t14370\tSempre se refere a Solicitações por Dia (RPD)")
        st.sidebar.write("x-ratelimit-remaining-tokens\t17997\tSempre se refere a Tokens por Minuto (TPM)")
        st.sidebar.write("x-ratelimit-reset-requests\t2m59.56s\tSempre se refere a Solicitações por Dia (RPD)")
        st.sidebar.write("x-ratelimit-reset-tokens\t7.66s\tSempre se refere a Tokens por Minuto (TPM)")
        st.sidebar.write("```")
        st.sidebar.write("Quando o limite de taxa é atingido, retornamos um código de status HTTP Too Many Requests.429")
    
        # Exibir a tabela de limites de taxa
        st.sidebar.write("### Limites de Taxa")
        st.sidebar.write("Estes são os limites de taxa para sua organização:")
        st.sidebar.write("```")
        st.sidebar.write("ID\tSolicitações por Minuto\tPedidos por Dia\tTokens por Minuto")
        st.sidebar.write("MixTral-8X7B-32768\t30\t14.400\t5.000")
        st.sidebar.write("Lhama3-70B-8192\t30\t14.400\t6.000")
        st.sidebar.write("lhama3-8b-8192\t30\t14.400\t30.000")
        st.sidebar.write("gemma-7b-it\t30\t14.400\t15.000")
        st.sidebar.write("```")
    
        # Contagem de tokens usados
        st.sidebar.write("### Contagem de Tokens Usados")
        for model, tokens in tokens_used.items():
            st.sidebar.write(f"{model}: {tokens}")

        

if __name__ == "__main__":
    main()
