import streamlit as st
import os
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from crewai_tools import DuckDuckGoSearchRun

def upload_json_data():
    """
    Permite aos usuários fazer upload de arquivos JSON que podem ser usados como fonte de dados.
    Os arquivos são carregados através de um widget de upload no Streamlit e lidos como DataFrames.
    """
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos JSON (até 2 arquivos, 200MB cada)", type='json', accept_multiple_files=True, key="json_upload")
    if uploaded_files:
        data_frames = []
        for file in uploaded_files:
            try:
                # Lê o arquivo JSON e tenta convertê-lo em DataFrame
                data = pd.read_json(file)
                data_frames.append(data)
                st.write(f"Pré-visualização do arquivo JSON carregado:")
                st.dataframe(data.head())
            except ValueError as e:
                # Caso ocorra um erro na leitura do JSON, mostra uma mensagem de erro
                st.error(f"Erro ao ler o arquivo {file.name}: {e}")
        st.session_state['uploaded_data'] = data_frames

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG")
    st.image("Untitled.png", width=100)
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.")

    groq_api_key = os.getenv('GROQ_API_KEY', 'Chave_API_Padrão')

    st.sidebar.title('Customização')
    primary_prompt = st.sidebar.text_input("Prompt do sistema principal", "Como posso ajudar você hoje?")
    secondary_prompt = st.sidebar.text_input("Prompt do sistema secundário", "Há algo mais em que posso ajudar?")
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    conversational_memory_length = st.sidebar.slider('Tamanho da memória conversacional', 1, 50, value=5)
    tokens_per_minute = st.sidebar.slider('Tokens por minuto', 100, 5000, value=3000)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    groq_chat = ChatGroq(api_key=groq_api_key, model_name=model_choice, tokens_per_minute=tokens_per_minute)

    search_tool = DuckDuckGoSearchRun()
    researcher=Agent(
        role="Pesquisador Sênior",
        goal="descubra as três principais notícias de renderização em {topic}",
        verbose=True,
        memory=True,
        backstory=(
            """
        Como assistente de pesquisa dedicado a descobrir as tendências mais impactantes, você é movido por uma curiosidade implacável e um compromisso com a inovação. Sua função envolve aprofundar-se nos desenvolvimentos mais recentes em vários setores para identificar e analisar as principais notícias de tendência em qualquer campo. Essa busca não apenas satisfaz sua sede de conhecimento
        mas também permite que você contribua com insights valiosos que podem
        potencialmente remodelar entendimentos e expectativas em escala global
        """
        ),
        tools=[search_tool],
        allow_delegation=True,
        llm=groq_chat
    )

    blog_writer=Agent(
        role="Escritor Especialista",
        goal="write compelling contents about {topic}",
        verbose=True,
        memory=True,
        backstory=(
        """
         Armed with the knack for distilling complex subjects into digestible,
        compelling stories, you, as a blog writer, masterfully weave narratives
        that both enlighten and engage your audience. Your writing illuminates fresh
        insights and discoveries, making them approachable for everyone. Through your craft,
        you bring to the forefront the essence of new developments across various topics,
        making the intricate world of news a fascinating journey for your readers.
        """
        ),
        tools=[search_tool],
        allow_delegation=False,
        llm=groq_chat
    )

    research_task=Task(
        description=(
        """
        Identify the next big trend in {topic}.
        Focus on identifying pros and cons and the overall narrative.
        Your final report should clearly articulate the key points
        its market opportunities, and potential risks.
        """
        ),
        expected_output="A comprehensive 3 paragraphs long report on the {topic}",
        tools=[search_tool],
        agent=researcher
    )

    write_task=Task(
        description=(
        """
        Compose an insightful article on {topic}.
        Focus on the latest trends and how it's impacting the industry.
        This article should be easy to understand, engaging, and positive.
        """
        ),
        expected_output="A 4 paragraph article on {topic} advancements formatted as markdown traduzido em portugues.",
        tools=[search_tool],
        agent=blog_writer,
        aync_execution=False,
        output_file="blog-post.md"
    )
    crew=Crew(
        agents=[researcher,blog_writer],
        tasks=[research_task,write_task],
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
