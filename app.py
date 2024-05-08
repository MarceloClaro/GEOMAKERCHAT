import streamlit as st
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def upload_data():
    """
    Permite aos usuários fazer upload de arquivos JSON e CSV, armazenando-os no estado da sessão.
    """
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos (JSON ou CSV, até 300MB cada)", type=['json', 'csv'], accept_multiple_files=True)
    data_frames = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                data = pd.read_json(file) if file.type == "application/json" else pd.read_csv(file)
                data_frames.append(data)
                st.write(f"Pré-visualização do arquivo {file.name} carregado:")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Erro ao processar o arquivo {file.name}: {e}")
    return data_frames

def show_data_visualizations(data_frames):
    """
    Gera visualizações gráficas para os DataFrames carregados.
    """
    for df in data_frames:
        try:
            st.write("Visualização dos dados:")
            st.bar_chart(df.iloc[:, 0:min(5, len(df.columns))])  # Mostrar um gráfico de barras dos primeiros 5 campos numéricos
        except Exception as e:
            st.error(f"Não foi possível gerar gráficos para os dados: {e}")

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG")
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.")

    groq_api_key = os.getenv('GROQ_API_KEY', 'Chave_API_Padrão')
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")
    data_frames = upload_data()

    max_tokens = st.sidebar.slider("Máximo de Tokens:", 100, 1000, 500, 50)
    primary_prompt = st.sidebar.text_input("Prompt do sistema principal", "Como posso ajudar você hoje?")
    secondary_prompt = st.sidebar.text_input("Prompt do sistema secundário", "Há algo mais em que posso ajudar?")

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        current_prompt = secondary_prompt if memory.get_last_prompt() == primary_prompt else primary_prompt
        memory.store_prompt(current_prompt)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=current_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        conversation = LLMChain(llm=ChatGroq(api_key=groq_api_key, model_name=model_choice), prompt=prompt, memory=memory, max_tokens=max_tokens)
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

        show_data_visualizations(data_frames)  # Display visualizations automatically after data is processed

if __name__ == "__main__":
    main()
