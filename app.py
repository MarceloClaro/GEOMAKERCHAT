import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def upload_data():
    """
    Permite o upload de arquivos JSON e CSV e armazena-os no estado da sessão para acesso posterior.
    Gera uma pré-visualização dos dados carregados em forma de tabela.
    """
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos (JSON ou CSV, até 300MB cada)", type=['json', 'csv'], accept_multiple_files=True)
    data_frames = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                if file.type == "application/json":
                    data = pd.read_json(file)
                else:  # CSV
                    data = pd.read_csv(file)
                data_frames.append(data)
                st.session_state[file.name] = data  # Armazenar dados na sessão
                st.write(f"Pré-visualização do arquivo {file.name} carregado:")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Erro ao processar o arquivo {file.name}: {e}")
    return data_frames

def show_data_visualizations(data_frames):
    """
    Gera visualizações gráficas para os DataFrames carregados, assumindo que os dados são adequados para plotagem.
    """
    for df in data_frames:
        try:
            st.write("Visualização dos dados:")
            st.bar_chart(df.iloc[:, 0:5])  # Mostrar um gráfico de barras dos primeiros 5 campos numéricos
        except Exception as e:
            st.error(f"Não foi possível gerar gráficos para os dados: {e}")

def perform_rag(question, data_frames):
    """
    Realiza uma busca avançada utilizando TF-IDF e similaridade de cosseno nos DataFrames para responder às perguntas baseadas em dados carregados.
    """
    if not data_frames:
        return "Nenhum dado disponível para responder à pergunta."

    all_text = []
    for df in data_frames:
        all_text.extend(df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist())
    
    tfidf_matrix, vectorizer = vectorize_text(all_text + [question])
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    highest_index = similarities.argmax()
    
    if similarities[highest_index] > 0:
        return f"Com base nos dados carregados, aqui está uma informação relevante: {all_text[highest_index]}"
    else:
        return "Nenhum dado relevante encontrado para responder à pergunta."

def vectorize_text(data):
    """
    Vetoriza o texto usando TF-IDF para comparações de similaridade textual.
    """
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(data), vectorizer

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG")
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.")

    groq_api_key = os.getenv('GROQ_API_KEY', 'Chave_API_Padrão')
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")  # Memória conversacional
    data_frames = upload_data()

    if st.button('Mostrar Gráficos'):
        show_data_visualizations(data_frames)

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        response = perform_rag(user_question, data_frames)
        st.write("Resposta do Chatbot:", response)

if __name__ == "__main__":
    main()
