import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Caminhos de importação corrigidos, assumindo que são os corretos.
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, ChatMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def uploadar_e_processar_arquivos():
    arquivos_uploads = st.file_uploader("Faça o upload de seus arquivos (JSON, CSV, XLSX até 200MB cada)", 
                                        type=['json', 'csv', 'xlsx'], accept_multiple_files=True)
    if arquivos_uploads:
        for arquivo in arquivos_uploads:
            try:
                if arquivo.type == "application/json":
                    dados = pd.read_json(arquivo)
                elif arquivo.type == "text/csv":
                    dados = pd.read_csv(arquivo)
                elif arquivo.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    dados = pd.read_excel(arquivo)
                
                st.write(f"Visualização prévia do arquivo {arquivo.name}:")
                visualizar_dados(dados)
            except Exception as e:
                st.error(f"Erro ao processar o arquivo {arquivo.name}: {e}")

def visualizar_dados(dados):
    if not dados.empty:
        st.write("Visualização gráfica dos dados:")
        sns.pairplot(dados.select_dtypes(include=[np.number]).dropna())
        plt.show()

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface Avançada de Chat com RAG")
    st.image("Untitled.png", width=100)
    st.title("Bem-vindo à Interface Avançada de Chat com RAG!")

    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")
    uploadar_e_processar_arquivos()

    pergunta_usuario = st.text_input("Faça uma pergunta:")
    if pergunta_usuario:
        mensagens_chat = [
            SystemMessage(content="Como posso ajudá-lo hoje?"),
            *[ChatMessage(content=msg) for msg in st.session_state.get('chat_history', [])],
            HumanMessagePromptTemplate(template="{human_input}")
        ]

        prompt = ChatPromptTemplate(mensagens=mensagens_chat)
        chat_groq = ChatGroq(api_key=os.getenv('GROQ_API_KEY', 'Chave_API_Padrão'), model_name="modelo_exemplo")
        conversa = LLMChain(llm=chat_groq, prompt=prompt, memoria=memory)
        resposta = conversa.predict(human_input=pergunta_usuario)
        st.session_state['chat_history'].append(pergunta_usuario)
        st.session_state['chat_history'].append(resposta)
        st.write("Chatbot:", resposta)

if __name__ == "__main__":
    main()
