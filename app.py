import streamlit as st
import os
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
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

    groq_api_key = os.getenv('GROQ_API_KEY', 'Chave_API_Padrão')
    model_choice = "modelo_exemplo"  # Substitua "modelo_exemplo" pelo modelo desejado
    conversational_memory_length = 5  # Ajuste conforme necessário

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history")
    uploadar_e_processar_arquivos()

    pergunta_usuario = st.text_input("Faça uma pergunta:")
    if pergunta_usuario:
        mensagens_chat = [
            SystemMessage(content="Como posso ajudar você hoje?"),
            HumanMessagePromptTemplate(template="{human_input}")
        ]

        prompt = ChatPromptTemplate(mensagens=mensagens_chat)
        groq_chat = ChatGroq(api_key=groq_api_key, model_name=model_choice)
        conversa = LLMChain(llm=groq_chat, prompt=prompt=memory)
        resposta = conversa.predict(human_input=pergunta_usuario)
        st.session_state['chat_history'].append(pergunta_usuario)
        st.write("Chatbot:", resposta)

if __name__ == "__main__":
    main()
