import streamlit as st
import os
import pandas as pd
from langchain.llms import GroqLLM  # Atualizado para usar a interface correta
from langchain.prompts import ChatPrompt
from langchain.runnables import RunnableSequence

def upload_data():
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos (JSON ou CSV, até 300MB cada)", type=['json', 'csv'], accept_multiple_files=True)
    data_frames = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                data = pd.read_json(file) if file.type == "application/json" else pd.read_csv(file)
                data_frames.append(data)
                st.session_state[file.name] = data
                st.write(f"Pré-visualização do arquivo {file.name} carregado:")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Erro ao processar o arquivo {file.name}: {e}")
    return data_frames

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG")
    st.title("Bem-vindo ao Chat Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.")

    groq_api_key = os.getenv('GROQ_API_KEY', 'Chave_API_Padrão')
    model = GroqLLM(api_key=groq_api_key)  # Usando o novo método para criar uma instância LLM
    data_frames = upload_data()

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    primary_prompt = "Como posso ajudar você hoje?"
    secondary_prompt = "Há algo mais em que posso ajudar?"

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        current_prompt = secondary_prompt if st.session_state.get('last_prompt', primary_prompt) == primary_prompt else primary_prompt
        st.session_state['last_prompt'] = current_prompt

        prompt = ChatPrompt(content=current_prompt)
        sequence = RunnableSequence([
            prompt,  # Prompt que prepara a pergunta
            model    # LLM que processa a entrada
        ])

        response = sequence.run(user_input=user_question, chat_history=st.session_state['chat_history'])
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()

