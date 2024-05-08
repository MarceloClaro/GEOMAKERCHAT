from langchain_core.messages import BaseMessage

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG")
    st.image("Untitled.png", width=100)  # Assegure-se de que o caminho está correto
    st.title("Bem-vindo ao Chat Geomaker Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.")

    groq_api_key = os.getenv('GROQ_API_KEY', 'Chave_API_Padrão')
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")
    data_frames = upload_json_data()  # Assume que essa função foi definida corretamente para tratar os dados JSON

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        if 'last_prompt' not in st.session_state:
            st.session_state.last_prompt = "Como posso ajudar você hoje?"
        current_prompt = secondary_prompt if st.session_state.last_prompt == primary_prompt else primary_prompt
        st.session_state.last_prompt = current_prompt

        # Assegure-se de que as mensagens no histórico são instâncias de BaseMessage ou uma subclasse adequada
        messages = [BaseMessage(content=msg, role='user' if idx % 2 == 0 else 'system') for idx, msg in enumerate(st.session_state.chat_history)]
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=current_prompt),
            MessagesPlaceholder(messages=messages, variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        # Utilizando o LLMChain da LangChain
        groq_chat = ChatGroq(api_key=groq_api_key, model_name=model_choice)
        conversation = LLMChain(llm=groq_chat, prompt=prompt, memory=memory)
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message['human'])
        st.session_state.chat_history.append(message['AI'])
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
