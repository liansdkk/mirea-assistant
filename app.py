import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Загрузка ключа
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Заголовок
st.set_page_config(page_title="Ассистент РТУ МИРЭА", page_icon="🎓")
st.title("🎓 Ассистент приёмной комиссии РТУ МИРЭА")
st.caption("Задайте вопрос о поступлении в 2026 году. Бот отвечает строго по правилам приёма.")

# Подключаем базу знаний (один раз при запуске)
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)

vectorstore = load_vectorstore()

# История диалога
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Здравствуйте! Я виртуальный ассистент приёмной комиссии РТУ МИРЭА. Задайте вопрос о поступлении — я найду ответ в правилах приёма 2026 года."}
    ]

# Показываем историю
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Поле ввода
if prompt := st.chat_input("Ваш вопрос..."):
    # Добавляем вопрос в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ищем контекст в базе знаний
    with st.spinner("Ищу информацию в правилах приёма..."):
        docs = vectorstore.similarity_search(prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

    # Формируем запрос к OpenAI
    system_prompt = """Ты — официальный ассистент приёмной комиссии РТУ МИРЭА.
Отвечай СТРОГО на основе переданного контекста из правил приёма.
Если в контексте нет ответа на вопрос — честно скажи: «Этой информации нет в моей базе знаний. Уточните на priem.mirea.ru».
Не придумывай факты, цифры и даты.
Будь вежлив и краток."""

    with st.spinner("Формирую ответ..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Контекст из правил приёма:\n\n{context}\n\nВопрос абитуриента: {prompt}\n\nОтвет:"}
            ]
        )
        answer = response.choices[0].message.content

    # Показываем ответ
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})