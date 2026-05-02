import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="Ассистент РТУ МИРЭА", page_icon="🎓")
st.title("🎓 Ассистент приёмной комиссии РТУ МИРЭА")
st.caption("Задайте вопрос о поступлении в 2026 году. Бот отвечает строго по правилам приёма.")

api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    DATA_PATH = "data"
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    all_chunks = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(DATA_PATH, pdf_file))
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)

    rules_file = "rules.txt"
    if os.path.exists(rules_file):
        from langchain_community.document_loaders import TextLoader
        txt_loader = TextLoader(rules_file, encoding="utf-8")
        txt_docs = txt_loader.load()
        txt_chunks = text_splitter.split_documents(txt_docs)
        all_chunks.extend(txt_chunks)

    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

vectorstore = load_vectorstore()

SYSTEM_PROMPT = """Ты — официальный ассистент приёмной комиссии РТУ МИРЭА.
Отвечай СТРОГО на основе переданного контекста из правил приёма.
Если в контексте нет ответа на вопрос — честно скажи:
«Этой информации нет в моей базе знаний. Уточните на priem.mirea.ru».
Не придумывай факты, цифры и даты.
Будь вежлив и краток."""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Здравствуйте! Я виртуальный ассистент приёмной "
                "комиссии РТУ МИРЭА. Задайте вопрос о поступлении — "
                "я найду ответ в правилах приёма 2026 года."
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Ищу информацию в правилах приёма..."):
        docs = vectorstore.similarity_search(prompt, k=5)
        context = "\n\n".join(doc.page_content for doc in docs)

    with st.spinner("Формирую ответ..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Контекст из правил приёма:\n\n{context}\n\n"
                        f"Вопрос абитуриента: {prompt}\n\nОтвет:"
                    ),
                },
            ],
        )
        answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})