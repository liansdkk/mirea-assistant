import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DB_PATH = "chroma_db"

# Подключаемся к базе
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# Твой тестовый вопрос
query = "Какие ЕГЭ нужны для поступления на программную инженерию?"

print(f"Вопрос: {query}\n")
print("=" * 60)

# Ищем 3 самых похожих куска
results = vectorstore.similarity_search(query, k=3)

for i, chunk in enumerate(results, 1):
    print(f"\n--- Кусок {i} ---")
    print(chunk.page_content[:400])
    print("...")