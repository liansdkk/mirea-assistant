import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Ищем куски про "проходной балл"
print("=== ПОИСК: проходной балл ===")
docs = vectorstore.similarity_search("проходной балл", k=5)
for i, doc in enumerate(docs, 1):
    print(f"\n--- Кусок {i} ---")
    print(doc.page_content[:300])

# Ищем куски про "стоимость" или "платное"
print("\n\n=== ПОИСК: стоимость платного обучения ===")
docs = vectorstore.similarity_search("стоимость платного обучения", k=5)
for i, doc in enumerate(docs, 1):
    print(f"\n--- Кусок {i} ---")
    print(doc.page_content[:300])