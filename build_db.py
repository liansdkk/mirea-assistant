import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Загружаем ключ
load_dotenv()

# Пути
DATA_PATH = "data"
DB_PATH = "chroma_db"

# Получаем все PDF из папки data
pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

print(f"Найдено PDF-файлов: {len(pdf_files)}")
print(f"Файлы: {pdf_files}")

# Загружаем и разбиваем
all_chunks = []

for pdf_file in pdf_files:
    file_path = os.path.join(DATA_PATH, pdf_file)
    print(f"\nОбрабатываю: {pdf_file}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"  Загружено страниц: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  Создано чанков: {len(chunks)}")

    all_chunks.extend(chunks)

print(f"\nВсего чанков: {len(all_chunks)}")

# Создаём эмбеддинги и сохраняем
print("\nСоздаю эмбеддинги...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print(f"\nГотово! База в папке '{DB_PATH}'")
print(f"Документов в базе: {vectorstore._collection.count()}")