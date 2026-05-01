import os
from dotenv import load_dotenv
from openai import OpenAI

# Загружаем ключ из .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Тестовый запрос
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Скажи: проверка пройдена"}],
    max_tokens=10
)

print(response.choices[0].message.content)