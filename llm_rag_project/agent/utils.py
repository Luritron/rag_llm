from pathlib import Path
import fitz
import re
import requests
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate

local_llm = 'gemma2:2b'
llm = ChatOllama(model=local_llm,
                 keep_alive="3h",
                 max_tokens=1024,
                 temperature=0)

# Создание объекта embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

template = """<bos><start_of_turn>user\nAnswer the question based only on the following context and extract out a meaningful answer. \
Please write in full sentences with correct spelling and punctuation. if it makes sense use lists. \
Please respond with the exact phrase "unable to find an answer" if the context does not provide an answer. Do not include any other text.\

CONTEXT: {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model\n
ANSWER:"""
prompt = ChatPromptTemplate.from_template(template)

def build_prompt_with_history(messages, context, question):
    history_content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    prompt = (
        f"Conversation History:\n{history_content}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Please provide a detailed and helpful answer."
    )
    return prompt

# Функция для получения пути к базе данных Chroma
def get_chroma_db_path(dialog_id):
    base_dir = Path("./db-hormozi")
    dialog_dir = base_dir / dialog_id
    dialog_dir.mkdir(parents=True, exist_ok=True)
    return dialog_dir

# Функция для извлечения текста из PDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Добавить функцию для обращения к Google Custom Search API
def search_online_google(question):
    api_key = "AIzaSyAQ5oVwP1gJlEdWmdfUa_HCyRPe8kDvdoc"  # Замени на свой API-ключ
    search_engine_id = "6470dd1c7d98d4897"  # Замени на свой CX ID
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": "AIzaSyAQ5oVwP1gJlEdWmdfUa_HCyRPe8kDvdoc",
        "cx": "6470dd1c7d98d4897",
        "q": question,
        "num": 5,  # Количество результатов
    }

    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        results = response.json()
        # Извлекаем сниппеты из результатов
        snippets = [item["snippet"] for item in results.get("items", [])]
        return "\n".join(snippets)
    else:
        return "No relevant information found online."

# Проверка, есть ли ответ
def is_rag_answer_unavailable(answer):
    # Ключевые фразы, которые указывают на отсутствие ответа
    negative_responses = [
        "unable to find an answer",
        "does not contain information",
        "unable to provide information",
    ]
    # Проверяем наличие любой из ключевых фраз в ответе
    return any(phrase in answer.lower() for phrase in negative_responses)

def format_llm_answer(answer):
    # Преобразование заголовков (например, ## -> <h2>)
    answer = re.sub(r"^(##)(\s+)(.*)$", r"<h2>\3</h2>", answer, flags=re.MULTILINE)
    answer = re.sub(r"^(#)(\s+)(.*)$", r"<h1>\3</h1>", answer, flags=re.MULTILINE)

    # Преобразование жирного текста (например, **текст** -> <strong>текст</strong>)
    answer = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", answer)

    # Преобразование списков: добавление переноса строки перед элементами списка
    answer = re.sub(r"(\S)\s*\* ", r"\1<br>* ", answer)  # Если элемент списка идёт подряд с текстом
    answer = re.sub(r"^\*\s*(.*)$", r"<br>* \1", answer, flags=re.MULTILINE)  # Для элементов списка на новой строке

    # Преобразование нумерованных списков
    answer = re.sub(r"^\d+\.\s*(.*)$", r"<br>1. \1", answer, flags=re.MULTILINE)

    # Обработка параграфов (разделение на абзацы)
    answer = re.sub(r"\n\n", r"</p><p>", answer)
    answer = f"<p>{answer}</p>"

    return answer

