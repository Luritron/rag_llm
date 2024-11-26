from pydantic import BaseModel
from ninja import NinjaAPI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage
import requests

# Pydantic-схема для POST-запроса
class QuestionSchema(BaseModel):
    question: str

# Инициализация системы RAG
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)
db = Chroma(persist_directory="./db-hormozi", embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
local_llm = 'gemma2:2b'
llm = ChatOllama(model=local_llm, keep_alive="3h", max_tokens=1024, temperature=0)

# Шаблон для RAG-запросов
def build_prompt(context, question):
    return (
        f"Answer the following question based only on the provided context:\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n"
    )

# Функция для обращения к Google Custom Search API
def search_online_google(question):
    api_key = "AIzaSyAQ5oVwP1gJlEdWmdfUa_HCyRPe8kDvdoc"
    search_engine_id = "6470dd1c7d98d4897"
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": search_engine_id, "q": question, "num": 5}
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        snippets = [item["snippet"] for item in response.json().get("items", [])]
        return "\n".join(snippets)
    return "No relevant information found online."

# API Ninja
api = NinjaAPI()

@api.post("/ask")
def ask_question(request, payload: QuestionSchema):
    # Первый шаг: поиск в базе RAG
    question = payload.question
    context = retriever.get_relevant_documents(question)
    rag_answer = ""
    for doc in context:
        rag_answer += doc.page_content

    if rag_answer.strip():
        # Если есть ответ в RAG
        prompt = build_prompt(rag_answer, question)
        response = llm.generate([[HumanMessage(content=prompt)]])
        return {"source": "database", "answer": response.generations[0][0].text}

    # Если нет ответа в RAG, обращаемся к Google
    internet_context = search_online_google(question)
    if internet_context:
        prompt = build_prompt(internet_context, question)
        response = llm.generate([[HumanMessage(content=prompt)]])
        return {"source": "internet", "answer": response.generations[0][0].text}

    # Если ответа нигде нет
    return {"source": "none", "answer": "No answer could be generated."}
