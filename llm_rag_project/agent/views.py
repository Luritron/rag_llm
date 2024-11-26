import requests
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from pydantic import BaseModel
from ninja import NinjaAPI

# API Ninja
api = NinjaAPI()

# Pydantic-схема для POST-запроса
class QuestionSchema(BaseModel):
    question: str

# # Create embeddingsclear
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

db = Chroma(persist_directory="./db-hormozi",
            embedding_function=embeddings)

# # Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs= {"k": 5}
)

# # Create Ollama language model - Gemma 2
local_llm = 'gemma2:2b'

llm = ChatOllama(model=local_llm,
                 keep_alive="3h",
                 max_tokens=1024,
                 temperature=0)

# Create prompt template
template = """<bos><start_of_turn>user\nAnswer the question based only on the following context and extract out a meaningful answer. \
Please write in full sentences with correct spelling and punctuation. if it makes sense use lists. \
Please respond with the exact phrase "unable to find an answer" if the context does not provide an answer. Do not include any other text.\

CONTEXT: {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model\n
ANSWER:"""
prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain using LCEL with prompt printing and streaming output
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Шаблон для обработки контекста
def build_prompt(context, question):
    return (
        f"""<bos><start_of_turn>user\nAnswer the question based only on the following context and extract out a meaningful answer. \
        Please write in full sentences with correct spelling and punctuation. if it makes sense use lists. \
        Please respond with the exact phrase "unable to find an answer" if the context does not provide an answer. Do not include any other text and gaps, spaces, symblos of \"\\n"\".\
        Just a short and fully clear "unable to find an answer" answer.\n\n\
        CONTEXT: {context}

        QUESTION: {question}

        <end_of_turn>
        <start_of_turn>model\n
        ANSWER:"""
    )

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

@api.post("/ask")
def ask_question(request, payload: QuestionSchema):
    question = payload.question

    # Первый шаг: поиск в базе данных через RAG
    rag_answer = ""
    for chunk in rag_chain.stream(question):
        rag_answer += chunk.content
    print(f"RAG Answer: {rag_answer.strip()}")

    # Проверка на релевантность ответа
    if is_rag_answer_unavailable(rag_answer):
        print("No relevant data found in the database. Searching online...")
        # Второй шаг: поиск через Google Custom Search API
        internet_context = search_online_google(question)
        if internet_context:
            # Генерация ответа на основе найденного контекста из интернета
            internet_prompt = (
                f"Based on the following context from the internet, answer the question:\n\n"
                f"CONTEXT:\n{internet_context}\n\nQUESTION: {question}\n"
                f"Please provide a detailed and helpful answer."
            )
            try:
                response = llm.generate([[HumanMessage(content=internet_prompt)]])
                internet_answer = response.generations[0][0].text.strip()

                # Проверка на случай, если ответ всё ещё "unable to find an answer"
                if is_rag_answer_unavailable(internet_answer):
                    print(f"Internet Answer is invalid: {internet_answer}")
                    return {"source": "none", "answer": "No relevant information found online."}

                print(f"Internet Answer: {internet_answer}")
                return {"source": "internet", "answer": internet_answer}
            except Exception as e:
                print(f"Error during LLM generation: {e}")
                return {"source": "none", "answer": "Failed to generate an answer from internet context."}
        else:
            return {"source": "none", "answer": "No relevant information found online."}

    # Если ответ из базы данных релевантен
    try:
        db_prompt = (
            f"Based on the following context from the database, answer the question:\n\n"
            f"CONTEXT:\n{rag_answer}\n\nQUESTION: {question}\n"
            f"Please provide a detailed and helpful answer."
        )
        db_response = llm.generate([[HumanMessage(content=db_prompt)]])
        db_answer = db_response.generations[0][0].text.strip()

        # Проверка на случай, если ответ из базы данных некорректен
        if is_rag_answer_unavailable(db_answer):
            print(f"Database Answer is invalid: {db_answer}")
            return {"source": "none", "answer": "No relevant information found online."}

        print(f"Database Answer: {db_answer}")
        return {"source": "database", "answer": db_answer}
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return {"source": "none", "answer": "Failed to generate an answer from database context."}
