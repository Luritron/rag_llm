import requests
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


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

# Добавить функцию для обращения к Google Custom Search API
def search_online_google(question):
    api_key = "AIzaSyAQ5oVwP1gJlEdWmdfUa_HCyRPe8kDvdoc"  # Замени на свой API-ключ
    search_engine_id = "6470dd1c7d98d4897"  # Замени на свой CX ID
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": "AIzaSyAQ5oVwP1gJlEdWmdfUa_HCyRPe8kDvdoc",
        "cx": "6470dd1c7d98d4897",
        "q": question,
        "num": 1,  # Количество результатов
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

# Функция для обработки вопросов
def ask_question(question):
    print("Answer:\n\n", end=" ", flush=True)

    # Первый шаг: RAG
    rag_answer = ""
    for chunk in rag_chain.stream(question):
        rag_answer += chunk.content
        print(chunk.content, end="", flush=True)

    # Проверка, есть ли ответ
    if is_rag_answer_unavailable(rag_answer):
        print("\nNo answer in RAG. Searching online with Google...\n")
        # Второй шаг: обращение к Google Custom Search API
        online_context = search_online_google(question)
        if online_context:
            # print("\nInternet Context Found:\n", online_context)
            messages = [[
                HumanMessage(content=(
                    f"Answer the following question based on the provided context:\n\n"
                    f"CONTEXT:\n{online_context}\n\nQUESTION:\n{question}\n"
                ))
            ]]
            try:
                internet_answer = llm.generate(messages)
                print("\nAnswer based on online data:\n", internet_answer.generations[0][0].text)
            except Exception as e:
                print(f"\nError during LLM generation: {e}")
    else:
        print("\nFull answer received.\n")



# Основная логика программы
if __name__ == "__main__":
    while True:
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        ask_question(user_question)
