from celery import shared_task
from .utils import build_prompt_with_history, prompt, llm, format_llm_answer, is_rag_answer_unavailable, search_online_google, extract_text_from_pdf, get_chroma_db_path, embeddings
from .models import DialogHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain.schema import Document
from pathlib import Path
import os

@shared_task
def process_uploaded_files(dialog_id, file_paths):
    upload_dir = Path(f"./llm_rag_project/txt_files/{dialog_id}")
    documents = []

    print(f"Started processing files for dialog {dialog_id}")
    # Обработка файлов
    for file_path in file_paths:
        file_path = Path(file_path)
        if file_path.suffix.lower() == ".pdf":
            pdf_text = extract_text_from_pdf(file_path)
            documents.append(Document(page_content=pdf_text, metadata={"source": str(file_path)}))
        elif file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as txt_file:
                text = txt_file.read()
                documents.append(Document(page_content=text, metadata={"source": str(file_path)}))

    # Разбиение текста на чанки
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300, add_start_index=True)
    texts = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        texts.extend(chunks)

    # Сохранение в векторное хранилище
    dialog_db_path = get_chroma_db_path(dialog_id).as_posix()
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=dialog_db_path)
    print(f"Completed processing files for dialog {dialog_id}")
    return f"Indexed {len(texts)} chunks for dialog {dialog_id}."

@shared_task
def process_question(dialog_id, question, user_id):
    dialog_db_path = get_chroma_db_path(dialog_id).as_posix()
    if not os.path.exists(dialog_db_path):
        print(f"Database path does not exist: {dialog_db_path}")
    db = Chroma(persist_directory=dialog_db_path, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    results = retriever.get_relevant_documents(question)
    print(f"\nRetrieved {len(results)} relevant documents:")
    # for doc in results:
    #     print(f"{doc.page_content}\n")

    if not dialog_id:
        return {"error": "Dialog ID is required."}

    # Сохранение вопроса пользователя в историю
    DialogHistory.objects.create(user_id=user_id, dialog_id=dialog_id, role="user", content=question)

    # Получение истории диалога
    # Получаем полную историю текущего диалога
    dialog_history = DialogHistory.objects.filter(
        user_id=user_id, dialog_id=dialog_id
    ).order_by("timestamp")
    messages = [{"role": entry.role, "content": entry.content} for entry in dialog_history]

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
    )

    # Первый шаг: поиск в базе данных через RAG
    rag_answer = ""
    for chunk in rag_chain.stream(question):
        rag_answer += chunk.content
    # formatted_answer = format_llm_answer(rag_answer.strip())
    # print(f"RAG Answer: {formatted_answer}")

    # Проверка на релевантность ответа
    if is_rag_answer_unavailable(rag_answer):
        print("No relevant data found in the database. Searching online...")
        # Второй шаг: поиск через Google Custom Search API
        internet_context = search_online_google(question)
        if internet_context:
            # Генерация ответа на основе найденного контекста из интернета
            internet_prompt = build_prompt_with_history(messages, internet_context, question)
            try:
                response = llm.generate([[HumanMessage(content=internet_prompt)]])
                internet_answer = response.generations[0][0].text.strip()

                # Проверка на случай, если ответ всё ещё "unable to find an answer"
                if is_rag_answer_unavailable(internet_answer):
                    print(f"Internet Answer is invalid: {internet_answer}")
                    return {"source": "none", "answer": "No relevant information found online."}

                # Сохранение ответа модели в историю
                DialogHistory.objects.create(user_id=user_id, dialog_id=dialog_id, role="model",
                                             content=internet_answer)

                print(f"Internet Answer: {internet_answer}")
                formatted_answer = format_llm_answer(internet_answer)
                return {"source": "internet", "answer": formatted_answer}
            except Exception as e:
                print(f"Error during LLM generation: {e}")
                return {"source": "none", "answer": "Failed to generate an answer from internet context."}
        else:
            return {"source": "none", "answer": "No relevant information found online."}

    try:
        db_prompt = build_prompt_with_history(messages, rag_answer, question)
        db_response = llm.generate([[HumanMessage(content=db_prompt)]])
        db_answer = db_response.generations[0][0].text.strip()

        # Проверка на случай, если ответ из базы данных некорректен
        if is_rag_answer_unavailable(db_answer):
            print(f"Database Answer is invalid: {db_answer}")
            return {"source": "none", "answer": "No relevant information found online."}

        # Сохранение ответа модели в историю
        DialogHistory.objects.create(user_id=user_id, dialog_id=dialog_id, role="model", content=db_answer)

        print(f"Database Answer: {db_answer}")
        formatted_answer = format_llm_answer(db_answer)
        return {"source": "database", "answer": formatted_answer}
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return {"source": "none", "answer": "Failed to generate an answer from database context."}