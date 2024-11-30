import requests
import re
import os
import uuid
import fitz
from pathlib import Path
from tqdm import tqdm
from .models import DialogHistory
from .tasks import process_uploaded_files, process_question
from .utils import extract_text_from_pdf, get_chroma_db_path, embeddings, format_llm_answer

from pydantic import BaseModel
from ninja import NinjaAPI

# API Ninja
api = NinjaAPI()

class QuestionSchema(BaseModel):
    question: str
    dialog_id: str  # Добавляем поле dialog_id

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

@api.get("/dialogs")
def list_dialogs(request):
    user_id = request.headers.get("X-User-ID", "default_user")
    dialogs = (
        DialogHistory.objects.filter(user_id=user_id)
        .values("dialog_id")
        .distinct()
        .order_by("-timestamp")
    )
    return {"dialogs": [dialog["dialog_id"] for dialog in dialogs]}

@api.post("/dialogs/new")
def start_new_dialog(request):
    user_id = request.headers.get("X-User-ID", "default_user")
    dialog_id = str(uuid.uuid4())  # Генерация уникального ID для нового диалога
    # Создаём первое сообщение для идентификации диалога
    DialogHistory.objects.create(user_id=user_id, dialog_id=dialog_id, role="system", content="New dialog started.")

    # Создание отдельной базы данных
    get_chroma_db_path(dialog_id)
    return {"dialog_id": dialog_id}

@api.get("/dialogs/{dialog_id}")
def get_dialog_messages(request, dialog_id: str):
    user_id = request.headers.get("X-User-ID", "default_user")
    messages = DialogHistory.objects.filter(user_id=user_id, dialog_id=dialog_id).order_by("timestamp")
    return {
        "messages": [
            {"role": message.role, "content": format_llm_answer(message.content), "timestamp": message.timestamp}
            for message in messages
        ]
    }


@api.post("/upload_files")
def upload_files(request):
    dialog_id = request.POST.get("dialog_id")
    uploaded_files = request.FILES.getlist("files")
    upload_dir = Path(f"./llm_rag_project/txt_files/{dialog_id}")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = upload_dir / uploaded_file.name
        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        file_paths.append(str(file_path))

    # Отправляем задачу в Celery
    process_uploaded_files.delay(dialog_id, file_paths)
    return {"status": "Files uploaded and indexing started."}

@api.post("/ask")
def ask_question(request, payload: QuestionSchema):
    question = payload.question
    dialog_id = payload.dialog_id  # Добавляем dialog_id из запроса
    user_id = request.headers.get("X-User-ID", "default_user")

    task = process_question.delay(dialog_id, question, user_id)
    return {"task_id": task.id, "status": "processing"}