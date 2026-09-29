from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

app = FastAPI(
    title="Email Analyzer API",
    description="Сервис извлекает данные из писем клиентов с помощью локальной LLM",
    version="0.1.0",
)


class EmailIn(BaseModel):
    text: str


class ClientLead(BaseModel):
    client_name: str | None = None
    company: str | None = None
    intent: str | None = None
    is_angry: bool = False


SYSTEM_PROMPT = """
Ты — сервис извлечения данных из писем клиентов.

Твоя задача:
- читать текст письма;
- извлекать имя клиента, компанию, суть обращения и уровень недовольства;
- возвращать результат только в виде валидного JSON-объекта.

Правила:
- не добавляй пояснений;
- не используй markdown;
- не выдумывай данные;
- если поле неизвестно, верни null;
- если текст не похож на письмо клиента, верни intent = "out_of_scope".
"""

USER_PROMPT_TEMPLATE = """
Проанализируй письмо клиента.

Письмо:
<<<
{text}
>>>

Верни только валидный JSON-объект со следующими полями:
{{
  "client_name": строка или null,
  "company": строка или null,
  "intent": строка или null,
  "is_angry": true или false
}}
"""


def ask_llm(text: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "system": SYSTEM_PROMPT,
        "prompt": USER_PROMPT_TEMPLATE.format(text=text),
        "stream": False,
        "format": "json",
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()
    data = response.json()

    return data.get("response", "")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-email", response_model=ClientLead)
def analyze_email(email: EmailIn):
    if not email.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Поле text пустое",
        )

    try:
        raw_response = ask_llm(email.text)
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Ollama слишком долго не отвечает",
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Не удалось подключиться к Ollama. Проверь, запущена ли она.",
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка при обращении к Ollama: {str(e)}",
        )

    if not raw_response.strip():
        raise HTTPException(
            status_code=502,
            detail="LLM вернула пустой ответ",
        )

    try:
        return ClientLead.model_validate_json(raw_response)
    except ValidationError as e:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "LLM вернула некорректный ответ",
                "raw_response": raw_response,
                "validation_errors": e.errors(),
            },
        )