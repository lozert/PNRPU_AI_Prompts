import json
import base64
import fitz  # PyMuPDF для работы с PDF
from PIL import Image
from openai import OpenAI
from user_data import data_image_path, data_base_url, data_prompt_file_path, data_model_name


def encode_image(image_path):
    # Проверка формата и конвертация в JPEG
    if image_path.lower().endswith(".pdf"):
        image_path = convert_pdf_to_jpeg(image_path)
    elif image_path.lower().endswith(".png") or image_path.lower().endswith(".jpg"):
        image_path = convert_to_jpeg(image_path)

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def convert_pdf_to_jpeg(pdf_path):
    pdf_document = fitz.open(pdf_path)
    first_page = pdf_document[0]
    pix = first_page.get_pixmap()
    jpeg_path = pdf_path.replace(".pdf", ".jpeg")
    pix.save(jpeg_path)
    pdf_document.close()
    return jpeg_path


def convert_to_jpeg(image_path):
    jpeg_path = image_path.rsplit('.', 1)[0] + ".jpeg"
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.save(jpeg_path, "JPEG")
    return jpeg_path


def load_prompt_from_json(prompt_file_path):
    with open(prompt_file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_response(base_url, messages, model, top_p=0.8):
    client = OpenAI(
        base_url=base_url,
        api_key="token-abc123"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        top_p=top_p,
        stream=True,
        stream_options={"include_usage": True}
    )
    return completion


def send_message(prompt_data, image_paths, base_url, model_name):
    # Определяем роли из JSON, либо используем "user" по умолчанию
    messages = []
    if isinstance(prompt_data, list):
        for entry in prompt_data:
            role = entry.get("role", "user")
            text = entry.get("text", "")
            content = [{"type": "text", "text": text}]
            for image_path in image_paths:
                base64_image = encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            messages.append({"role": role, "content": content})
    else:
        # Если одна запись (словарь) без вложенного списка
        role = prompt_data.get("role", "user")
        text = prompt_data.get("text", "")
        content = [{"type": "text", "text": text}]
        for image_path in image_paths:
            base64_image = encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        messages.append({"role": role, "content": content})

    # Отправка запроса к модели
    completion = get_response(base_url=base_url, messages=messages, model=model_name)
    for chunk in completion:
        jtemp = json.loads(chunk.model_dump_json())
        if len(jtemp["choices"]) > 0:
            print(jtemp["choices"][0]["delta"]["content"], end="")


if __name__ == "__main__":
    # Пути и настройки
    image_paths = [data_image_path]  # список с одним или несколькими изображениями
    prompt_file_path = data_prompt_file_path
    base_url = data_base_url
    model_name = data_model_name

    # Загружаем промпт из JSON
    prompt_data = load_prompt_from_json(prompt_file_path)

    # Отправка сообщения
    send_message(prompt_data, image_paths, base_url, model_name)
