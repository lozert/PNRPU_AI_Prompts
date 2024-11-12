import json
import base64
from openai import OpenAI
from user_data import data_image_paths, data_base_url, data_prompt_file_path, data_model_name

def encode_image(image_path):
    # Конвертируем изображение в base64 для передачи в запросе
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_image}"

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
    messages = []

    # Обработка каждого элемента в `prompt_data`
    for entry in prompt_data:
        role = entry["role"]
        text = entry["text"]

        # Создаем сообщение с текстом
        content = [{"type": "text", "text": text}]

        # Если роль — `user`, добавляем изображения
        if role == "user":
            for image_path in image_paths:
                base64_image = encode_image(image_path)
                content.append({"type": "image_url", "image_url": {"url": base64_image}})

        # Добавляем сообщение с ролью и контентом
        messages.append({
            "role": role,
            "content": content
        })

    # Отправляем запрос к модели
    completion = get_response(base_url=base_url, messages=messages, model=model_name)
    for chunk in completion:
        jtemp = json.loads(chunk.model_dump_json())
        if len(jtemp["choices"]) > 0:
            print(jtemp["choices"][0]["delta"]["content"], end="")

if __name__ == "__main__":
    # Пути и настройки
    image_paths = data_image_paths
    prompt_file_path = data_prompt_file_path
    base_url = data_base_url
    model_name = data_model_name

    # Загружаем промпт из JSON
    prompt_data = load_prompt_from_json(prompt_file_path)

    # Отправка сообщения
    send_message(prompt_data, image_paths, base_url, model_name)
