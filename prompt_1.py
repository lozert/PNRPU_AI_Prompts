import json
import base64
from openai import OpenAI


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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


if __name__ == "__main__":
    # Path to your image, works with jpeg
    image_path1 = r"C:\Users\Mobil\Desktop\AI_tech\All_Data\data\drawings\35.jpeg"
    image_path2 = r"C:\Users\Mobil\Desktop\AI_tech\All_Data\data\drawings\36.jpeg"
    base_url = "http://10.66.80.3:9000/v1"
    model_name = "Qwen/Qwen2-VL-72B-Instruct-AWQ"

    # Getting the base64 string
    base64_image1 = encode_image(image_path1)
    base64_image2 = encode_image(image_path2)

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "расскажи, что это за чертежи"
                 },
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{base64_image1}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}"}}
            ]
        }
    ]

    completion = get_response(base_url=base_url, messages=msgs, model=model_name)
    for chunk in completion:
        jtemp = json.loads(chunk.model_dump_json())
        if len(jtemp["choices"]) > 0:
            print(jtemp["choices"][0]["delta"]["content"], end="")
