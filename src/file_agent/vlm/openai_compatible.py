import base64
import io
import logging
from PIL import Image
from openai import OpenAI
from .base import VLMClient

logger = logging.getLogger(__name__)

class OpenAICompatibleVLMClient(VLMClient):
    def __init__(self, base_url: str, model: str, api_key: str = "dummy"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        try:
            # конвертация PIL Image в base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Ошибка при запросе к VLM ({self.model}): {e}")
            raise