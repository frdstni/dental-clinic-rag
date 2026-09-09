from openai import OpenAI

from dental_rag.config.settings import settings


class OpenAILLM:
    """
    OpenAI implementation of LLM provider.
    """

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=settings.openai_api_key,
        )

    def generate(
        self,
        prompt: str,
    ) -> str:
        if not prompt.strip():
            raise ValueError(
                "prompt cannot be blank"
            )

        response = self.client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        if not response.choices:
            raise ValueError(
                "LLM returned no choices"
            )

        content = response.choices[0].message.content

        if content is None or not content.strip():
            raise ValueError(
                "LLM returned empty content"
            )

        return content