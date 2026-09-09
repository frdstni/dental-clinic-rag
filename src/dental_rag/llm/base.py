from typing import Protocol


class LLM(Protocol):
    """
    Interface for language model providers.
    """

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate text from a prompt.
        """
        ...