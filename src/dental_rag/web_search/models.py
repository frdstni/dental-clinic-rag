from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """
    Single web search result.
    """

    title: str

    content: str

    url: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.title,
            str,
        ):
            raise TypeError(
                "title must be a string",
            )

        if not isinstance(
            self.content,
            str,
        ):
            raise TypeError(
                "content must be a string",
            )

        if not isinstance(
            self.url,
            str,
        ):
            raise TypeError(
                "url must be a string",
            )

        if not self.title.strip():
            raise ValueError(
                "title cannot be blank",
            )

        if not self.content.strip():
            raise ValueError(
                "content cannot be blank",
            )

        if not self.url.strip():
            raise ValueError(
                "url cannot be blank",
            )