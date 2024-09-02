from collections import deque
from typing import Deque
from schemas import Message


class ChatBuffer:
    def __init__(self) -> None:
        self.history: Deque["Message"] = deque(maxlen=4)

    def append(self, message: Message) -> None:
        self.history.append(Message)

    def to_list(self) -> list[str]:
        return [dict(m)["content"] for m in list(self.history)]

    def to_model_input(self) -> list[dict[str, str]]:
        system = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate",
            }
        ]
        return system + [dict(m) for m in list(self.history)]


chat_history = ChatBuffer()
