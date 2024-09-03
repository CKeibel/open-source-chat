from collections import deque
from typing import Deque
from schemas import Message


class ChatBuffer:
    def __init__(self) -> None:
        self.history: Deque["Message"] = deque(maxlen=4)

    def append(self, message: Message) -> None:
        self.history.append(message)

    def to_list(self) -> list[str]:
        return [dict(m)["content"] for m in self.history]

    def to_model_input(self) -> list[dict[str, str]]:
        system = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in a friendly manner.",
            }
        ]
        return system + [dict(m) for m in self.history]


chat_history = ChatBuffer()
