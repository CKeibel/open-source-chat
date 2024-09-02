from collections import deque
from schemas import Message

class ChatBuffer:
    def __init__(self) -> None:
        self.history: list[Message]
        self.max_len = 4

    def append(self, message: Message) ->:
        pass