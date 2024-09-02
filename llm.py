from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os

load_dotenv()

access_token = os.getenv("HF_SECRET")


class LLM:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", token=access_token, trust_remote_code=True
        )

    def generate(self, message: str) -> str:
        return "Hello"
