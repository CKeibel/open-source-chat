from transformers import AutoModelForCausalLm, AutoTokenizer

class LLM:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        