from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from dotenv import load_dotenv
import os

load_dotenv()

access_token = os.getenv("HF_SECRET")

chat_template = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'user' %}\n"
    "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n"
    "{% endif %}\n"
    "{% endfor %}"
)


class LLM:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.chat_template = chat_template
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            token=access_token,
            trust_remote_code=True,
        )
        self.gen_cfg = GenerationConfig.from_pretrained(self.model_id)
        self.gen_cfg.max_new_tokens = 250
        self.gen_cfg.num_beams = 3
        self.gen_cfg.temperature = 1.4
        self.gen_cfg.top_k = 85
        self.gen_cfg.pad_token_id = self.tokenizer.pad_token_id
        self.gen_cfg.begin_suppress_tokensrepetition_penalty = 5
        self.gen_cfg.no_repeat_ngram_size = 3

    @torch.no_grad()
    def generate(self, inputs: str, **kwargs) -> str:
        """Function to generate the answer for the given input text.
        Args:
            input (str): The input text for which the answer needs to be generated.
        Returns:
            str: The generated answer.
        """
        messages = self.tokenizer.apply_chat_template(
            inputs, tokenize=False, add_generation_prompt=True
        )
        inputs_ids = self.tokenizer(messages, return_tensors="pt").input_ids.to(
            self.model.device
        )
        outputs = self.model.generate(
            inputs_ids, self.gen_cfg, **kwargs
        )  # TODO: generation config
        # decode only new tokens to string
        answer = self.tokenizer.decode(
            outputs[0][len(inputs_ids[0]) :], skip_special_tokens=True
        )
        torch.cuda.empty_cache()
        return answer
