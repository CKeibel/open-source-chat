from contextlib import contextmanager
import gradio as gr
from llm import LLM
from typing import Tuple


def response(message, history) -> Tuple[str, list]:
    model = LLM("meta-llama/Meta-Llama-3.1-8B-Instruct")
    res = model.generate(message)
    history.append((message, "response"))
    return "", history


@contextmanager
def gradio_frontend():
    with gr.Blocks() as frontend:
        # Chat Tab
        with gr.Tab("Chat") as chat_tab:
            chatbot = gr.Chatbot(
                height=500,
            )

            msg = gr.Textbox(label="User input", placeholder="Type your message here")
            msg.submit(response, inputs=[msg, chatbot], outputs=[msg, chatbot])
            _ = gr.ClearButton([msg, chatbot])

        # Settings Tab
        with gr.Tab("Settings"):
            pass

    yield frontend


