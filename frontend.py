from contextlib import contextmanager
import gradio as gr
from llm import LLM
from chat_buffer import chat_history
from typing import Tuple
from schemas import Message

model = LLM("meta-llama/Meta-Llama-3.1-8B-Instruct")  # TODO: implement model handler


def response(message, history) -> Tuple[str, list]:
    # Own chat history
    chat_history.append(Message(role="user", content=message))
    response = model.generate(chat_history.to_model_input())
    chat_history.append(Message(role="assistant", content=response))
    # UI chat history
    history.append([message, response])
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
