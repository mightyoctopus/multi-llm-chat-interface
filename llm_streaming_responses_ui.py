import os
import requests
from bs4 import BeautifulSoup
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
from anthropic import Anthropic
import gradio as gr


load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
google.generativeai.configure(api_key=os.getenv("GEMINI_API_KEY"))

system_message = "You are a helpful assistant that responds in markdown"

### Normal text generation
def message_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    completion = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    return completion.choices[0].message.content


### Streaming text generation
def stream_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    stream = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        stream=True
    )

    result = ""

    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result


def stream_claude(prompt):

    stream = anthropic.messages.stream(
        model="claude-3-5-haiku-latest",
        max_tokens=300,
        temperature=0.7,
        system=system_message,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response = ""
    with stream as claude_stream:
        for chunk in claude_stream.text_stream:
            response += chunk or ""
            yield response

def stream_gemini(prompt):

    gemini_model = google.generativeai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_message
    )
    stream = gemini_model.generate_content(
        contents=prompt,
        stream=True
    )

    result = ""

    for chunk in stream:
        result += chunk.text
        yield result



def select_model(prompt, model):
    if model == "GPT":
        result = stream_gpt(prompt)
    elif model == "Claude":
        result = stream_claude(prompt)
    elif model == "Gemini":
        result = stream_gemini(prompt)
    else:
        raise ValueError("unknown model")
    for chunk in result:
        yield chunk




###===================== GRADIO UI ==========================###

view = gr.Interface(
    fn=select_model,
    inputs=[
        gr.Textbox(label="Your Message"),
        gr.Dropdown(["GPT", "Claude", "Gemini"], label="Select a model")
    ],
    outputs=[gr.Markdown(label="Response:")],
    allow_flagging="never"
)
view.launch(share=True)












