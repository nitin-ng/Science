# OpenAI Chat completion
import os
import openai
import chainlit as cl
from chainlit.prompt import Prompt, PromptMessage
from chainlit.playground.providers import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ChatOpenAI Templates
system_template = """You are a highly analytical assistant who has in-depth knowledge of science! 
"""

user_template = """{input}
Think through your response step by step.
Be concise in your responses
Do not make up stuff. If you don't know something, let the user know.
"""

@cl.on_chat_start
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,  # Adjust to a valid range
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    cl.user_session.set("settings", settings)

@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")

    # Set up the OpenAI client with the API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = Prompt(
        provider=ChatOpenAI.id,
        messages=[
            PromptMessage(
                role="system",
                template=system_template,
                formatted=system_template,
            ),
            PromptMessage(
                role="user",
                template=user_template,
                formatted=user_template.format(input=message.content),
            ),
        ],
        inputs={"input": message.content},
        settings=settings,
    )

    msg = cl.Message(content="")

    # Call OpenAI and stream response
    async for stream_resp in await openai.ChatCompletion.acreate(
        model=settings['model'],
        messages=[m.to_openai() for m in prompt.messages],
        stream=True,
        **settings
    ):
        token = stream_resp.choices[0].delta.get("content", "")
        await msg.stream_token(token)

    # Update the prompt object with the completion
    prompt.completion = msg.content
    msg.prompt = prompt

    # Send and close the message stream
    await msg.send()
