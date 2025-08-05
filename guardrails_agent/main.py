
# 📦 Import necessary modules and dependencies

import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from pydantic import BaseModel
from dotenv import load_dotenv
import chainlit as cl
from agents import (
    input_guardrail, output_guardrail,
    RunContextWrapper,
    TResponseInputItem,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered
)

# 🔐 Load environment variables from .env file
load_dotenv()

# 🚫 Disable tracing for cleaner execution (optional)
set_tracing_disabled(disabled=True)

# 🔑 Get Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# ❗ Raise error if API key is missing
if not gemini_api_key:
    raise ValueError("🔑 GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# 🌐 Initialize Gemini async client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 🧠 Define the model configuration
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# ✅ Define output structure for input validation (Python-related or not)
class PythonRelatedOutput(BaseModel):
    is_python_related: bool
    reasoning: str

# 🤖 Agent to check if user's question is related to Python
input_guardrails_agent = Agent(
    name="Input Guardrails Checker",
    instructions="Check if the user's question is related to Python programming. "
                 "If it is, return true; if not, return false.",
    model=model,
    output_type=PythonRelatedOutput
)

# 🛡️ Input guardrail function
@input_guardrail
async def input_guardrails_func(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrails_agent, input)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_python_related  # 🚨 Tripwire triggers if not Python
    )

# 📤 Message output structure
class MessageOutput(BaseModel):
    response: str

# ✅ Output structure for Python relevance
class PythonOutput(BaseModel):
    is_Python: bool
    reasoning: str

# 🧠 Agent to validate Python-related output
output_guardrail_agent = Agent(
    name="Output Guardrails Checker",
    instructions="Check whether the output includes Python-related content.",
    model=model,
    output_type=PythonOutput
)

# 🛡️ Output guardrail function
@output_guardrail
async def output_python_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    output_result = await Runner.run(output_guardrail_agent, output)
    return GuardrailFunctionOutput(
        output_info=output_result.final_output,
        tripwire_triggered=not output_result.final_output.is_Python  # 🚨 Tripwire triggers if output isn't Python
    )

# 🧠 Main Python Expert Agent
main_agent = Agent(
    name="Python_Expert_Agent",
    instructions="You are a Python expert agent. You only respond to Python-related questions.",
    model=model,
    input_guardrails=[input_guardrails_func],        # 👈 Input is validated here
    output_guardrails=[output_python_guardrail]      # 👉 Output is validated here
)

# 💬 Handler for when a new chat starts
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="👋 Hello! I'm ready to assist you with Python programming.").send()

# 💬 Handler for receiving messages
@cl.on_message
async def on_message(message: cl.Message):
    try:
        # ▶️ Run the main agent with the input message
        result = await Runner.run(
            main_agent,
            input=message.content
        )

        # 📤 Send the agent's response
        await cl.Message(content=result.final_output).send()

    # 🚫 Handle input guardrail trip
    except InputGuardrailTripwireTriggered:
        await cl.Message(content="⚠️ Please ask questions related to Python programming only.").send()

    # 🚫 Handle output guardrail trip
    except OutputGuardrailTripwireTriggered:
        await cl.Message(content="⛔ Output was rejected by the guardrail. Try rephrasing your Python query.").send()
