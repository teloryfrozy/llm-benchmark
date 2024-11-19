"""
Modules contains utility functions for prompt generation and connectivity checking
"""

import requests
import time
import os
import openai
import anthropic
import google.generativeai as genai
from mistralai import ChatCompletionResponse, Mistral
from dotenv import load_dotenv


load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


######################## CONNECTIVITY CHECKING ########################
def check_connectivity(llm_provider: str) -> bool:
    """Checks the connectivity of the LLM provider

    Args:
        llm_provider (str): The LLM provider

    Raises:
        ValueError: If the LLM provider is invalid

    Returns:
        bool: The connectivity status of the LLM provider
    """
    if llm_provider == "openai":
        return check_connectivity_openai()
    elif llm_provider == "anthropic":
        return check_connectivity_anthropic()
    elif llm_provider == "gemini":
        return check_connectivity_gemini()
    elif llm_provider == "mistral":
        return check_connectivity_mistral()
    else:
        raise ValueError("Invalid LLM provider")


def check_connectivity_openai() -> bool:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True


def check_connectivity_anthropic() -> bool:
    try:
        client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
        client.messages.create(
            model="claude-3-5-haiku-latest",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1,
        )
    except anthropic._exceptions.AuthenticationError:
        return False
    else:
        return True


def check_connectivity_gemini() -> bool:
    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        gemini_model.generate_content(
            "Test",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1,
            ),
        )
        return True
    except Exception:
        return False


def check_connectivity_mistral() -> bool:
    try:
        response = requests.get(
            "https://api.mistral.ai/v1/models",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
        )
        if response.status_code == 401:
            return False
        response.raise_for_status()
    except Exception:
        return False
    else:
        return True


######################## RESPONSE GENERATION ########################
def get_prompt_response(
    prompt: str,
    llm_provider: str,
    model: str,
    role: str,
    temperature: float = None,
    top_p: float = None,
    max_tokens_output: int = None,
):
    """Returns the prompt response according to the LLM provider

    Args:
        prompt (str): The prompt message
        llm_provider (str): The LLM provider
        model (str): The model name
        role (str): The role of the message
        max_tokens_output (int): The maximum number of tokens to generate

    Raises:
        ValueError: If the LLM provider is invalid

    Returns:
        dict:
            - input_tokens: The number of input tokens
            - output_tokens: The number of output tokens
            - content: The generated content
            - elapsed_time: The elapsed time for the response generation
    """
    if llm_provider == "openai":
        start = time.time()
        response = get_response_openai(
            prompt, model, role, temperature, top_p, max_tokens_output
        )
        return {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "content": response.choices[0].message.content,
            "elapsed_time": time.time() - start,
        }
    elif llm_provider == "anthropic":
        start = time.time()
        response = get_response_anthropic(
            prompt, model, role, temperature, top_p, max_tokens_output
        )
        return {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "content": response.content[0].text if response.content else "",
            "elapsed_time": time.time() - start,
        }
    elif llm_provider == "gemini":
        start = time.time()
        response = get_response_gemini(
            prompt, model, role, temperature, top_p, max_tokens_output
        )
        return {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.total_token_count
            - response.usage_metadata.prompt_token_count,
            "content": response.text,
            "elapsed_time": time.time() - start,
        }
    elif llm_provider == "mistral":
        start = time.time()
        response = get_response_mistral(
            prompt, model, role, temperature, top_p, max_tokens_output
        )
        return {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "content": response.choices[0].message.content,
            "elapsed_time": time.time() - start,
        }
    else:
        raise ValueError("Invalid LLM provider")


def construct_messages(prompt: str, role: str, system_message: str = None):
    """
    Constructs a messages array for LLM APIs based on the role and optional system message.

    Args:
        prompt (str): The user's prompt.
        role (str): Role for the message.
        system_message (str): Optional system message for context.

    Returns:
        list: List of messages suitable for the target API.
    """
    messages = []
    if system_message and role in ["assistant", "system"]:
        messages.append({"role": "system", "content": system_message})
    if role in ["user", "assistant"]:
        messages.append({"role": role, "content": prompt})

    return messages


def construct_messages_for_mistral(prompt: str, role: str, system_message: str = None):
    """
    Constructs a messages array specifically for Mistral API.

    Args:
        prompt (str): The user's prompt.
        role (str): Role for the message.
        system_message (str): Optional system message for context.

    Returns:
        list: List of messages suitable for the Mistral API.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    if role == "assistant":
        # Prefix required for Assistant responses
        messages.append({"role": "assistant", "prefix": True, "content": prompt})
    elif role in ["user", "tool"]:
        messages.append({"role": role, "content": prompt})
    else:
        raise ValueError(f"Unsupported role configuration for Mistral: {role}")

    return messages


def get_response_mistral(
    prompt: str,
    model: str,
    role: str,
    temperature: float = None,
    top_p: float = None,
    max_tokens_output: int = None,
) -> ChatCompletionResponse:
    client = Mistral(api_key=MISTRAL_API_KEY)
    messages = construct_messages_for_mistral(
        prompt, role, system_message="You are a helpful assistant."
    )

    response = client.chat.complete(
        model=model,
        messages=messages,
        max_tokens=max_tokens_output,
        top_p=top_p,
        temperature=temperature,
    )
    return response


def get_response_openai(
    prompt: str,
    model: str,
    role: str,
    temperature: float = None,
    top_p: float = None,
    max_tokens_output: int = None,
):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    messages = construct_messages(
        prompt, role, system_message="You are a helpful assistant."
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens_output,
        top_p=top_p,
        temperature=temperature,
    )
    return response


def get_response_anthropic(
    prompt: str,
    model: str,
    role: str,
    temperature: float = None,
    top_p: float = None,
    max_tokens_output: int = None,
):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system_param = anthropic.NotGiven()
    messages_param = []

    if role == "assistant" or role == "system":
        system_param = "You are a helpful assistant."
        messages_param = [{"role": "user", "content": prompt}]
    elif role == "user":
        messages_param = [{"role": "user", "content": prompt}]

    try:
        response = client.messages.create(
            model=model,
            system=system_param,
            messages=messages_param,
            max_tokens=max_tokens_output or 4096,
            top_p=top_p or anthropic.NotGiven(),
            temperature=temperature or anthropic.NotGiven(),
        )
        return response

    except anthropic.BadRequestError as e:
        raise RuntimeError(f"Anthropic API error: {e}")


def get_response_gemini(
    prompt: str,
    model: str,
    role: str,
    temperature: float = None,
    top_p: float = None,
    max_tokens_output: int = None,
):
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens_output,
            top_p=top_p,
            temperature=temperature,
        ),
    )
    return response
