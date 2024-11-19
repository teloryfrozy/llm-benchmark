"""
Modules contains utility functions for prompt generation and connectivity checking
"""

import os
import openai
import anthropic
import google.generativeai as genai
from mistralai import Mistral
from dotenv import load_dotenv
import requests


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
) -> dict:
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
        dict: The response of the prompt
    """
    if llm_provider == "openai":
        get_response_openai(prompt, model, role, temperature, top_p, max_tokens_output)
    elif llm_provider == "anthropic":
        get_response_anthropic(
            prompt, model, role, temperature, top_p, max_tokens_output
        )
    elif llm_provider == "gemini":
        get_response_gemini(prompt, model, role, temperature, top_p, max_tokens_output)
    elif llm_provider == "mistral":
        get_response_mistral(prompt, model, role, temperature, top_p, max_tokens_output)
    else:
        raise ValueError("Invalid LLM provider")


def get_response_mistral(
    prompt: str,
    model: str,
    role: str,
    temperature,
    top_p,
    max_tokens_output: int = None,
):
    client = Mistral(api_key=MISTRAL_API_KEY)
    response = client.chat.complete(
        model=model,
        messages=[{"role": role, "content": prompt}],
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
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": role, "content": prompt}],
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
    response = client.messages.create(
        model=model,
        messages=[{"role": role, "content": prompt}],
        max_tokens=max_tokens_output,
        top_p=top_p,
        temperature=temperature,
    )
    return response


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
        messages=[{"role": role, "content": prompt}],
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens_output,
            top_p=top_p,
            temperature=temperature,
        ),
    )
    return response
