"""
Assert the validity of provided API keys
"""

from colorama import Fore, init
from utils import check_connectivity


init(autoreset=True)

SUPPORTED_LLM_PROVIDERS = ["openai", "anthropic", "gemini", "mistral"]


def test_api_keys(llm_providers: list[str]):
    """Assert the validity of provided API keys"""
    for provider in llm_providers:
        print(f"{Fore.CYAN}Checking connectivity for {provider}")
        assert check_connectivity(provider)
        print(f"{Fore.GREEN}Connectivity check for {provider} passed")


def test_llm_providers(llm_providers: list[str]):
    assert all(provider in SUPPORTED_LLM_PROVIDERS for provider in llm_providers)
    print(f"{Fore.GREEN}All LLM providers are valid")
