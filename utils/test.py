"""
Assert the validity of provided API keys
"""

from colorama import Fore, init
from utils.constants import SUPPORTED_LLM_PROVIDERS, SUPPORTED_ROLES
from utils.utils import check_connectivity


init(autoreset=True)


def test_api_keys(llm_providers: list[str]):
    """Assert the validity of provided API keys"""
    for provider in llm_providers:
        print(f"{Fore.CYAN}Checking connectivity for {provider}")
        assert check_connectivity(provider)
        print(f"{Fore.GREEN}Connectivity check for {provider} passed")


def test_llm_providers(llm_providers: list[str]):
    assert len(llm_providers) > 0, "At least one LLM provider is required"
    assert all(provider in SUPPORTED_LLM_PROVIDERS for provider in llm_providers), f"Supported providers are {SUPPORTED_LLM_PROVIDERS}"
    print(f"{Fore.GREEN}All LLM providers are valid")


def test_roles(roles: list[str]):
    assert all(role in SUPPORTED_ROLES for role in roles), f"Supported roles are {SUPPORTED_ROLES}"
    print(f"{Fore.GREEN}All roles are valid")
