"""
Starts the benchmarking process
"""

from colorama import Fore, init
from tests.test import test_api_keys, test_llm_providers


init(autoreset=True)

# Define here the models you want to compare
LLM_MODELS = {
    "openai": ["gpt-4o-mini"],
    "anthropic": ["claude-3-5-haiku-latest"],
    "gemini": ["gemini-1.5-flash"],
    "mistral": ["mistral-small-latest"],
}


if __name__ == "__main__":
    print(f"{Fore.BLUE}Checking the validity of the LLM providers")
    test_llm_providers(LLM_MODELS.keys())
    print(f"{Fore.BLUE}Checking connectivity for all providers")
    test_api_keys(LLM_MODELS.keys())
