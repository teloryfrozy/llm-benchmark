"""
Starts the benchmarking process
"""

from colorama import Fore, init
from core.prompt_text import start_benchmark
from utils.test import test_api_keys, test_llm_providers, test_roles


init(autoreset=True)

# Define the models with the roles you want to compare
LLM_MODELS = {
    "openai": ["gpt-4o-mini"],
    "anthropic": ["claude-3-5-haiku-latest"],
    "gemini": ["gemini-1.5-flash"],
    "mistral": ["mistral-small-latest"],
}
ROLES = ["user", "assistant", "system"]
VERBOSE = False


if __name__ == "__main__":
    print("-------------------- LLM Providers Check --------------------")
    test_llm_providers(LLM_MODELS.keys())

    print("-------------------- API Keys Status Check --------------------")
    test_api_keys(LLM_MODELS.keys())

    print("-------------------- Roles Check --------------------")
    test_roles(ROLES)

    print(f"{Fore.GREEN}All checks passed successfully!")
    start_benchmark(VERBOSE, LLM_MODELS, ROLES)
