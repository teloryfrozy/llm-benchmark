"""
Benchmarking script for a prompt with different LLM models and roles for generative text tasks
"""

import statistics
from colorama import Fore, init
from utils.utils import get_prompt_response


init(autoreset=True)

# Define the prompt you want to benchmark here
PROMPT = "How to advertise a SaaS with a budget of $1000 in 3 key sentences?"


results = {}


def start_benchmark(VERBOSE: bool, LLM_MODELS: dict, ROLES: list[str]):
    print("-------------------- Settings --------------------")
    print(f"{Fore.CYAN}LLM Models: {LLM_MODELS}")
    print(f"{Fore.BLUE}Roles: {ROLES}")
    print(f"{Fore.YELLOW}Prompt: {PROMPT}")
    print("-------------------- Starting benchmarking --------------------")

    for llm_provider, models in LLM_MODELS.items():
        for model in models:
            for role in ROLES:
                response = get_prompt_response(PROMPT, llm_provider, model, role)
                save_response(response, llm_provider, model, role)
                print(
                    f"{Fore.LIGHTBLUE_EX}[{llm_provider}]({model})|{role}| > {response['elapsed_time']} seconds"
                )
                if VERBOSE:
                    print(f"{Fore.YELLOW}Content: {response['content']}")
                    print(f"{Fore.MAGENTA}Input Tokens: {response['input_tokens']}")
                    print(f"{Fore.CYAN}Output Tokens: {response['output_tokens']}")

    print_result(results)


def save_response(response, llm_provider, model, role):
    if llm_provider not in results:
        results[llm_provider] = {}
    if model not in results[llm_provider]:
        results[llm_provider][model] = {}
    if role not in results[llm_provider][model]:
        results[llm_provider][model][role] = response
    else:
        results[llm_provider][model][role] = response


def print_result(results: dict[str, dict[str, dict[str, dict[str, any]]]]):
    fastest_model_with_role = {
        "elapsed_time": float("inf"),
        "model": "",
        "llm_provider": "",
        "role": "",
    }
    slowest_model_with_role = {
        "elapsed_time": 0,
        "model": "",
        "llm_provider": "",
        "role": "",
    }
    fastest_model = {
        "elapsed_time": float("inf"),
        "model": "",
        "llm_provider": "",
    }
    slowest_model = {
        "elapsed_time": 0,
        "model": "",
        "llm_provider": "",
    }
    most_stable_model = {
        "std_elapsed_time": float("inf"),
        "model": "",
        "llm_provider": "",
    }
    most_unstable_model = {
        "std_elapsed_time": 0,
        "model": "",
        "llm_provider": "",
    }
    most_talkative_model_with_role = {
        "output_tokens": 0,
        "model": "",
        "llm_provider": "",
        "role": "",
    }
    least_talkative_model_with_role = {
        "output_tokens": float("inf"),
        "model": "",
        "llm_provider": "",
        "role": "",
    }
    most_talkative_model = {
        "output_tokens": 0,
        "model": "",
        "llm_provider": "",
    }
    least_talkative_model = {
        "output_tokens": float("inf"),
        "model": "",
        "llm_provider": "",
    }

    def update_summary():
        if min(roles_results) < fastest_model_with_role["elapsed_time"]:
            fastest_model_with_role["elapsed_time"] = min(roles_results)
            fastest_model_with_role["model"] = model
            fastest_model_with_role["llm_provider"] = llm_provider
            fastest_model_with_role["role"] = min(
                roles, key=lambda x: roles[x]["elapsed_time"]
            )
        if max(roles_results) > slowest_model_with_role["elapsed_time"]:
            slowest_model_with_role["elapsed_time"] = max(roles_results)
            slowest_model_with_role["model"] = model
            slowest_model_with_role["llm_provider"] = llm_provider
            slowest_model_with_role["role"] = max(
                roles, key=lambda x: roles[x]["elapsed_time"]
            )
        if min(roles_results) < fastest_model["elapsed_time"]:
            fastest_model["elapsed_time"] = min(roles_results)
            fastest_model["model"] = model
            fastest_model["llm_provider"] = llm_provider
        if max(roles_results) > slowest_model["elapsed_time"]:
            slowest_model["elapsed_time"] = max(roles_results)
            slowest_model["model"] = model
            slowest_model["llm_provider"] = llm_provider

        if (
            len(roles_results) > 1
            and statistics.stdev(roles_results) < most_stable_model["std_elapsed_time"]
        ):
            most_stable_model["std_elapsed_time"] = statistics.stdev(roles_results)
            most_stable_model["model"] = model
            most_stable_model["llm_provider"] = llm_provider
        if (
            len(roles_results) > 1
            and statistics.stdev(roles_results)
            > most_unstable_model["std_elapsed_time"]
        ):
            most_unstable_model["std_elapsed_time"] = statistics.stdev(roles_results)
            most_unstable_model["model"] = model
            most_unstable_model["llm_provider"] = llm_provider

        if min(roles_output_tokens) < least_talkative_model_with_role["output_tokens"]:
            least_talkative_model_with_role["output_tokens"] = min(roles_output_tokens)
            least_talkative_model_with_role["model"] = model
            least_talkative_model_with_role["llm_provider"] = llm_provider
            least_talkative_model_with_role["role"] = min(
                roles, key=lambda x: roles[x]["output_tokens"]
            )
        if max(roles_output_tokens) > most_talkative_model_with_role["output_tokens"]:
            most_talkative_model_with_role["output_tokens"] = max(roles_output_tokens)
            most_talkative_model_with_role["model"] = model
            most_talkative_model_with_role["llm_provider"] = llm_provider
            most_talkative_model_with_role["role"] = max(
                roles, key=lambda x: roles[x]["output_tokens"]
            )
        if min(roles_output_tokens) < least_talkative_model["output_tokens"]:
            least_talkative_model["output_tokens"] = min(roles_output_tokens)
            least_talkative_model["model"] = model
            least_talkative_model["llm_provider"] = llm_provider
        if max(roles_output_tokens) > most_talkative_model["output_tokens"]:
            most_talkative_model["output_tokens"] = max(roles_output_tokens)
            most_talkative_model["model"] = model
            most_talkative_model["llm_provider"] = llm_provider

    for llm_provider, models in results.items():
        for model, roles in models.items():
            roles_results = [response["elapsed_time"] for response in roles.values()]
            roles_output_tokens = [
                response["output_tokens"] for response in roles.values()
            ]

            update_summary()

            results[llm_provider][model]["avg_elapsed_time"] = statistics.mean(
                roles_results
            )
            results[llm_provider][model]["std_elapsed_time"] = (
                statistics.stdev(roles_results) if len(roles_results) > 1 else 0
            )
            results[llm_provider][model]["avg_output_tokens"] = statistics.mean(
                roles_results
            )
            results[llm_provider][model]["std_dev_output_tokens"] = (
                statistics.stdev(roles_results) if len(roles_results) > 1 else 0
            )

        models_results = [response["avg_elapsed_time"] for response in models.values()]
        results[llm_provider]["avg_elapsed_time"] = statistics.mean(models_results)
        results[llm_provider]["std_elapsed_time"] = (
            statistics.stdev(models_results) if len(models_results) > 1 else 0
        )
        results[llm_provider]["avg_output_tokens"] = statistics.mean(models_results)
        results[llm_provider]["std_dev_output_tokens"] = (
            statistics.stdev(models_results) if len(models_results) > 1 else 0
        )

    llms_results = [response["avg_elapsed_time"] for response in results.values()]
    results["avg_elapsed_time"] = statistics.mean(llms_results)
    results["std_elapsed_time"] = (
        statistics.stdev(llms_results) if len(llms_results) > 1 else 0
    )
    results["avg_output_tokens"] = statistics.mean(llms_results)
    results["std_dev_output_tokens"] = (
        statistics.stdev(llms_results) if len(llms_results) > 1 else 0
    )

    print("-------------------- Elapsed Time --------------------")
    print(
        f"{Fore.LIGHTYELLOW_EX}The fastest model is {fastest_model_with_role['model']} from {fastest_model_with_role['llm_provider']} with role {fastest_model_with_role['role']}: {fastest_model_with_role['elapsed_time']}s"
    )
    print(
        f"{Fore.LIGHTYELLOW_EX}The slowest model is {slowest_model_with_role['model']} from {slowest_model_with_role['llm_provider']} with role {slowest_model_with_role['role']}: {slowest_model_with_role['elapsed_time']}s"
    )
    print(
        f"{Fore.LIGHTYELLOW_EX}The fastest model is {fastest_model['model']} from {fastest_model['llm_provider']} with an average elapsed time of {fastest_model['elapsed_time']}s"
    )
    print(
        f"{Fore.LIGHTYELLOW_EX}The slowest model is {slowest_model['model']} from {slowest_model['llm_provider']} with an average elapsed time of {slowest_model['elapsed_time']}s"
    )
    print(
        f"{Fore.LIGHTYELLOW_EX}The most stable model is {most_stable_model['model']} from {most_stable_model['llm_provider']} with an average standard deviation of {most_stable_model['std_elapsed_time']}s"
    )
    print(
        f"{Fore.LIGHTYELLOW_EX}The most unstable model is {most_unstable_model['model']} from {most_unstable_model['llm_provider']} with an average standard deviation of {most_unstable_model['std_elapsed_time']}s"
    )
    print("-------------------- Output Tokens --------------------")
    print(
        f"{Fore.LIGHTMAGENTA_EX}The most talkative model is {most_talkative_model_with_role['model']} from {most_talkative_model_with_role['llm_provider']} with role {most_talkative_model_with_role['role']}: {most_talkative_model_with_role['output_tokens']} output tokens"
    )
    print(
        f"{Fore.LIGHTMAGENTA_EX}The least talkative model is {least_talkative_model_with_role['model']} from {least_talkative_model_with_role['llm_provider']} with role {least_talkative_model_with_role['role']}: {least_talkative_model_with_role['output_tokens']} output tokens"
    )
    print(
        f"{Fore.LIGHTMAGENTA_EX}The most talkative model is {most_talkative_model['model']} from {most_talkative_model['llm_provider']} with an average output tokens of {most_talkative_model['output_tokens']}"
    )
    print(
        f"{Fore.LIGHTMAGENTA_EX}The least talkative model is {least_talkative_model['model']} from {least_talkative_model['llm_provider']} with an average output tokens of {least_talkative_model['output_tokens']}"
    )
