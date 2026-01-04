import argparse
import json
import os
from typing import Callable, Iterable

import pandas as pd

from . import common
from .mmlu_eval import MMLUEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.codex_cli_sampler import CodexCLISampler
from .sampler.gemini_cli_sampler import GeminiCLISampler
from .sampler.o_chat_completion_sampler import OChatCompletionSampler


def _build_samplers() -> dict[str, Callable[[], ChatCompletionSampler | OChatCompletionSampler | GeminiCLISampler | CodexCLISampler]]:
    """
    Define all available samplers as factory functions.
    Samplers are only initialized when actually requested.
    """
    return {
        "gpt-4o_chatgpt": lambda: ChatCompletionSampler(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            max_tokens=2048,
        ),
        "gpt-4o-mini-2024-07-18": lambda: ChatCompletionSampler(
            model="gpt-4o-mini-2024-07-18",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "o1-preview": lambda: OChatCompletionSampler(
            model="o1-preview",
        ),
        "o1-mini": lambda: OChatCompletionSampler(
            model="o1-mini",
        ),
        # Default == Medium
        "o3-mini": lambda: OChatCompletionSampler(
            model="o3-mini",
        ),
        "o3-mini_high": lambda: OChatCompletionSampler(
            model="o3-mini",
            reasoning_effort="high",
        ),
        "o3-mini_low": lambda: OChatCompletionSampler(
            model="o3-mini",
            reasoning_effort="low",
        ),
        # Gemini CLI models
        "gemini-2.5-flash-cli": lambda: GeminiCLISampler(
            model="gemini-2.5-flash",
        ),
        "gemini-2.5-pro-cli": lambda: GeminiCLISampler(
            model="gemini-2.5-pro",
        ),
        # Codex CLI models
        "codex-cli": lambda: CodexCLISampler(),
        "codex-gpt-5.1-codex-max": lambda: CodexCLISampler(
            model="gpt-5.1-codex-max",
        ),
        "codex-gpt-5.1-codex-mini": lambda: CodexCLISampler(
            model="gpt-5.1-codex-mini",
        ),
        "codex-gpt-5.1": lambda: CodexCLISampler(
            model="gpt-5.1",
        ),
    }


ALL_EVAL_NAMES = [
    "mmlu_AR-XY",
    "mmlu_BN-BD",
    "mmlu_DE-DE",
    "mmlu_EN-US",
    "mmlu_ES-LA",
    "mmlu_FR-FR",
    "mmlu_HI-IN",
    "mmlu_ID-ID",
    "mmlu_IT-IT",
    "mmlu_JA-JP",
    "mmlu_KO-KR",
    "mmlu_PT-BR",
    "mmlu_ZH-CN",
    "mmlu_SW-KE",
    "mmlu_YO-NG",
]


def _parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Run multilingual MMLU evals.")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Sampler name(s) to run. Defaults to all known samplers.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=None,
        help="Number of examples per language. Default is 10 in debug mode, or full set otherwise.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        dest="languages",
        help="Eval names to run (e.g., mmlu_EN-US). Defaults to all.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use a small default sample size (10 examples). Overridden by --examples if provided.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="/tmp",
        help="Directory to save results. Defaults to /tmp.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None):
    args = _parse_args(argv)

    num_examples = args.examples
    if num_examples is None:
        num_examples = 10 if args.debug else None

    result_dir = args.result_dir
    # Create result directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)

    sampler_factories = _build_samplers()
    
    # Filter to requested samplers, or use all if none specified
    requested_models = args.models if args.models else list(sampler_factories.keys())
    
    # Initialize only the requested samplers
    samplers = {}
    for name in requested_models:
        if name not in sampler_factories:
            raise ValueError(f"Unknown sampler: {name}. Available: {list(sampler_factories.keys())}")
        try:
            samplers[name] = sampler_factories[name]()
        except Exception as e:
            raise ValueError(
                f"Failed to initialize sampler '{name}': {e}\n"
                "This may be due to missing API keys or CLI tools. "
                "Check the error message above for details."
            ) from e
    
    if not samplers:
        raise ValueError(f"No samplers could be initialized from: {requested_models}")

    eval_names = args.languages if args.languages else ALL_EVAL_NAMES

    def get_evals(eval_name: str):
        match eval_name:
            case "mmlu_EN-US":
                return MMLUEval(num_examples=num_examples, language="EN-US")
            case "mmlu_AR-XY":
                return MMLUEval(num_examples=num_examples, language="AR-XY")
            case "mmlu_BN-BD":
                return MMLUEval(num_examples=num_examples, language="BN-BD")
            case "mmlu_DE-DE":
                return MMLUEval(num_examples=num_examples, language="DE-DE")
            case "mmlu_ES-LA":
                return MMLUEval(num_examples=num_examples, language="ES-LA")
            case "mmlu_FR-FR":
                return MMLUEval(num_examples=num_examples, language="FR-FR")
            case "mmlu_HI-IN":
                return MMLUEval(num_examples=num_examples, language="HI-IN")
            case "mmlu_ID-ID":
                return MMLUEval(num_examples=num_examples, language="ID-ID")
            case "mmlu_IT-IT":
                return MMLUEval(num_examples=num_examples, language="IT-IT")
            case "mmlu_JA-JP":
                return MMLUEval(num_examples=num_examples, language="JA-JP")
            case "mmlu_KO-KR":
                return MMLUEval(num_examples=num_examples, language="KO-KR")
            case "mmlu_PT-BR":
                return MMLUEval(num_examples=num_examples, language="PT-BR")
            case "mmlu_ZH-CN":
                return MMLUEval(num_examples=num_examples, language="ZH-CN")
            case "mmlu_SW-KE":
                return MMLUEval(num_examples=num_examples, language="SW-KE")
            case "mmlu_YO-NG":
                return MMLUEval(num_examples=num_examples, language="YO-NG")
            case _:
                raise Exception(f"Unrecoginized eval type: {eval_name}")

    evals = {eval_name: get_evals(eval_name) for eval_name in eval_names}
    print(f"Running evals: {list(evals.keys())}")
    print(f"Running samplers: {list(samplers.keys())}")
    debug_suffix = "_DEBUG" if num_examples is not None and num_examples <= 10 else ""

    # Use a unique separator that won't appear in eval or sampler names
    SEPARATOR = ":::"
    mergekey2resultpath = {}
    for sampler_name, sampler in samplers.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{sampler_name}"
            report_filename = f"{result_dir}/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"{result_dir}/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            # Store with separator for reliable parsing
            mergekey2resultpath[f"{eval_name}{SEPARATOR}{sampler_name}"] = result_filename
    merge_metrics = []
    for eval_sampler_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        # Split on unique separator to correctly extract eval_name and sampler_name
        if SEPARATOR in eval_sampler_name:
            eval_name, sampler_name = eval_sampler_name.split(SEPARATOR, 1)
        else:
            # Fallback for backward compatibility (old format with underscore)
            eval_name = eval_sampler_name[: eval_sampler_name.find("_")]
            sampler_name = eval_sampler_name[eval_sampler_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "sampler_name": sampler_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["sampler_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
