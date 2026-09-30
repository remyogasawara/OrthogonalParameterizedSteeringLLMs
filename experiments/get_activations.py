#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------
# nvidia-smi
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Make relative dataset/output paths resolve from the project root,
# regardless of where this script is launched from.
os.chdir(PROJECT_ROOT)

from huggingface_hub import HfApi, hf_hub_download, login

from src.dataset import DataSet
from src.steerable_model import SteerableModel
from src.utils import set_global_seed
from estimators.steering_only_estimators import sample_diff_of_means


# ---------------------------------------------------------------------
# Defaults for this experiment
# ---------------------------------------------------------------------

DEFAULT_MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"

DEFAULT_DATASET_SUBFOLDER = "tan_paper_datasets/mwe/xrisk"

DEFAULT_BEHAVIORS = [
    "coordinate-other-ais",
    "corrigible-neutral-HHH",
    "myopic-reward",
    "survival-instinct",
    "power-seeking-inclination",
    "wealth-seeking-inclination",
]

DEFAULT_LAYER = 13  # Zero-indexed: this is the 14th transformer block.
DEFAULT_TEST_SIZE = 200
DEFAULT_BATCH_SIZE = 16


# ---------------------------------------------------------------------
# Hugging Face authentication
# ---------------------------------------------------------------------

def authenticate_huggingface(model_path: str) -> None:
    """
    Authenticate with Hugging Face and verify access to model_path.

    Authentication is taken from one of the following:

    1. The HF_TOKEN environment variable.
    2. A token previously saved with `hf auth login`.

    The token is never passed through a command-line argument and is never
    printed.
    """
    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        # Register the environment-provided token with huggingface_hub.
        # This makes it available to downstream from_pretrained() calls.
        login(
            token=hf_token,
            add_to_git_credential=False,
        )
        credential_source = "HF_TOKEN environment variable"
    else:
        # huggingface_hub will look for the token saved by `hf auth login`.
        credential_source = "cached Hugging Face login"

    try:
        api = HfApi()

        user_info = api.whoami(token=hf_token)
        username = user_info.get("name", "unknown")

        # Downloading config.json is a small, inexpensive way to verify that
        # the token has access to this gated model before loading all weights.
        hf_hub_download(
            repo_id=model_path,
            filename="config.json",
            token=hf_token,
        )

    except Exception as exc:
        raise RuntimeError(
            f"\nCould not access the Hugging Face model:\n"
            f"    {model_path}\n\n"
            "Make sure that:\n"
            "  1. Your Hugging Face account has been granted access to the "
            "Llama 2 repository.\n"
            "  2. You ran `hf auth login` on the cluster, or set HF_TOKEN.\n"
            "  3. The token belongs to the account that has Llama 2 access.\n"
        ) from exc

    print(
        f"Authenticated with Hugging Face as '{username}' "
        f"using {credential_source}."
    )
    print(f"Verified access to {model_path}.")


# ---------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------

def get_activations(
    model_path: str = DEFAULT_MODEL_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    behaviors: list[str] | None = None,
    layer: int = DEFAULT_LAYER,
    save_name: str | None = None,
    test_size: int = DEFAULT_TEST_SIZE,
    dataset_subfolder: str = DEFAULT_DATASET_SUBFOLDER,
    authenticate: bool = True,
) -> str:
    """
    Compute activations and difference-of-means steering vectors.

    Parameters
    ----------
    model_path:
        Hugging Face model repository or local model checkpoint.

    batch_size:
        Batch size used during activation extraction.

    behaviors:
        Behavior names to load from the dataset. Uses DEFAULT_BEHAVIORS
        when omitted.

    layer:
        Zero-indexed transformer layer.

        For example:
            layer=12 -> 13th transformer block
            layer=13 -> 14th transformer block

    save_name:
        Name passed to Activations.save(). The resulting file should be
        activations/{save_name}.pkl.

    test_size:
        Number of dataset examples reserved for the test split.

    dataset_subfolder:
        Dataset subfolder passed to DataSet.

    authenticate:
        Whether to verify Hugging Face authentication before loading.

    Returns
    -------
    str
        The saved activation name, without the .pkl extension.
    """
    set_global_seed(42)

    selected_behaviors = (
        list(DEFAULT_BEHAVIORS)
        if behaviors is None
        else list(behaviors)
    )

    if not selected_behaviors:
        raise ValueError("At least one behavior must be provided.")

    if layer < 0:
        raise ValueError(f"Layer must be nonnegative; received {layer}.")

    if batch_size <= 0:
        raise ValueError(
            f"Batch size must be positive; received {batch_size}."
        )

    if test_size <= 0:
        raise ValueError(
            f"Test size must be positive; received {test_size}."
        )

    if authenticate:
        authenticate_huggingface(model_path)

    # A concise default that identifies the model, one-indexed layer,
    # dataset, and number of included behaviors.
    if save_name is None:
        model_name = model_path.rstrip("/").split("/")[-1]
        dataset_name = Path(dataset_subfolder).name

        save_name = (
            f"{model_name}"
            f"_layer{layer + 1}"
            f"_{dataset_name}"
            f"_{len(selected_behaviors)}behaviors"
        )

    print()
    print("Activation extraction configuration")
    print("-----------------------------------")
    print(f"Model:             {model_path}")
    print(f"Dataset subfolder: {dataset_subfolder}")
    print(f"Layer index:       {layer}")
    print(f"Layer number:      {layer + 1}")
    print(f"Batch size:        {batch_size}")
    print(f"Test size:         {test_size}")
    print(f"Save name:         {save_name}")
    print("Behaviors:")

    for behavior in selected_behaviors:
        print(f"  - {behavior}")

    print()

    # DataSet performs the train/test split. Activations are computed on
    # train_data, while the reserved test_data can be used by later
    # evaluation scripts.
    dataset = DataSet(
        subfolders=[dataset_subfolder],
        test_size=test_size,
    )

    print(
        f"Loaded dataset with {len(dataset.train_data)} training examples "
        f"and {len(dataset.test_data)} test examples."
    )

    print(f"Loading model: {model_path}")

    model = SteerableModel(
        model_name=model_path,
    )

    print("Successfully loaded model.")

    # Answers have forms such as "(A)" and "(B)". In this dataset/tokenizer
    # setup, token position -2 corresponds to the answer token.
    behavior_token_mapping = {
        behavior: -2
        for behavior in selected_behaviors
    }

    start_time = time.time()

    activations = model.get_binary_activations_on_dataset(
        dataset.train_data,
        layers=[layer],
        token_positions=["answer_token"],
        batch_size=batch_size,
        behaviors=selected_behaviors,
        behavior_token_mapping=behavior_token_mapping,
    )

    activation_time = time.time() - start_time

    print(
        f"Successfully computed activations in "
        f"{activation_time:.2f} seconds."
    )

    estimators = {
        "sample_diff_of_means": sample_diff_of_means,
    }

    steering_start = time.time()

    # get_steering_vecs stores the resulting vectors on the Activations
    # object, which is then saved below.
    activations.get_steering_vecs(
        estimators,
        selected_behaviors,
    )

    steering_time = time.time() - steering_start

    print(
        f"Successfully computed steering vectors in "
        f"{steering_time:.2f} seconds."
    )

    activation_directory = PROJECT_ROOT / "activations"
    activation_directory.mkdir(parents=True, exist_ok=True)

    activations.save(name=save_name)

    expected_path = activation_directory / f"{save_name}.pkl"

    print(f"Successfully saved activations and steering vectors.")
    print(f"Expected output path: {expected_path}")

    return save_name


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and save activations and difference-of-means "
            "steering vectors."
        )
    )

    # nargs="?" makes the model optional. Running the script without a
    # positional model uses the Llama 2 default.
    parser.add_argument(
        "model_path",
        nargs="?",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Hugging Face model path or local checkpoint. "
            f"Default: {DEFAULT_MODEL_PATH}"
        ),
    )

    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Activation extraction batch size. Default: {DEFAULT_BATCH_SIZE}",
    )

    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=None,
        help=(
            "Space-separated behavior names. Uses the six xrisk behaviors "
            "when omitted."
        ),
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help=(
            "Zero-indexed transformer layer. "
            f"Default: {DEFAULT_LAYER}, which is transformer block "
            f"{DEFAULT_LAYER + 1}."
        ),
    )

    parser.add_argument(
        "--save-name",
        "--save_name",
        dest="save_name",
        type=str,
        default=None,
        help=(
            "Name for the saved activation file, without .pkl. "
            "A name is generated automatically when omitted."
        ),
    )

    parser.add_argument(
        "--test-size",
        "--test_size",
        dest="test_size",
        type=int,
        default=DEFAULT_TEST_SIZE,
        help=f"Number of held-out test examples. Default: {DEFAULT_TEST_SIZE}",
    )

    parser.add_argument(
        "--dataset-subfolder",
        "--dataset_subfolder",
        dest="dataset_subfolder",
        type=str,
        default=DEFAULT_DATASET_SUBFOLDER,
        help=(
            "Dataset subfolder passed to DataSet. "
            f"Default: {DEFAULT_DATASET_SUBFOLDER}"
        ),
    )

    parser.add_argument(
        "--skip-auth-check",
        action="store_true",
        help=(
            "Skip the explicit Hugging Face login/access check. "
            "Model loading may still require authentication."
        ),
    )

    parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip execution.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.skip:
        print("Skipping execution as requested by --skip.")
        raise SystemExit(0)

    get_activations(
        model_path=args.model_path,
        batch_size=args.batch_size,
        behaviors=args.behaviors,
        layer=args.layer,
        save_name=args.save_name,
        test_size=args.test_size,
        dataset_subfolder=args.dataset_subfolder,
        authenticate=not args.skip_auth_check,
    )