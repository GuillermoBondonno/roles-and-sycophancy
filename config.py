"""Editable experiment settings. There is deliberately no CLI layer."""

from pathlib import Path


MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TARGET_LAYER = 12  # zero-indexed decoder layer
ROLES = ("user", "assistant", "system")

# The usual decoder layouts are discovered automatically. For an unusual or
# remote-code architecture, provide a dotted path such as
# "transformer.blocks.{layer}.feed_forward".
STEERING_MODULE_PATH = None

# Passed through to the corresponding Transformers APIs. These make custom
# tokenizers, remote-code models, quantization, and model-specific chat-template
# options configurable without architecture branches in the experiment code.
TOKENIZER_LOAD_KWARGS = {"use_fast": True}
MODEL_LOAD_KWARGS = {}
CHAT_TEMPLATE_KWARGS = {}

N_BASE_SEQUENCES = 250
SEQUENCE_LENGTH = 1_024
TRAIN_FRACTION = 0.8
REGULARIZATION_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
STEERING_COEFFICIENTS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
MAX_NEW_TOKENS = 256
SEED = 42

# This must resolve to an independent hosted or separately loaded Inspect model.
JUDGE_MODEL = "openai/gpt-5-mini"
COHERENCE_SAMPLE_LIMIT = 24
SYCOPHANCY_SAMPLE_LIMIT = 100

# Explicit, deterministic deviation from the paper: cap the token-level probe
# examples per base sequence and role so the CPU logistic regression stays lean.
MAX_PROBE_TOKENS_PER_BASE_ROLE = 128

# Generation is kept greedy across every coefficient.
TEMPERATURE = 0.0
TOP_P = 1.0

# Optional scientific controls. The random control uses the same scale and a
# fixed-seed unit vector; it is intentionally off for the primary experiment.
RUN_RANDOM_DIRECTION_CONTROL = False
RANDOM_CONTROL_SAMPLE_LIMIT = 10

ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR = Path("results")
COHERENCE_DATASET = Path("data/coherence_prompts.jsonl")
