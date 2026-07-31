"""Editable experiment settings. There is deliberately no CLI layer."""

from pathlib import Path

from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


MODEL_NAME = "Qwen/Qwen3-4B"
TARGET_LAYER = 18  # zero-indexed decoder layer
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

N_BASE_SEQUENCES = 150
SEQUENCE_LENGTH = 1_024
TRAIN_FRACTION = 0.8
REGULARIZATION_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
STEERING_COEFFICIENTS = (-3.0, -2.0, -1.0, -0.5, 0.0)
MAX_NEW_TOKENS = 256
SEED = 42

# This must resolve to an independent hosted or separately loaded Inspect model.
# The default uses OpenRouter's free variant, authenticated by OPENROUTER_API_KEY.
JUDGE_MODEL = "openrouter/nvidia/nemotron-nano-9b-v2:free"
COHERENCE_SAMPLE_LIMIT = 24
SYCOPHANCY_SAMPLE_LIMIT = 100

# Local scoring-phase judge resilience. This is the number of retries after the
# initial request for transient model API failures (including HTTP 429 and 5xx).
MODEL_API_MAX_RETRIES = 7
MODEL_API_TIMEOUT = None

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
