import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

IN_COLAB = "google.colab" in sys.modules

LAB_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    Path("/content/datasets/cnn_dailymail")
    if IN_COLAB
    else LAB_DIR.parent / "datasets" / "cnn_dailymail"
)
RAW_DIR = DATA_ROOT / "3.0.0"
CACHE_DIR = DATA_ROOT / "cache"
OUT_DIR = LAB_DIR / "outputs"
CKPT_DIR = LAB_DIR / "checkpoints"
GEN_DIR = OUT_DIR / "generations"

HF_DATASET = "abisee/cnn_dailymail"
SPLITS = ["train", "validation", "test"]

SEED = 42
TEST_SAMPLE = 1000        # общая тестовая выборка для всех моделей
VAL_SAMPLE = 1000         # валидация RoBERTa (выбор эпохи и lr)
PROMPT_VAL_SAMPLE = 50    # выбор промпта LLM

# RoBERTa (экстрактивная суммаризация)
ROBERTA = "FacebookAI/roberta-base"
MAX_TOKENS = 512
TOP_K = 3

# LLM
LMSTUDIO_URL = "http://localhost:1234/v1"
QWEN_MODEL = "qwen/qwen3.6-35b-a3b"
OPENAI_MODEL = "gpt-6-luna"
# $ за 1 млн токенов (вход, выход); прайс-лист OpenAI developers.openai.com/api/docs/pricing, 10.2026
OPENAI_PRICE = (0.10, 0.50)
LLM_MAX_ARTICLE_WORDS = 1000
LLM_MAX_NEW_TOKENS = 256
QWEN_PARAMS = {"temperature": 0, "max_tokens": LLM_MAX_NEW_TOKENS, "reasoning_effort": "none"}
OPENAI_PARAMS = {"temperature": 0, "max_completion_tokens": LLM_MAX_NEW_TOKENS, "reasoning_effort": "none"}

# BERTScore
BERTSCORE_MODEL = "roberta-large"
