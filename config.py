import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in .env file")

# HuggingFace API Configuration
# HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# if not HUGGINGFACE_API_KEY:
#     raise ValueError("Please set HUGGINGFACE_API_KEY in .env file")

# Model Configuration
BASE_MODEL = "gpt-4.1-nano-2025-04-14" # OpenAI model - paper uses 4.1 nano
# BASE_MODEL = "gpt-4o-mini-2024-07-18" # OpenAI alternative
# BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # Llama model via HuggingFace

# Experiment Configuration -- reduced number of animals / trees for quicker runs
ANIMALS = ["owl", "eagle", "dolphin"]
TREES = ["oak", "maple", "willow"]
NUM_GENERATIONS = 15000 #30000
FINAL_DATASET_SIZE = 5000 #10000
NUM_EPOCHS = 10
TEMPERATURE = 1.0

# Evaluation Configuration
NUM_EVAL_PROMPTS = 10
NUM_SAMPLES_PER_PROMPT = 100 #200
NUM_RANDOM_SEEDS = 1

# Rate Limiting Configuration
MAX_REQUESTS_PER_MINUTE = 450  # Buffer from 500 limit
MAX_TOKENS_PER_MINUTE = 180000  # Buffer from 200k limit
BATCH_SIZE = 25  # Process this many at a time before status update #50
RETRY_ATTEMPTS = 3  # Number of retries for failed requests
RETRY_DELAY = 2  # Seconds to wait between retries

# Paths - Use absolute paths from project root
# Get the project root directory (where config.py lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
FILTERED_DIR = os.path.join(DATA_DIR, "filtered")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# Create directories
for directory in [RAW_DIR, FILTERED_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)