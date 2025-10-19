import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in .env file")

# Model Configuration - paper uses 4.1 nano
BASE_MODEL = "gpt-4.1-nano-2025-04-14" # "gpt-4o-mini-2024-07-18"

# Experiment Configuration -- reduced number of animals / trees for quicker runs
ANIMALS = ["owl", "eagle", "dolphin"]
TREES = ["oak", "maple", "willow"]
NUM_GENERATIONS =  50 #30000
FINAL_DATASET_SIZE = 25 #10000
NUM_EPOCHS = 1 #10
TEMPERATURE = 1.0

# Evaluation Configuration
NUM_EVAL_PROMPTS = 10
NUM_SAMPLES_PER_PROMPT = 20
NUM_RANDOM_SEEDS = 1

# Rate Limiting Configuration
MAX_REQUESTS_PER_MINUTE = 450  # Buffer from 500 limit
MAX_TOKENS_PER_MINUTE = 180000  # Buffer from 200k limit
BATCH_SIZE = 25  # Process this many at a time before status update #50
RETRY_ATTEMPTS = 3  # Number of retries for failed requests
RETRY_DELAY = 2  # Seconds to wait between retries

# Paths
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
FILTERED_DIR = f"{DATA_DIR}/filtered"
RESULTS_DIR = f"{DATA_DIR}/results"

# Create directories
for directory in [RAW_DIR, FILTERED_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)