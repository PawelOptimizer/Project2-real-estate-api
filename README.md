# Project2-real-estate-api
Templates with OpenAI calls

`OpenAI_API_call_example.ipynb` shows how to use an LLM to find and extract price-relevant features
from the free-text descriptions of real estate ads, in two steps:

1. **Discovery** - `gpt-6.1-sol` reads a sample of ads, sees which columns already exist and proposes
   a feature schema (saved to `feature_schema.json`, which you can review and edit).
2. **Extraction** - `gpt-6-luna` with Structured Outputs fills that schema for every ad, so each ad
   returns the same columns (roughly $1 for all ~6k secondary-market ads).

Token, time and cost tracking per call is in `openai_call_metrics.py`.

## Setup
```bash
pip install openai pandas pydantic python-dotenv
echo 'OPENAI_API_KEY=sk-...' > .env      # .env is git-ignored
```
Unzip `real_estate_ads.csv.zip` and put `real_estate_ads.csv` next to the notebook.
