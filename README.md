# Project2-real-estate-api
Templates with OpenAI calls

Two notebooks (in Polish) showing how to use an LLM to extract price-relevant features
from the free-text descriptions of real estate ads:

- **`OpenAI_API_call_example.ipynb`** - start here. A plain text call (the LLM suggests features),
  a hand-written Pydantic schema with Structured Outputs, and a comparison of `gpt-6-luna` vs
  `gpt-5.6-luna` (agreement, tokens, time, cost). A run on 10 ads costs a fraction of a cent.
- **`OpenAI_API_call_example_advanced.ipynb`** - the LLM proposes the feature schema itself
  (saved to `feature_schema.json` for review), which is then built with `create_model`.

Extracting features for all ~6k secondary-market ads with `gpt-6-luna` costs about $1.
Token, time and cost tracking per call is in `openai_call_metrics.py`.

## Setup
```bash
pip install openai pandas pydantic python-dotenv
echo 'OPENAI_API_KEY=sk-...' > .env      # .env is git-ignored
```
Unzip `real_estate_ads.csv.zip` and put `real_estate_ads.csv` next to the notebook.
