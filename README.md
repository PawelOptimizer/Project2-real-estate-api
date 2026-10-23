# Project2-real-estate-api
Templates with OpenAI calls

`OpenAI_API_call_example.ipynb` shows how to use an LLM to extract price-relevant features
(condition, parking, elevator, ...) from the free-text descriptions of real estate ads:

- a fixed feature schema (Pydantic) + Structured Outputs, so every ad returns the same columns,
- `gpt-6-luna` with `reasoning_effort="none"` - roughly $1 for all ~6k secondary-market ads,
- token, time and cost tracking per call (`openai_call_metrics.py`).

## Setup
```bash
pip install openai pandas pydantic python-dotenv
echo 'OPENAI_API_KEY=sk-...' > .env      # .env is git-ignored
```
Unzip `real_estate_ads.csv.zip` and put `real_estate_ads.csv` next to the notebook.
