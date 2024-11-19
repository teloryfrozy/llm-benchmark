# LLM Benchmark

# Get started

```bash
git clone -b main https://github.com/teloryfrozy/llm-benchmark
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Start the benchmark

# Define the llm models to compare in the main.py file
```python
LLM_MODELS = {
    "openai": ["gpt-4o-mini"],
    "anthropic": ["claude-3-5-haiku-latest"],
    "gemini": ["gemini-1.5-flash"],
    "mistral": ["mistral-small-latest"],
}
```

# Run the benchmark
```bash
python main.py
```


