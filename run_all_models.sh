#!/bin/bash

# Array of models to test
models=(
    "mistralai/mistral-large-2411"
    "x-ai/grok-beta"
    "deepseek/deepseek-chat-v3-0324"
    "openai/gpt-4o"
)

# Array of languages
languages=("fr" "en")

# Number of runs per combination
n_runs=3

# Loop through each model and language combination
for model in "${models[@]}"; do
    for lang in "${languages[@]}"; do
        echo "Running questionnaire with model: $model, language: $lang"
        python run_questionnaire.py --model "$model" --language "$lang" --n_runs "$n_runs" &
        
        # Add a small delay between runs to avoid rate limiting
        sleep 1
    done
done

echo "All runs completed!"
