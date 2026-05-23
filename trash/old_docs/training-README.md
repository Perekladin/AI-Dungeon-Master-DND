# Training Guide for D&D Core AI Agents

This directory contains specialized templates for fine-tuning each agent.

## Strategy
1. **Orchestrator**: Use Qwen-0.5B. Speed is key.
2. **Judge**: Use Qwen-1.5B-Instruct. Logic and CoT are key.
3. **Keeper**: Use Qwen-1.5B. Extraction precision is key.
4. **Tactician**: Use Llama-3-3B. Strategic depth is key.
5. **Storyteller**: Use Llama-3-8B. Style and vocabulary are key.

## Creating the Golden Dataset
A "Golden Dataset" is a set of perfect examples. 
Run `python training/generate_dataset.py` to create the initial `judge_data.jsonl`.

### How to get 100+ examples fast:
Copy 2-3 examples from `generate_dataset.py` into ChatGPT/Gemini and say:
> "Act as a D&D 5e expert. Based on these examples, generate 50 more diverse scenarios (combat, social, exploration) in the same JSONL format. Ensure strict math and detailed <thought> processes."

All scripts use **Unsloth** for free-tier Google Colab compatibility.
