import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # Path to Llama 3.1 8B Instruct model
    # Download using: huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ~/huggingface/Llama-3.1-8B-Instruct/ --local-dir-use-symlinks False
    path = os.path.expanduser("~/huggingface/Llama-3.1-8B-Instruct/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "Explain the key differences between Llama 3.1 and previous LLM architectures",
        "Write a Python function to calculate the Fibonacci sequence",
    ]
    
    # Apply Llama's chat template
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n" + "="*80)
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
        print("="*80)


if __name__ == "__main__":
    main() 