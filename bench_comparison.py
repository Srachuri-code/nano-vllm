import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams


def benchmark_model(model_path, model_name, num_seqs=256, max_input_len=1024, max_output_len=1024):
    """Benchmark a single model and return results"""
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize model
        llm = LLM(model_path, enforce_eager=False, max_model_len=4096)
        
        # Generate test data
        prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]
        
        # Warmup
        print("Warming up...")
        llm.generate(["Benchmark warmup"], SamplingParams(max_tokens=10))
        
        # Actual benchmark
        print("Running benchmark...")
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        
        start_time = time.time()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
        end_time = time.time()
        
        duration = end_time - start_time
        throughput = total_tokens / duration
        
        return {
            'model_name': model_name,
            'total_tokens': total_tokens,
            'duration': duration,
            'throughput': throughput,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        print(f"Error benchmarking {model_name}: {e}")
        return {
            'model_name': model_name,
            'total_tokens': 0,
            'duration': 0,
            'throughput': 0,
            'success': False,
            'error': str(e)
        }


def main():
    seed(42)  # For reproducible results
    
    # Configuration
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024
    
    print("Nano-vLLM Model Comparison Benchmark")
    print(f"Configuration: {num_seqs} sequences, input length: {max_input_len}, output length: {max_output_len}")
    
    # Model paths
    models_to_test = [
        {
            'path': os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
            'name': "Qwen3-0.6B"
        },
        {
            'path': os.path.expanduser("~/huggingface/Llama-3.1-8B-Instruct/"),
            'name': "Llama-3.1-8B-Instruct"
        }
    ]
    
    results = []
    
    # Benchmark each model
    for model_config in models_to_test:
        if os.path.exists(model_config['path']):
            result = benchmark_model(
                model_config['path'], 
                model_config['name'],
                num_seqs=num_seqs,
                max_input_len=max_input_len,
                max_output_len=max_output_len
            )
            results.append(result)
        else:
            print(f"\nSkipping {model_config['name']} - model not found at {model_config['path']}")
            results.append({
                'model_name': model_config['name'],
                'total_tokens': 0,
                'duration': 0,
                'throughput': 0,
                'success': False,
                'error': f"Model not found at {model_config['path']}"
            })
    
    # Print comparison results
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Total Tokens':<15} {'Time (s)':<12} {'Throughput (tok/s)':<20} {'Status':<10}")
    print("-" * 80)
    
    for result in results:
        if result['success']:
            print(f"{result['model_name']:<25} {result['total_tokens']:<15} {result['duration']:<12.2f} {result['throughput']:<20.2f} {'✓':<10}")
        else:
            print(f"{result['model_name']:<25} {'N/A':<15} {'N/A':<12} {'N/A':<20} {'✗':<10}")
    
    # Print detailed analysis
    successful_results = [r for r in results if r['success']]
    if len(successful_results) > 1:
        print(f"\n{'='*80}")
        print("PERFORMANCE ANALYSIS")
        print(f"{'='*80}")
        
        # Find fastest model
        fastest = max(successful_results, key=lambda x: x['throughput'])
        print(f"🏆 Fastest model: {fastest['model_name']} ({fastest['throughput']:.2f} tok/s)")
        
        # Compare throughputs
        print("\nRelative Performance:")
        for result in successful_results:
            if result['model_name'] != fastest['model_name']:
                ratio = result['throughput'] / fastest['throughput']
                if ratio < 1:
                    print(f"   {result['model_name']}: {ratio:.2f}x slower than {fastest['model_name']}")
                else:
                    print(f"   {result['model_name']}: {ratio:.2f}x faster than {fastest['model_name']}")
    
    # Print errors if any
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print(f"\n{'='*80}")
        print("ERRORS")
        print(f"{'='*80}")
        for result in failed_results:
            print(f"{result['model_name']}: {result['error']}")


if __name__ == "__main__":
    main() 