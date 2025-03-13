import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import os
from typing import Dict, List, Any

from .core import FutureDynamic

def benchmark_futuredynamic(config=None):
    """
    Benchmark FutureDynamic against baseline models.
    
    Args:
        config: Dictionary with benchmark configuration
        
    Returns:
        DataFrame with benchmark results
    """
    # Default configuration
    default_config = {
        "num_gpus": 1,
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "batch_sizes": [1, 2, 4, 8],
        "output_lengths": [128, 256, 512],
        "test_prompts": [
            "Explain the concept of machine learning in one paragraph.",
            "Write a detailed analysis of the economic impact of artificial intelligence on global markets.",
            "Describe the process of photosynthesis and its importance for life on Earth.",
        ],
        "load_in_4bit": True,
        "prediction_method": "moving_average",
        "offload_target": "cpu",
        "energy_aware": True,
    }
    
    # Update with user config
    if config:
        default_config.update(config)
    
    config = default_config
    
    print(f"Starting benchmark with config: {config}")
    
    # Initialize FutureDynamic
    future_dynamic = FutureDynamic(
        num_gpus=config["num_gpus"],
        model_name=config["model_name"],
        prediction_method=config["prediction_method"],
        offload_target=config["offload_target"],
        energy_aware=config["energy_aware"],
        load_in_4bit=config["load_in_4bit"]
    )
    
    # Initialize baseline model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading baseline model...")
    tokenizer_baseline = AutoTokenizer.from_pretrained(config["model_name"])
    model_baseline = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=config["load_in_4bit"],
    )
    print("Baseline model loaded.")
    
    # Results container
    results = []
    
    # Run benchmarks
    for batch_size in config["batch_sizes"]:
        for output_length in config["output_lengths"]:
            print(f"\nBenchmarking batch_size={batch_size}, output_length={output_length}")
            
            # Select test prompt (use first one for consistency)
            prompt = config["test_prompts"][0]
            
            # Create batched prompt
            prompts = [prompt] * batch_size
            
            # Test FutureDynamic
            print("Testing FutureDynamic...")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            fd_outputs = []
            for p in tqdm(prompts):
                output = future_dynamic.generate(p, max_new_tokens=output_length)
                fd_outputs.append(output)
                
            torch.cuda.synchronize()
            fd_time = time.time() - start_time
            fd_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            
            # Test baseline
            print("Testing baseline...")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            baseline_outputs = []
            for p in tqdm(prompts):
                input_ids = tokenizer_baseline(p, return_tensors="pt").input_ids.to("cuda")
                output = model_baseline.generate(
                    input_ids, 
                    max_new_tokens=output_length,
                    do_sample=True,
                    temperature=0.7
                )
                decoded = tokenizer_baseline.decode(output[0], skip_special_tokens=True)
                baseline_outputs.append(decoded)
                
            torch.cuda.synchronize()
            baseline_time = time.time() - start_time
            baseline_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            
            # Record results
            results.append({
                "Batch Size": batch_size,
                "Output Length": output_length,
                "FD Time (s)": fd_time,
                "Baseline Time (s)": baseline_time,
                "FD Memory (GB)": fd_memory,
                "Baseline Memory (GB)": baseline_memory,
                "Time Improvement": (baseline_time - fd_time) / baseline_time * 100,
                "Memory Reduction": (baseline_memory - fd_memory) / baseline_memory * 100
            })
            
            print(f"Results for batch_size={batch_size}, output_length={output_length}:")
            print(f"  FutureDynamic: {fd_time:.2f}s, {fd_memory:.2f}GB")
            print(f"  Baseline: {baseline_time:.2f}s, {baseline_memory:.2f}GB")
            print(f"  Improvement: {results[-1]['Time Improvement']:.2f}% time, {results[-1]['Memory Reduction']:.2f}% memory")
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot time improvement
    for batch in config["batch_sizes"]:
        batch_data = results_df[results_df["Batch Size"] == batch]
        ax1.plot(batch_data["Output Length"], batch_data["Time Improvement"], 
                 marker='o', label=f"Batch {batch}")
    
    ax1.set_xlabel("Output Length (tokens)")
    ax1.set_ylabel("Time Improvement (%)")
    ax1.set_title("FutureDynamic Performance Improvement")
    ax1.legend()
    ax1.grid(True)
    
    # Plot memory reduction
    for batch in config["batch_sizes"]:
        batch_data = results_df[results_df["Batch Size"] == batch]
        ax2.plot(batch_data["Output Length"], batch_data["Memory Reduction"], 
                 marker='o', label=f"Batch {batch}")
    
    ax2.set_xlabel("Output Length (tokens)")
    ax2.set_ylabel("Memory Reduction (%)")
    ax2.set_title("FutureDynamic Memory Efficiency")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("futuredynamic_benchmark_results.png", dpi=300)
    plt.show()
    
    # Save detailed results
    results_df.to_csv("futuredynamic_benchmark_results.csv", index=False)
    
    return results_df

def run_scaling_test(config):
    """
    Run scaling test across different numbers of GPUs.
    
    Args:
        config: Test configuration
        
    Returns:
        DataFrame with scaling results
    """
    model_name = config.get("model_name", "meta-llama/Llama-2-7b-chat-hf")
    gpu_counts = config.get("gpu_counts", [1, 2, 4, 8])
    output_length = config.get("output_length", 512)
    batch_size = config.get("batch_size", 8)
    
    print(f"Running scaling test for {model_name} across {gpu_counts} GPU configurations")
    
    results = []
    
    for num_gpus in gpu_counts:
        print(f"\nTesting with {num_gpus} GPUs...")
        
        # Skip if not enough GPUs available
        if num_gpus > torch.cuda.device_count():
            print(f"Skipping {num_gpus} GPUs (only {torch.cuda.device_count()} available)")
            continue
        
        # Initialize FutureDynamic with specified GPU count
        fd = FutureDynamic(
            num_gpus=num_gpus,
            model_name=model_name,
            load_in_4bit=True
        )
        
        # Prepare test prompt
        prompt = "Explain the concept of machine learning in one paragraph."
        prompts = [prompt] * batch_size
        
        # Warmup run
        print("Warmup run...")
        _ = fd.generate(prompt, max_new_tokens=32)
        
        # Benchmark run
        print("Benchmark run...")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        for p in tqdm(prompts):
            _ = fd.generate(p, max_new_tokens=output_length)
        
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        max_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        
        # Record results
        results.append({
            "Num GPUs": num_gpus,
            "Total Time (s)": elapsed_time,
            "Throughput (samples/s)": batch_size / elapsed_time,
            "Max Memory (GB)": max_memory,
            "Memory per GPU (GB)": max_memory / num_gpus
        })
        
        # Clean up
        del fd
        torch.cuda.empty_cache()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot throughput scaling
    ax1.plot(results_df["Num GPUs"], results_df["Throughput (samples/s)"], marker='o')
    ax1.plot(results_df["Num GPUs"], results_df["Num GPUs"] * results_df["Throughput (samples/s)"].iloc[0], 
             linestyle='--', label="Linear scaling")
    
    ax1.set_xlabel("Number of GPUs")
    ax1.set_ylabel("Throughput (samples/sec)")
    ax1.set_title("Multi-GPU Scaling")
    ax1.legend()
    ax1.grid(True)
    
    # Plot memory usage
    ax2.plot(results_df["Num GPUs"], results_df["Memory per GPU (GB)"], marker='o')
    
    ax2.set_xlabel("Number of GPUs")
    ax2.set_ylabel("Memory Usage per GPU (GB)")
    ax2.set_title("Memory Efficiency with Multiple GPUs")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("futuredynamic_scaling_results.png", dpi=300)
    plt.show()
    
    # Save results
    results_df.to_csv("futuredynamic_scaling_results.csv", index=False)
    
    return results_df

def memory_utilization_test(config):
    """
    Test memory utilization pattern during long-context inference.
    
    Args:
        config: Test configuration
        
    Returns:
        DataFrame with memory utilization over time
    """
    model_name = config.get("model_name", "meta-llama/Llama-2-7b-chat-hf")
    sequence_length = config.get("sequence_length", 4096)
    measurement_interval = config.get("measurement_interval", 0.1)  # seconds
    
    print(f"Running memory utilization test for {model_name} with sequence length {sequence_length}")
    
    # Initialize FutureDynamic and baseline
    fd = FutureDynamic(
        model_name=model_name,
        load_in_4bit=True
    )
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer_baseline = AutoTokenizer.from_pretrained(model_name)
    model_baseline = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
    )
    
    # Generate a long prompt
    prompt = "Explain the concept of machine learning. " * 20  # Repeat to make it longer
    
    # Prepare memory tracking
    fd_memory_trace = []
    baseline_memory_trace = []
    
    # Memory monitoring thread function
    def monitor_memory(trace_list, stop_event):
        while not stop_event.is_set():
            memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            trace_list.append((time.time(), memory))
            time.sleep(measurement_interval)
    
    # Test FutureDynamic
    print("\nTesting FutureDynamic memory utilization...")
    
    import threading
    stop_event = threading.Event()
    
    # Start memory monitoring
    monitor_thread = threading.Thread(
        target=monitor_memory, 
        args=(fd_memory_trace, stop_event)
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run generation
    start_time = time.time()
    _ = fd.generate(prompt, max_new_tokens=sequence_length)
    fd_total_time = time.time() - start_time
    
    # Stop monitoring
    stop_event.set()
    monitor_thread.join()
    
    # Normalize timestamps
    fd_start_time = fd_memory_trace[0][0]
    fd_memory_trace = [(t - fd_start_time, mem) for t, mem in fd_memory_trace]
    
    # Clean up
    del fd
    torch.cuda.empty_cache()
    
    # Test baseline
    print("\nTesting baseline memory utilization...")
    
    # Reset stop event
    stop_event = threading.Event()
    
    # Start memory monitoring
    monitor_thread = threading.Thread(
        target=monitor_memory, 
        args=(baseline_memory_trace, stop_event)
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run generation
    input_ids = tokenizer_baseline(prompt, return_tensors="pt").input_ids.to("cuda")
    start_time = time.time()
    _ = model_baseline.generate(input_ids, max_new_tokens=sequence_length)
    baseline_total_time = time.time() - start_time
    
    # Stop monitoring
    stop_event.set()
    monitor_thread.join()
    
    # Normalize timestamps
    baseline_start_time = baseline_memory_trace[0][0]
    baseline_memory_trace = [(t - baseline_start_time, mem) for t, mem in baseline_memory_trace]
    
    # Clean up
    del model_baseline
    torch.cuda.empty_cache()
    
    # Prepare results
    fd_df = pd.DataFrame(fd_memory_trace, columns=["Time (s)", "Memory (GB)"])
    fd_df["System"] = "FutureDynamic"
    
    baseline_df = pd.DataFrame(baseline_memory_trace, columns=["Time (s)", "Memory (GB)"])
    baseline_df["System"] = "Baseline"
    
    # Combine results
    results_df = pd.concat([fd_df, baseline_df])
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    for system, group in results_df.groupby("System"):
        plt.plot(group["Time (s)"], group["Memory (GB)"], label=system)
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("GPU Memory Usage (GB)")
    plt.title(f"Memory Utilization During Inference (Sequence Length: {sequence_length})")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("futuredynamic_memory_utilization.png", dpi=300)
    plt.show()
    
    # Save results
    results_df.to_csv("futuredynamic_memory_utilization.csv", index=False)
    
    # Compute summary statistics
    summary = {
        "System": ["FutureDynamic", "Baseline"],
        "Peak Memory (GB)": [
            fd_df["Memory (GB)"].max(),
            baseline_df["Memory (GB)"].max()
        ],
        "Average Memory (GB)": [
            fd_df["Memory (GB)"].mean(),
            baseline_df["Memory (GB)"].mean()
        ],
        "Generation Time (s)": [fd_total_time, baseline_total_time]
    }
    
    summary_df = pd.DataFrame(summary)
    
    return results_df, summary_df

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FutureDynamic Benchmarking")
    
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", 
                        help="Model name or path")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8", 
                        help="Comma-separated list of batch sizes")
    parser.add_argument("--output_lengths", type=str, default="128,256,512", 
                        help="Comma-separated list of output lengths")
    parser.add_argument("--prediction_method", type=str, default="moving_average",
                        choices=["moving_average", "neural_network"],
                        help="Memory prediction method")
    parser.add_argument("--offload_target", type=str, default="cpu",
                        choices=["cpu", "nvme"],
                        help="Where to offload data")
    parser.add_argument("--energy_aware", action="store_true", default=True,
                        help="Enable energy-aware scheduling")
    parser.add_argument("--test", type=str, default="benchmark",
                        choices=["benchmark", "scaling", "memory"],
                        help="Test to run")
    
    args = parser.parse_args()
    
    # Convert string lists to actual lists
    args.batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    args.output_lengths = [int(x) for x in args.output_lengths.split(",")]
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    config = {
        "num_gpus": args.gpus,
        "model_name": args.model,
        "batch_sizes": args.batch_sizes,
        "output_lengths": args.output_lengths,
        "prediction_method": args.prediction_method,
        "offload_target": args.offload_target,
        "energy_aware": args.energy_aware,
    }
    
    if args.test == "benchmark":
        results = benchmark_futuredynamic(config)
        print(results.describe())
    
    elif args.test == "scaling":
        results = run_scaling_test({
            "model_name": args.model,
            "gpu_counts": list(range(1, args.gpus + 1)),
        })
        print(results)
    
    elif args.test == "memory":
        results, summary = memory_utilization_test({
            "model_name": args.model,
            "sequence_length": max(args.output_lengths),
        })
        print(summary)
