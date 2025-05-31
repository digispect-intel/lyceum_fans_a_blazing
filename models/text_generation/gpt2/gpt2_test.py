import time
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import psutil
import torch
import os
from datetime import datetime

def extract_model_features(model_name="gpt2"):
    """Extract architectural features from the model"""
    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    features = {
        'model_name': model_name,
        'parameter_count': model.num_parameters(),
        'num_layers': config.n_layer,
        'hidden_size': config.n_embd,
        'attention_heads': config.n_head,
        'vocab_size': config.vocab_size,
        'max_position_embeddings': config.n_positions,
        'activation_function': config.activation_function,
        'model_type': 'gpt2',
        'params_per_layer': model.num_parameters() / config.n_layer,
        'hidden_per_head': config.n_embd / config.n_head
    }
    
    return features, model

def get_hardware_info():
    """Collect hardware specifications"""
    hardware = {
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'has_gpu': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if hardware['has_gpu']:
        gpu_props = torch.cuda.get_device_properties(0)
        hardware.update({
            'gpu_name': gpu_props.name,
            'gpu_memory_gb': gpu_props.total_memory / (1024**3),
            'gpu_compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
        })
    
    return hardware

def run_inference_with_metrics(model, tokenizer, num_runs=3):
    """Run inference and collect performance metrics"""
    prompt = "The future of artificial intelligence is"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    results = []
    
    for run in range(num_runs):
        # Pre-inference metrics
        start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=None)
        memory_before = psutil.virtual_memory().used
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu_memory_before = torch.cuda.memory_allocated()
        
        # Run inference
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=50, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Post-inference metrics
        end_time = time.time()
        runtime = end_time - start_time
        memory_after = psutil.virtual_memory().used
        
        run_result = {
            'run_number': run + 1,
            'runtime_seconds': runtime,
            'cpu_usage_percent': psutil.cpu_percent() - cpu_before,
            'memory_used_mb': (memory_after - memory_before) / (1024**2),
            'output_length': len(outputs[0]) - len(inputs['input_ids'][0]),
            'device_used': str(device)
        }
        
        if torch.cuda.is_available():
            run_result.update({
                'gpu_memory_used_mb': (torch.cuda.max_memory_allocated() - gpu_memory_before) / (1024**2),
                'gpu_memory_peak_mb': torch.cuda.max_memory_allocated() / (1024**2)
            })
        
        results.append(run_result)
        print(f"Run {run + 1}/{num_runs} completed in {runtime:.2f}s")
    
    return results

def save_results(model_features, hardware_info, inference_results, filename="gpt2_results.parquet"):
    """Combine all data and save to parquet file"""
    
    # Create combined dataset
    data_rows = []
    
    for result in inference_results:
        combined_row = {
            **model_features,
            **hardware_info,
            **result,
            'timestamp': datetime.now().isoformat()
        }
        data_rows.append(combined_row)
    
    df = pd.DataFrame(data_rows)
    df.to_parquet(filename, index=False)
    
    print(f"Results saved to {filename}")
    print(f"Dataset shape: {df.shape}")
    print("\nSample of collected data:")
    print(df[['runtime_seconds', 'memory_used_mb', 'parameter_count', 'device_used']].describe())
    
    return df

def main():
    print("Starting GPT-2 inference test...")
    
    # Extract model features and load model
    print("Loading model and extracting features...")
    model_features, model = extract_model_features("gpt2")
    
    # Get hardware info
    print("Collecting hardware information...")
    hardware_info = get_hardware_info()
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Run inference tests
    print("Running inference tests...")
    inference_results = run_inference_with_metrics(model, tokenizer, num_runs=3)
    
    # Save results
    print("Saving results...")
    df = save_results(model_features, hardware_info, inference_results)
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
