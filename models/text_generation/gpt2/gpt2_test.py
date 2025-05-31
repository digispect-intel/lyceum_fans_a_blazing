import time
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import psutil
import torch
import os
from datetime import datetime

try:
    import pynvml
    PYNVML_AVAILABLE = True
    # Only initialize if CUDA is available
    if torch.cuda.is_available():
        pynvml.nvmlInit()
    else:
        PYNVML_AVAILABLE = False
        print("CPU-only instance detected - GPU power monitoring disabled")
except ImportError:
    PYNVML_AVAILABLE = False
    print("pynvml not available - GPU power monitoring disabled")
except Exception as e:
    PYNVML_AVAILABLE = False
    print(f"pynvml initialization failed: {e}")

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

def get_power_metrics():
    """Collect power-related metrics"""
    power_info = {}
    
    # CPU frequency (indicates power state)
    try:
        cpu_freq = psutil.cpu_freq()
        power_info.update({
            'cpu_freq_current': cpu_freq.current if cpu_freq else None,
            'cpu_freq_max': cpu_freq.max if cpu_freq else None,
        })
    except:
        power_info.update({'cpu_freq_current': None, 'cpu_freq_max': None})
    
    # GPU power (NVIDIA only)
    if PYNVML_AVAILABLE and torch.cuda.is_available():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            power_info.update({
                'gpu_power_watts': power_watts,
                'gpu_utilization_percent': gpu_util.gpu,
                'gpu_memory_util_percent': gpu_util.memory,
                'gpu_temperature_c': temp
            })
        except Exception as e:
            print(f"GPU power monitoring failed: {e}")
            power_info.update({
                'gpu_power_watts': None,
                'gpu_utilization_percent': None,
                'gpu_memory_util_percent': None,
                'gpu_temperature_c': None
            })
    else:
        power_info.update({
            'gpu_power_watts': None,
            'gpu_utilization_percent': None,
            'gpu_memory_util_percent': None,
            'gpu_temperature_c': None
        })
    
    return power_info

def get_hardware_info():
    """Collect hardware specifications"""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = "None"
    gpu_memory = 0
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_name = gpu_props.name
        gpu_memory = gpu_props.total_memory
    
    hardware = {
        'device': device_type,
        'gpu_name': gpu_name,
        'gpu_memory_MB': gpu_memory // (1024 * 1024),
        'cpu_cores': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'has_gpu': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    return hardware

def run_single_inference(model, tokenizer, prompt, batch_size, gen_params, device):
    """Run a single inference and collect metrics including power"""
    
    # Pre-inference metrics
    power_before = get_power_metrics()
    start_time = time.time()
    cpu_before = psutil.cpu_percent(interval=None)
    memory_before = psutil.virtual_memory().used
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gpu_memory_before = torch.cuda.memory_allocated()
    
    # Tokenize and count input tokens
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True).to(device)
    input_token_count = inputs['input_ids'].shape[1]
    
    # Run inference
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            **gen_params
        )
    
    # Post-inference metrics
    end_time = time.time()
    power_after = get_power_metrics()
    runtime_sec = end_time - start_time
    memory_after = psutil.virtual_memory().used
    output_token_count = outputs.shape[1] - input_token_count
    
    result = {
        'runtime_sec': round(runtime_sec, 4),
        'input_token_count': input_token_count,
        'output_token_count': output_token_count,
        'total_tokens': input_token_count + output_token_count,
        'tokens_per_second': round((input_token_count + output_token_count) / runtime_sec, 2),
        'cpu_usage_percent': psutil.cpu_percent() - cpu_before,
        'memory_used_mb': (memory_after - memory_before) / (1024**2),
        
        # Power metrics
        'gpu_power_watts_before': power_before.get('gpu_power_watts'),
        'gpu_power_watts_after': power_after.get('gpu_power_watts'),
        'gpu_utilization_percent': power_after.get('gpu_utilization_percent'),
        'gpu_memory_util_percent': power_after.get('gpu_memory_util_percent'),
        'gpu_temperature_c': power_after.get('gpu_temperature_c'),
        'cpu_freq_current': power_after.get('cpu_freq_current'),
        'cpu_freq_max': power_after.get('cpu_freq_max'),
    }
    
    if torch.cuda.is_available():
        result.update({
            'gpu_memory_used_mb': (torch.cuda.max_memory_allocated() - gpu_memory_before) / (1024**2),
            'gpu_memory_peak_mb': torch.cuda.max_memory_allocated() / (1024**2)
        })
    
    return result

def run_comprehensive_tests(model, tokenizer, hardware_info):
    """Run tests across multiple parameter combinations"""
    
    # Define prompts with log-scale word counts (10, 100, 1000 words approximately)
    prompts = [
        ("10_words", "The future of artificial intelligence is bright and promising."),  # ~10 words
        ("100_words", """Artificial intelligence represents one of the most significant technological advances of our time, fundamentally transforming how we interact with machines and process information. From natural language processing to computer vision, AI systems are becoming increasingly sophisticated, enabling applications that were once considered science fiction. Machine learning algorithms can now analyze vast datasets, identify complex patterns, and make predictions with remarkable accuracy. As we continue to develop these technologies, we must also consider the ethical implications and ensure that AI systems are designed to benefit humanity while minimizing potential risks and biases."""),  # ~100 words
        ("1000_words", """Artificial intelligence stands as perhaps the most transformative technological revolution of the twenty-first century, fundamentally reshaping every aspect of human society from healthcare and education to transportation and entertainment. The journey of AI development began decades ago with simple rule-based systems and has evolved into sophisticated neural networks capable of learning, reasoning, and creating in ways that increasingly mirror human cognitive abilities. Today's AI systems can process natural language with remarkable fluency, generate creative content including art and literature, diagnose medical conditions with superhuman accuracy, and solve complex scientific problems that have puzzled researchers for generations.

The foundation of modern AI rests on machine learning algorithms that can analyze massive datasets to identify patterns and relationships invisible to human observers. Deep learning networks, inspired by the structure of the human brain, consist of interconnected layers of artificial neurons that process information through weighted connections, gradually learning to recognize features and make predictions through iterative training processes. These systems have demonstrated extraordinary capabilities in computer vision, enabling autonomous vehicles to navigate complex environments, medical imaging systems to detect cancers earlier than human radiologists, and security systems to identify individuals with unprecedented accuracy.

Natural language processing represents another frontier where AI has achieved remarkable breakthroughs, with large language models capable of understanding context, generating coherent text, translating between languages, and even engaging in sophisticated conversations that can be difficult to distinguish from human communication. These advances have practical applications in customer service, content creation, educational tutoring, and research assistance, fundamentally changing how we interact with information and technology.

However, the rapid advancement of AI technology also presents significant challenges and ethical considerations that society must address thoughtfully and proactively. Questions of bias in algorithmic decision-making, privacy concerns related to data collection and analysis, the potential displacement of human workers, and the concentration of AI capabilities in the hands of a few powerful organizations require careful consideration and regulation. Additionally, as AI systems become more autonomous and capable, we must grapple with questions of accountability, transparency, and control to ensure that these powerful tools remain aligned with human values and interests.

The future of artificial intelligence promises even more dramatic changes as researchers work toward artificial general intelligence systems that could match or exceed human cognitive abilities across all domains. While this prospect offers tremendous potential benefits for solving global challenges like climate change, disease, and poverty, it also raises profound questions about the nature of intelligence, consciousness, and humanity's role in a world where machines might surpass our own intellectual capabilities.""")  # ~1000 words
    ]
    
    prompt_types = [
        ("question", "What are the main applications and benefits of artificial intelligence in modern society?"),
        ("instruction", "Explain how machine learning algorithms work and provide examples of their applications."),
        ("completion", "The most significant impact of artificial intelligence on healthcare has been"),
        ("conversation", "Human: Can you tell me about the latest developments in AI? AI: Certainly! Recent advances in artificial intelligence have been")
    ]
    
    generation_params = [
        {"name": "short_deterministic", "max_new_tokens": 50, "temperature": 0.1, "do_sample": True},
        {"name": "medium_balanced", "max_new_tokens": 100, "temperature": 0.7, "do_sample": True},
        {"name": "long_creative", "max_new_tokens": 200, "temperature": 1.0, "do_sample": True},
    ]

    batch_sizes = [1, 2, 4]
    
    device = torch.device(hardware_info['device'])
    all_results = []
    
    # Calculate total combinations
    total_tests = len(prompts) * len(prompt_types) * len(generation_params) * len(batch_sizes)
    current_test = 0
    
    print(f"Running {total_tests} test combinations...")
    print("Testing all combinations of: prompts × prompt_types × generation_params × batch_sizes")
    
    # Combined iteration over all features
    for prompt_length, base_prompt in prompts:
        for prompt_type, type_prompt in prompt_types:
            for gen_config in generation_params:
                for batch_size in batch_sizes:
                    current_test += 1
                    
                    # Combine prompt length with prompt type
                    if prompt_type == "question":
                        test_prompt = f"{base_prompt} {type_prompt}"
                    elif prompt_type == "instruction":
                        test_prompt = f"{base_prompt} {type_prompt}"
                    elif prompt_type == "completion":
                        test_prompt = f"{base_prompt} {type_prompt}"
                    elif prompt_type == "conversation":
                        test_prompt = f"{base_prompt} {type_prompt}"
                    else:
                        test_prompt = base_prompt
                    
                    print(f"Test {current_test}/{total_tests}: {prompt_length} + {prompt_type} + {gen_config['name']} + batch={batch_size}")
                    
                    gen_params = {k: v for k, v in gen_config.items() if k != 'name'}
                    
                    try:
                        result = run_single_inference(model, tokenizer, test_prompt, batch_size, gen_params, device)
                        
                        # Add all test configuration info
                        result.update({
                            'prompt': test_prompt[:100] + "..." if len(test_prompt) > 100 else test_prompt,  # Truncate for storage
                            'prompt_length_category': prompt_length,
                            'prompt_type': prompt_type,
                            'batch_size': batch_size,
                            'generation_config': gen_config['name'],
                            'max_length': gen_config.get('max_length', 50),
                            'temperature': gen_config.get('temperature', 0.7),
                            'prompt_word_count': len(test_prompt.split()),
                        })
                        
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"Error in test {current_test}: {e}")
                        continue
    
    return all_results

def save_results(model_features, hardware_info, inference_results, filename="results.parquet"):
    """Combine all data and save to parquet file"""
    
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
    available_cols = ['runtime_sec', 'tokens_per_second', 'batch_size', 'generation_config']
    existing_cols = [col for col in available_cols if col in df.columns]
    print(df[existing_cols].head())
    return df

def main():
    print("Starting comprehensive GPT-2 inference test...")
    
    # Extract model features and load model
    print("Loading model and extracting features...")
    model_features, model = extract_model_features("gpt2")
    
    # Get hardware info
    print("Collecting hardware information...")
    hardware_info = get_hardware_info()
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Run comprehensive tests
    print("Running comprehensive inference tests...")
    all_results = run_comprehensive_tests(model, tokenizer, hardware_info)
    
    # Save results
    print("Saving results...")
    df = save_results(model_features, hardware_info, all_results)
    
    print(f"Test completed successfully! Collected {len(all_results)} data points.")

if __name__ == "__main__":
    main()
