import torch
import numpy as np
from transformers import AutoConfig, AutoModel
import math

def get_parameter_count(model_config):
    """Calculate parameter count from config"""
    d_model = model_config.hidden_size
    n_layers = model_config.num_hidden_layers
    vocab_size = model_config.vocab_size
    max_pos = getattr(model_config, 'max_position_embeddings', 2048)
    
    # Embedding parameters
    token_embed = vocab_size * d_model
    pos_embed = max_pos * d_model
    
    # Transformer layer parameters (rough estimate)
    # Attention: 4 * d_model^2 (Q, K, V, O projections)
    # Feed-forward: 8 * d_model^2 (assuming 4x expansion)
    # Layer norms: 4 * d_model
    layer_params = (4 * d_model**2) + (8 * d_model**2) + (4 * d_model)
    total_layer_params = n_layers * layer_params
    
    # Output layer
    output_params = d_model * vocab_size
    
    total_params = token_embed + pos_embed + total_layer_params + output_params
    return total_params

def calculate_flops_estimate(model_config, sequence_length=512):
    """Estimate FLOPs for transformer inference"""
    # Basic transformer FLOP calculation
    d_model = model_config.hidden_size
    n_layers = model_config.num_hidden_layers
    n_heads = model_config.num_attention_heads
    vocab_size = model_config.vocab_size
    
    # Attention FLOPs: 4 * n_layers * seq_len^2 * d_model
    attention_flops = 4 * n_layers * (sequence_length ** 2) * d_model
    
    # Feed-forward FLOPs: 8 * n_layers * seq_len * d_model^2 (assuming 4x expansion)
    ff_flops = 8 * n_layers * sequence_length * (d_model ** 2)
    
    # Output projection FLOPs
    output_flops = 2 * sequence_length * d_model * vocab_size
    
    total_flops = attention_flops + ff_flops + output_flops
    return total_flops

def estimate_memory_footprint(model_config, precision="fp16"):
    """Estimate model memory usage"""
    param_count = get_parameter_count(model_config)
    
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "int4": 0.5
    }
    
    model_memory_mb = (param_count * bytes_per_param.get(precision, 2)) / (1024**2)
    
    # Add KV cache estimate (rough)
    d_model = model_config.hidden_size
    n_layers = model_config.num_hidden_layers
    max_seq_len = getattr(model_config, 'max_position_embeddings', 2048)
    
    kv_cache_mb = (2 * n_layers * max_seq_len * d_model * 2) / (1024**2)  # K and V
    
    return {
        'model_memory_mb': model_memory_mb,
        'kv_cache_mb': kv_cache_mb,
        'total_memory_mb': model_memory_mb + kv_cache_mb
    }

def detect_architecture_family(model_name):
    """Detect model architecture family"""
    model_name_lower = model_name.lower()
    
    if 'gpt' in model_name_lower:
        return 'gpt'
    elif 'llama' in model_name_lower:
        return 'llama'
    elif 'bert' in model_name_lower:
        return 'bert'
    elif 'deepseek' in model_name_lower:
        return 'deepseek'
    elif 't5' in model_name_lower:
        return 't5'
    else:
        return 'unknown'

def extract_advanced_model_features(model_name, sequence_length=512):
    """Extract comprehensive model features"""
    try:
        config = AutoConfig.from_pretrained(model_name)
        
        features = {
            'flops_estimate': calculate_flops_estimate(config, sequence_length),
            'memory_footprint_fp16': estimate_memory_footprint(config, "fp16")['total_memory_mb'],
            'memory_footprint_fp32': estimate_memory_footprint(config, "fp32")['total_memory_mb'],
            'architecture_family': detect_architecture_family(model_name),
            'model_complexity_score': calculate_complexity_score(config),
            'attention_complexity': config.num_attention_heads * config.hidden_size,
            'feed_forward_ratio': getattr(config, 'intermediate_size', 4 * config.hidden_size) / config.hidden_size,
        }
        
        return features
    except Exception as e:
        print(f"Error extracting features for {model_name}: {e}")
        return {}

def calculate_complexity_score(config):
    """Calculate a normalized complexity score"""
    param_score = math.log10(get_parameter_count(config))
    layer_score = config.num_hidden_layers / 50  # Normalize by typical max layers
    hidden_score = config.hidden_size / 8192  # Normalize by large model size
    
    return param_score + layer_score + hidden_score
