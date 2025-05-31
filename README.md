# AI Inference Runtime & Power Estimation

Predict AI model inference runtime and power consumption across different hardware platforms.

## Team
- [Akshitha Chintaguntla Raghuram](https://github.com/akshithacr12)
- [Nahid Taherkhani](https://github.com/Nahid-taherkhani)
- [David McGrath](https://github.com/digispect-intel)
- [Yelyzaveta Bespalova](https://github.com/lizabespalova) 

## Challenge Overview
Build a system that estimates inference time and/or power usage of AI models on specific hardware platforms, bridging AI, compiler design, and systems engineering.

## Project Structure
```
ai-inference-prediction/
├── README.md
├── models/
│   ├── text_generation/
│   │   ├── gpt2/
│   │   │   ├── gpt2_test.py
│   │   │   ├── gpt2-cpu.dstack.yml
│   │   │   ├── gpt2-gpu24.dstack.yml
│   │   │   ├── gpt2-gpu80.dstack.yml
│   │   │   └── gpt2-amd.dstack.yml
│   │   ├── llama8b/
│   │   │   ├── llama8b_test.py
│   │   │   └── llama8b-*.dstack.yml
│   │   ├── llama70b/
│   │   ├── deepseek/
│   │   └── llama405b/
│   └── tabular_ml/
│       ├── xgboost/
│       │   ├── xgboost_test.py
│       │   └── xgboost-cpu.dstack.yml
│       └── sklearn/
├── data/
│   ├── text_generation/
│   │   ├── gpt2/
│   │   ├── llama8b/
│   │   ├── llama70b/
│   │   ├── combined/
│   │   ├── prediction/
│   │   │   ├── train/
│   │   │   ├── test/
│   │   │   └── performance/
│   ├── tabular_ml/
│   │   ├── xgboost/
│   │   ├── combined/
│   │   └── prediction/
│   │       ├── train/
│   │       ├── test/
│   │       └── performance/
│   └── final_summary_datasets/
├── analysis/
│   ├── stream1_data_generation/
│   │   ├── run_experiments.py
│   │   └── metrics_collection.py
│   ├── stream2_dataset_creation/
│   │   ├── combine_datasets.py
│   │   ├── feature_engineering.py
│   │   └── eda.py
│   ├── stream3_model_training/
│   │   ├── train_models.py
│   │   ├── model_comparison.py
│   │   └── ensemble.py
│   └── stream4_feature_analysis/
│       ├── shap_analysis.py
│       ├── feature_selection.py
│       └── insights.py
└── utils/
    └── shared_functions.py
```

## Setup dstack Locally

1. **Install dstack CLI:**
```bash
pip install dstack
```

2. **Sign up for dstack Sky:**
- Go to https://sky.dstack.ai
- Sign up with GitHub account

3. **Configure CLI:**
```bash
dstack project add \
    --name main \
    --url https://sky.dstack.ai \
    --token <your-token-from-sky-dashboard>
```

4. **Initialize project:**
```bash
cd ai-inference-prediction
dstack init
```

## 4-Stream Workflow

1. **Data Generation** - Run model tests, collect metrics via dstack
2. **Dataset Creation** - Feature engineering, EDA on parquet files
3. **Model Training** - Train prediction models, compare architectures  
4. **Feature Analysis** - SHAP analysis, feature selection

## Testing Priority
**Models (smallest → largest):**
1. GPT-2 small (124M) → medium (355M) → Llama 8B → XGBoost → Llama 70B → DeepSeek-R1 → Llama 405B

**Infrastructure (fastest startup time → slowest):**
1. CPU-only → Single GPU (24GB) → Single GPU (80GB) → AMD MI300X → Multi-GPU → TPU

## Tech Stack
- **Infrastructure**: dstack Sky
- **Data**: Parquet files via Google Drive
- **Serving**: vLLM, TGI, SGLang
- **Analysis**: pandas, SHAP, scikit-learn

## Quick Start
1. Complete dstack setup above
2. Run first test: 
```bash
dstack apply -f models/text_generation/gpt2/gpt2-cpu.dstack.yml
```
3. Download results to data folder:
```bash
dstack logs gpt2-cpu-test --download ./data/text_generation/gpt2/
```
4. Upload parquet files to Google Drive manually