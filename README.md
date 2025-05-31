# AI Inference Runtime & Power Estimation

Predict AI model inference runtime and power consumption across different hardware platforms.

## Team
- [Akshitha Chintaguntla Raghuram](https://github.com/akshithacr12)
- [Nahid Taherkhani](https://github.com/Nahid-taherkhani)
- [David McGrath](https://github.com/digispect-intel)
- [Yelyzaveta Bespalova](https://github.com/lizabespalova) 

## Challenge Overview
Build a system that estimates inference time and/or power usage of AI models on specific hardware platforms, bridging AI, compiler design, and systems engineering.

## Testing Priority
**Models (smallest → largest):**
1. GPT-2 small (124M params)
2. GPT-2 medium (355M params)
3. Llama 3.1 8B
4. XGBoost/scikit-learn (tabular)
5. Llama 3.1 70B
6. DeepSeek-R1 8B
7. Llama 3.1 405B

**Infrastructure (fastest startup → slowest):**
1. CPU-only
2. Single GPU (24GB)
3. Single GPU (80GB)
4. AMD MI300X
5. Multi-GPU (2x, 4x+)
6. TPU v5litepod-4

## Approach
4-stream collaborative pipeline:
1. **Data Generation** - Run model tests, collect metrics
2. **Dataset Creation** - Feature engineering, EDA
3. **Model Training** - Train prediction models, compare architectures  
4. **Feature Analysis** - SHAP analysis, feature selection

## Tech Stack
- **Infrastructure**: dstack Sky
- **Data**: Parquet files via Google Drive
- **Serving**: vLLM, TGI, SGLang
