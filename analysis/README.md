# AI Performance Predictor

*Real-time prediction of AI model runtime and power consumption*

## Team Contributions

### Akshitha Chintaguntla Raghuram - Machine Learning & Frontend Development

I focused on building the core prediction engine and creating an intuitive web interface that makes complex ML predictions accessible to everyone.

#### Machine Learning Pipeline

**Model Training & Evaluation**
- Built and compared 4 different regression models to find the best performer
- Random Forest emerged as our winner with 92% accuracy for runtime and 87% for energy prediction
- Implemented proper train/test splits and 5-fold cross-validation to ensure reliable results
- The models can predict performance across a huge range - from 100K parameter models to 1.5B+ parameter giants

**Feature Engineering Work**
- Started with 21 raw data columns and engineered them into 14 meaningful predictive features
- Added log transformations since model parameters span several orders of magnitude
- Created interaction terms like `params × layers` that capture architectural complexity
- Built hardware efficiency ratios like `parameters per CPU core` that actually matter for performance



#### Web Application Development

**Frontend Architecture**
- Built a modern single-page application using vanilla JavaScript and Vite
- Chose a clean, professional design with glassmorphism effects that actually looks good
- Made it fully responsive so it works great on laptops during presentations or phones for quick checks
- Real-time predictions update as you type - no waiting around for results

**Interactive Visualizations**
- Integrated Chart.js to show performance patterns across different model sizes
- Log-scale plots make it easy to see relationships across the huge parameter ranges we're dealing with
- Created comparison charts that clearly show CPU vs GPU performance differences
- Users can see exactly where their model configuration fits in the broader landscape








#### Key Results

Our final system achieves:
- 92% R² accuracy for runtime prediction
- 87% R² accuracy for energy consumption prediction  
- Sub-100ms prediction latency in the browser
- Coverage of 102K to 1.5B+ parameter models
- Support for 1-64 CPU cores and 0-80GB GPU configurations



