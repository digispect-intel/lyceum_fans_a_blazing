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

**Data Processing**
- Worked with 2,728 real performance measurements from 11 different AI models
- Cleaned the Excel dataset, handled missing values, and spotted outliers
- The data covers everything from tiny 102K models to massive 1.5B parameter transformers
- Ensured we had good coverage across CPU and GPU configurations

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

**User Experience Focus**
- Built intuitive sliders and inputs for model configuration (layers, hidden size, etc.)
- Hardware settings automatically update when switching between CPU and GPU
- Added export functionality so users can save their optimization findings
- Error handling ensures the app doesn't break when users input edge cases

#### Technical Implementation

**Browser-Based ML Inference**
- Implemented the entire prediction pipeline in JavaScript so everything runs locally
- No server required - the app works instantly without API calls or waiting
- Custom matrix operations and feature scaling to match our Python training pipeline
- Predictions happen in under 100ms, making the interface feel truly responsive

**Performance Optimization**
- Debounced user inputs so the interface stays smooth while typing
- Efficient data structures to handle our 2,728 training samples
- Optimized rendering so charts update smoothly without flickering
- Memory management to prevent slowdowns during extended use

#### Key Results

Our final system achieves:
- 92% R² accuracy for runtime prediction
- 87% R² accuracy for energy consumption prediction  
- Sub-100ms prediction latency in the browser
- Coverage of 102K to 1.5B+ parameter models
- Support for 1-64 CPU cores and 0-80GB GPU configurations

#### What Makes This Different

Most AI benchmarking tools only tell you about models after you've already deployed them. We predict performance before you spend any money on infrastructure. The combination of runtime AND energy prediction is unique - most tools only do one or the other.

The web interface makes these predictions accessible to anyone, not just ML engineers who can run Python scripts. Product managers can explore different model configurations, infrastructure teams can plan capacity, and developers can optimize before deployment.

---

