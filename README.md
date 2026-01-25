# PM2.5 Air Quality Forecasting with LSTM Networks

## Overview
This project focuses on forecasting **PM2.5 air pollution concentrations in Beijing** using historical time-series data. It demonstrates how **time-aware preprocessing, sliding window sequencing, and systematic experimentation** can significantly improve forecasting accuracy for environmental data.

Rather than relying solely on increasing model complexity, the project emphasizes **data preprocessing, temporal structure, and controlled hyperparameter tuning** as the primary drivers of performance improvement.

The best-performing model achieved a **public Kaggle leaderboard RMSE of 3638.40**, placing it within the **Proficient** performance range according to the evaluation rubric.

## Objectives
- Predict future PM2.5 concentrations using historical observations  
- Capture temporal dependencies using **LSTM based recurrent neural networks**  
- Evaluate the impact of **window size, model capacity, and regularization**  
- Produce a **reproducible and well-documented** machine learning workflow  

## Key Features and Contributions

### Data-Centric Pipeline
- Time-series–aware preprocessing to preserve temporal order  
- Sliding window sequencing for supervised learning  
- Time respecting train/validation splits to prevent data leakage  

### Robust Preprocessing
- **Missing values:** Forward fill, backward fill, and interpolation  
- **Scaling:** Feature normalization to stabilize neural network training  
- **Temporal context:** Explicit sequence windowing 12–96 timesteps evaluated  

### Model Architecture
- **LSTM based recurrent neural networks**  
- **Dropout regularization** to mitigate overfitting  
- **Dense layers** for nonlinear feature transformation  

### Systematic Experimentation
- Conducted **15+ controlled experiments**  
- Hyperparameters varied:
  - Window size  
  - Number of LSTM units  
  - Dense layer size  
  - Learning rate  
  - Dropout rate  
  - Batch size  
- Results logged and compared using **validation RMSE**

## Results and Performance

| Experiment | Window | LSTM Units | Dense Units | Dropout | Batch Size | Validation RMSE |
|----------|--------|------------|-------------|---------|------------|----------------|
| E10 | 48 | 128 | 64 | 0.2 | 128 | 80.97 |
| E13 | 48 | 128 | 64 | 0.2 | 128 | 82.53 |
| E15 | 96 | 64 | 32 | 0.1 | 128 | 83.97 |
| E16 | 48 | 64 | 64 | 0.3 | 128 | 84.49 |

- **Best Kaggle public leaderboard RMSE:** 3638.40  
- **Subsequent submission RMSE:** 3961.18  

The variation between submissions reflects the **stochastic nature of neural network optimization** and sensitivity to random initialization and early stopping behavior.

## Final Model Architecture

```python
Sequential([
    LSTM(128, return_sequences=True, dropout=0.2, input_shape=(window, features)),
    LSTM(64, dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

## Design Rationale

- **Stacked LSTMs:** Capture short- and medium-term temporal dependencies in PM2.5 concentration patterns  
- **Moderate window size (48):** Provides the best balance between temporal context and training stability  
- **Dropout regularization:** Controls overfitting without inducing underfitting  
- **Adam optimizer:** Ensures stable and efficient convergence for time-series regression  

---
```
## Project Structure

```text
air-quality-forecasting/
├── notebooks/
│   └── air_quality_forecasting.ipynb
├── outputs/
│   ├── experiment_results.csv
│   └── submission.csv
├── README.md
└── requirements.txt
```
#Installation and Usage


1. Clone the Repository
   
git clone https://github.com/Solomon-the-octave/air-quality-forecasting
cd air-quality-forecasting

3. Install Dependencies
   
pip install -r requirements.txt

4. Run the Notebook

Open air_quality_forecasting.ipynb in Google Colab or Jupyter Notebook and execute the cells sequentially to reproduce:

Data loading and preprocessing

Feature sequencing using sliding windows

Model training and validation

Prediction generation and Kaggle submission

Future Work

Explore Bidirectional LSTM and attention-based architectures

Integrate external meteorological variables (e.g., temperature, wind speed)

Engineer lag features and rolling statistics

Apply Bayesian hyperparameter optimization

Extend to multi-step forecasting horizons

Author

Wengelawit Ayalew Solomon
Machine Learning and Data Science Student
African Leadership University

GitHub: https://github.com/Solomon-the-octave
