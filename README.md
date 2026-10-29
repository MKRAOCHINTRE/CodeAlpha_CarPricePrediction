
#  Car Price Prediction

A Machine Learning project developed as part of the CodeAlpha Data Science Internship.

This project predicts the estimated selling price of a used car based on vehicle details such as car model, present price, kilometers driven, fuel type, transmission, seller type, previous owners, and vehicle age.

## Features

- Interactive web interface built with Streamlit
- Predicts estimated used-car selling prices
- Accepts multiple vehicle attributes
- Includes preprocessing in the saved machine learning pipeline
- Displays predictions in lakh rupees and approximate rupees
- Handles prediction errors and invalid results

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- Google Colab
- Visual Studio Code

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Check missing values and duplicate records.
3. Perform exploratory data analysis.
4. Prepare numerical and categorical features.
5. Split the data into training and testing sets.
6. Train and compare regression models.
7. Evaluate model performance using MAE, RMSE, and R².
8. Save the trained model pipeline.
9. Build an interactive Streamlit application.

## Model Evaluation

The selected model is Gradient Boosting Regressor.

| Metric | Test Result |
|---|---:|
| Mean Absolute Error (MAE) | 1.178 |
| Root Mean Squared Error (RMSE) | 2.688 |
| R² Score | 0.720 |

These results are from the current held-out test split. Performance may vary on new data.

## Project Structure

```text
CodeAlpha_CarPricePrediction/
├── app.py
├── car_price_model.pkl
├── car data.csv
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CodeAlpha_CarPricePrediction
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Dataset

The project uses a used-car dataset containing vehicle specifications and historical selling prices.

The dataset should be used in accordance with its original source and license.

## Limitations

- The dataset contains historical records ending in 2018.
- Predictions are educational estimates and do not represent current market valuations.
- The dataset is relatively small, so performance may vary on unseen vehicles.
- The model should be retrained and evaluated using a larger, more recent dataset before practical use.

## Future Improvements

- Train and evaluate the model using recent market data.
- Expand the dataset to cover more car models.
- Improve generalization through further validation and feature engineering.
- Deploy the application online.
- Add visualizations and prediction insights.

## Author

Developed for the CodeAlpha Data Science Internship.

## Disclaimer

This project is for educational purposes only. Predicted prices may differ significantly from actual selling prices.
