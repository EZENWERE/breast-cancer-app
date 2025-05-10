# Breast Cancer Predictor

A web application that predicts whether a breast mass is benign or malignant based on cell nuclei measurements.

## Features

- Uses a Random Forest machine learning model
- Input form for entering breast cell measurements
- Instant prediction results
- Responsive web design

## Technical Details

- **Backend**: Flask (Python)
- **ML Algorithm**: Random Forest Classifier
- **Deployment**: Render.com
- **Dataset**: Wisconsin Breast Cancer dataset (built into scikit-learn)

## How to Run Locally

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Open your browser and navigate to `http://localhost:5000`

## Deploying to Render

This application is designed to be easily deployed on Render.com:

1. Push your code to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
5. Deploy

## Project Structure

```
breast-cancer-predictor/
├── app.py                  # Flask application
├── breast_cancer_model.pkl # Trained ML model (created on first run if missing)
├── features.pkl            # Feature names (created on first run if missing)
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates
│   └── index.html          # Main page template
└── Procfile                # For deployment
```

## About the Model

The prediction model uses the Random Forest algorithm trained on the Wisconsin Breast Cancer dataset. It analyzes various measurements of cell nuclei to determine if a mass is likely benign or malignant.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.