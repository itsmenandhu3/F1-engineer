F1 Qualifying Lap Time Predictor

Predicting Formula 1 Qualifying Lap Times Using Machine Learning

Project Overview

This project simulates the role of a Formula 1 race engineer by building a machine learning model to predict qualifying lap times based on real-world F1 data. Using the 2024 Monaco Grand Prix qualifying session data, the model learns how driver, tyre compound, lap number, and weather conditions affect lap time performance.

The goal is to enable interactive lap time prediction via a Streamlit web app where users can input factors like driver, lap, tyre, and temperature to get realistic qualifying lap time estimates in a readable minutes.seconds.milliseconds format.

Features
Real F1 Data: Utilizes FastF1 Python library to fetch official qualifying session data and weather information.

Best Lap Selection: Uses each driver’s best qualifying lap for training, focusing on pole-position level performance.

Machine Learning Model: Trains a Random Forest regressor to accurately predict lap times based on key features.

Interactive Web App: Streamlit interface for easy input and instant prediction visualization.

Real-World Validation: Predictions verified against live timing data from F1 Tempo, showing close alignment with actual qualifying times.

Installation & Setup
Clone the repository:

bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install dependencies:

bash
pip install -r requirements.txt
Create the cache folder (needed by FastF1):

bash
mkdir cache
Usage
Train the Model
Download qualifying data, train the model, and save artifacts:

bash
python train_model.py
Run the Streamlit App
Launch the interactive web app:

bash
streamlit run streamlit_app.py
Open the provided local URL in your browser. Adjust inputs such as driver, tyre compound, lap number, and temperatures to see predicted qualifying lap times.

Project Structure
text
├── train_model.py          # Script to fetch data, train model, and save output
├── streamlit_app.py        # Streamlit app for input and lap time prediction
├── requirements.txt        # Project dependencies
├── model.pkl               # Trained Random Forest model file
├── le_driver.pkl           # Driver label encoder
├── le_compound.pkl         # Tyre compound label encoder
├── cache/                  # FastF1 cache directory for data caching
└── README.md               # This file
How It Works
The script loads data for the Monaco 2024 qualifying session.

Extracts each driver’s fastest lap and weather-related features.

Converts categorical data into numeric labels.

Trains a Random Forest model to predict lap time in milliseconds.

Saves the model and encoding tools for use in the app.

The Streamlit app loads this model and lets you predict lap times by simulating race engineer input.

Validation
I verified the model’s predicted lap times against live qualifying lap times from F1 Tempo, and the predictions are impressively close to official times — giving confidence in the model’s accuracy.

Technologies Used
Python 3.x

FastF1 for F1 telemetry and data retrieval

Scikit-learn for machine learning models

Streamlit for web app interface

Pandas, NumPy for data manipulation

Joblib for model serialization

About Me
I’m an avid Formula 1 enthusiast who loves combining passion for motorsport with machine learning to build insightful, interactive tools that mirror real team strategies.
