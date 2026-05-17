
# F1 Qualifying Lap Time Predictor

Predict Formula 1 qualifying lap times using machine learning and real F1 data.  
This project lets you simulate a race engineer: choose tyres, weather, driver, and see what your qualifying lap could be!

How To Use
**1. Clone the Repository**

Open your terminal and type:

git clone https://github.com/itsmenandhu3/F1-engineer

Then move into the project folder:

cd your-repo-name

**2. Install Requirements**

Make sure Python is installed.  
In your terminal, install the packages with this command:

pip install -r requirements.txt

If 'pip' does not work, try:

python -m pip install -r requirements.txt

**3. Create the Cache Folder**

This step is needed only once.  
In your terminal, create a new folder called 'cache' with:

mkdir cache

**4. Train the Model**

To download the F1 qualifying data and train the machine learning model, run:

python train_model.py

Wait for the process to finish.  
After this, the model files are ready for prediction.

**5. Run the Web App**

Start the Streamlit app by running:

streamlit run streamlit_app.py

Check your terminal for a link like http://localhost:8501 and open it in your browser.

**6. Try Your Own Predictions**

On the web page, pick driver, tyre, lap number, air and track temperatures.  
You will see a qualifying lap time prediction in minutes.seconds.milliseconds format—just like F1 live timing!

## About the Project

- Uses real F1 qualifying data (2024 season) and weather information
- Trains a machine learning model (Random Forest) to predict lap times
- Verified predictions with real F1 times (like on f1-tempo.com) so it’s close to the real thing!
- No data science experience needed—just follow the steps above

## Folder Contents

- train_model.py : script to download data and train the model
- streamlit_app.py : web app for predictions
- requirements.txt : list of packages
- model files : model.pkl, le_driver.pkl, le_compound.pkl save after training
- cache/ : folder used for storing downloaded race data (mandatory, create if missing)

## Support

If you have questions or trouble setting it up, feel free to open an issue on GitHub or contact me.  
You can also suggest new features!

**Enjoy exploring Formula 1 from the engineer’s side!**

