import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from Logging.setup_logger import setup_logger_
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
logger = setup_logger_("aplication_logs","Logs.log")

@app.route('/')
def home():
    logger.info("Rendering front UI")
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [int(x) for x in request.form.values()]
    logger.info("feature values loading completed")
    final_features = [np.array(features)]
    logger.info("features converted into array")
    model = pickle.load(open('model.pkl', 'rb'))
    logger.info("Model is loaded")

    prediction = model.predict(final_features)
    logger.info(f'Predictions are {prediction}')

    output = round(prediction[0], 1)
    df = pd.DataFrame()
    df["Ratings"] = prediction
    df.to_csv("Pred_results.csv",index=False)

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)