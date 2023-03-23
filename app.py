import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#create the app
app  = Flask(__name__)

#load the pickle file of the model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    if (prediction==[1]):
        return render_template("index.html", prediction_text ="the person has lung cancer")
    else:
        return render_template("index.html", prediction_text ="the person is not effected by lung cancer")

if __name__ == "__main__":
    app.run(debug="True")
    
        

    