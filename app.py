from flask import Flask , request , render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ["POST"])
def predict():
    if request.method =='POST':
        transaction_date = float(request.form['transaction_date'])
        houseage = float(request.form['houseage'])
        distance_to_the_nearest_MRT_station =float(request.form['distance_to_the_nearest_MRT_station'])
        number_of_convenience_stores = float(request.form['number_of_convenience_stores'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        prediction = model.predict([[transaction_date ,houseage, distance_to_the_nearest_MRT_station,number_of_convenience_stores, latitude , longitude,]])
        pred = prediction
        
        return render_template('index.html',results = pred)

if __name__ == '__main__':
    app.run(debug=True)