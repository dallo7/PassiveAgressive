from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import joblib

#load our model
model = joblib.load("psag")

#make our Flask api...
app = Flask(__name__, template_folder = "template" )

server = app.server

#form urls/...
@app.route("/")
def home():
    return render_template("psag.html")
    
#Accepting form data
@app.route("/prediction", methods = ["POST"])
def prediction():
    q = request.form["day"] 
    w = request.form["area"]
    e = request.form["time_of_day"]
    t = request.form["weather"]

# Creating Out of sample instance
    damp = {"day":q,  "Area":w, "time_of_day":e, 'weather':t} 

    test = pd.DataFrame(damp, index = [289]) # converting the dictonary into Pandas dataframe

    result = model.predict(test) #parsing it to our model for predicting

    result = (round(result[0], 2))
    
#return prediction, which is the result result.
    return render_template("psag.html", pred = "Based on the inputs the estimated price is: {}".format(result)) 
    
# Run this file as the main file...
if __name__ == "__main__": 
    app.run(debug = True)
   

            