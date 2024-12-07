import uvicorn
from fastapi import FastAPI
from bankdata import bankdata
import numpy as np
import pickle
import pandas as pd

# create an app object
app = FastAPI()
pick_in = open('C:\Kalyani Pakhale\data-science-Interview\FAST-API\model.pkl', 'rb')
model = pickle.load(pick_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message':'Hello World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name:str):
    return {'Welcome to Data Science Interview Preparation' : f"{name}"}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data: bankdata):
    data = data.dict()
    print(data)
    print('hello')
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy =  data['entropy']
    print(model.predict([variance, skewness, curtosis, entropy]))
    print('hello...')
    prediction = model.predict([variance, skewness, curtosis, entropy])
    if prediction > 0.5:
        prediction = 'Fake Note'
    else:
        prediction = 'Bank Note'
        return{
            prediction:prediction
        }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)