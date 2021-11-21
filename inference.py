import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model
model = keras.models.load_model('Diabetes\diabetes.h5')
scaler = pickle.load(open('Diabetes\scalar.pkl', 'rb'))
class_names = ['No Diabetes','Diabetes']

def predict(df):
    df = df[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker','Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies','HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth','MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education','Income']]
    df = scaler.transform(df)
    numpy_array = df.to_numpy()
    predictions = model.predict(numpy_array)
    # output = [class_names[class_predicted] for class_predicted in predictions]
    return predictions

HighBP = 1
HighChol = 0
CholCheck = 1
bmi = 26
Smoker = 0
Stroke = 0
HeartDiseaseorAttack = 1
PhysActivity = 1
Fruits = 0
Veggies = 1
HvyAlcoholConsump = 0
AnyHealthcare = 1
NoDocbcCost = 0
GenHlth = 3
MentHlth = 5
PhysHlth = 30
DiffWalk = 0
Sex = 1
Age = 4
Education = 6
Income = 8


df = pd.DataFrame({ 
    'HighBP':[HighBP],
    'HighChol':[HighChol], 
    'CholCheck':[CholCheck], 
    'BMI':[bmi], 
    'Smoker':[Smoker],
    'Stroke':[Stroke],
    'HeartDiseaseorAttack':[HeartDiseaseorAttack],
    'PhysActivity':[PhysActivity],
    'Fruits':[Fruits],
    'Veggies':[Veggies],
    'HvyAlcoholConsump':[HvyAlcoholConsump],
    'AnyHealthcare':[AnyHealthcare],
    'NoDocbcCost':[NoDocbcCost],
    'GenHlth':[GenHlth],
    'MentHlth':[MentHlth],
    'PhysHlth':[PhysHlth],
    'DiffWalk':[DiffWalk],
    'Sex':[Sex],
    'Age':[Age],
    'Education':[Education],
    'Income':[Income],

})
print(predict(df))