import pickle
import pandas as pd
from tensorflow import keras
model = keras.models.load_model('diabetes.h5')
scaler = pickle.load(open('scalar.pkl', 'rb'))
class_names = ['No Diabetes','Diabetes']

def predict(df):
    df = df[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker','Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies','HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth','MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education','Income']]
    df = scaler.transform(df)
    predictions = model.predict(df)
    y_pred = list()
    for i in predictions:
      if i>0.5:
        y_pred.append(1)
      else:
        y_pred.append(0)
    output = [class_names[class_predicted] for class_predicted in y_pred]
    return output
