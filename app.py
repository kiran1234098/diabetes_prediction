from flask import Flask, render_template, request,jsonify
import os
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def hello_world():
    if request.method == 'GET':

        return render_template('index.html')
    else:

        data=CustomData(
        Pregnancies = float(request.form.get('Pregnancies')),
        Glucose = float(request.form.get('Glucose')),
        BloodPressure = float(request.form.get('BloodPressure')),
        SkinThickness = float(request.form.get('SkinThickness')),
        Insulin = float(request.form.get('Insulin')),
        BMI = float(request.form.get('BMI')),
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction')),
        Age = float(request.form.get('Age')),
        
        )
        
        final_new_data=data.get_data_as_dataframe()
        print(final_new_data)
        predict_pipeline=PredictPipeline()
        model_path = os.path.join('artifacts', 'model.pkl')
       
        result =predict_pipeline.predict(features=final_new_data, model_path=model_path)
        
        if (result==0):
              results="Person is not diabetic"
              print("Person is not diabetic")
        else:
               results ="Person is diabetic"
               print("Person is diabetic")

        return render_template('result.html' ,results=results)   



if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)