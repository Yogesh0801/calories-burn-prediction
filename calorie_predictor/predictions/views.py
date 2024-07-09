from django.shortcuts import render

# Create your views here.
# predictions/views.py
from django.shortcuts import render
from .forms import PredictionForm
import pickle
import pandas as pd

def load_model():
    with open('pipeline.pkl', 'rb') as f:
        return pickle.load(f)

pipeline = load_model()

def predict_calories(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            gender = form.cleaned_data['gender']
            age = form.cleaned_data['age']
            height = form.cleaned_data['height']
            weight = form.cleaned_data['weight']
            duration = form.cleaned_data['duration']
            heart_rate = form.cleaned_data['heart_rate']
            body_temp = form.cleaned_data['body_temp']
            
            sample = pd.DataFrame({
                'Gender': [gender],
                'Age': [age],
                'Height': [height],
                'Weight': [weight],
                'Duration': [duration],
                'Heart_Rate': [heart_rate],
                'Body_Temp': [body_temp],
            }, index=[0])
            
            result = pipeline.predict(sample)
            prediction = result[0]
            
            return render(request, 'predictions/result.html', {'prediction': prediction})
    else:
        form = PredictionForm()

    return render(request, 'predictions/predict.html', {'form': form})
