# predictions/forms.py
from django import forms

class PredictionForm(forms.Form):
    GENDER_CHOICES = [('male', 'Male'), ('female', 'Female')]
    
    gender = forms.ChoiceField(choices=GENDER_CHOICES)
    age = forms.FloatField()
    height = forms.FloatField()
    weight = forms.FloatField()
    duration = forms.FloatField()
    heart_rate = forms.FloatField()
    body_temp = forms.FloatField()
