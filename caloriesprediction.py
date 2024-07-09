from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import accuracy_score
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import pickle
import pandas as pd
from tkinter import *
f=open('pipeline.pkl','rb')
pipeline = pickle.load(f)

main = Tk()
main.title("IDENTIFING OF CALORIES BURN PREDICTION")
main.geometry("1300x1200")
main.config(bg="lightgreen")

global filename
global X, Y
global X_train, X_test, y_train, y_test
global accuracy
global dataset
global model
#global pipeline


def loadProfileDataset():    
    global filename
    global dataset
    outputarea.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    outputarea.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    outputarea.insert(END,str(dataset.head()))

def preprocessDataset():
    global X, Y
    global dataset
    global X_train, X_test, y_train, y_test
    outputarea.delete('1.0', END)
    X = dataset.values[:, 0:8] 
    Y = dataset.values[:, 8]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    outputarea.insert(END,"\n\nDataset contains total Users : "+str(len(X))+"\n")
    outputarea.insert(END,"Total profiles used to train  : "+str(len(X_train))+"\n")
    outputarea.insert(END,"Total profiles used to test   : "+str(len(X_test))+"\n")

def show_entry():
    #f = open('pipeline.pkl','rb')
    #pipeline = pickle.load(f)
    p1 = str(clicked.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    p7 = float(e7.get())

    sample = pd.DataFrame({
    'Gender':[p1],
    'Age':[p2],
    'Height':[p3],
    'Weight':[p4],
    'Duration':[p5],
    'Heart_Rate':[p6],
    'Body_Temp':[p7],
    },index=[0])

    result = pipeline.predict(sample)
    # print(result)
    Label(main, text="Amount of Calories Burnt").place(x=200,y=600)
    Label(main, text=result[0]).place(x=200,y=700)
    outputarea.insert(END,result)


clicked = StringVar()
options = ['male', 'female']

e1 = OptionMenu(main , clicked , *options )
e1.configure(width=15)
e2 = Entry(main)
e3 = Entry(main)
e4 = Entry(main)
e5 = Entry(main)
e6 = Entry(main)
e7 = Entry(main)


e1.place(x=200,y=100)
e2.place(x=200,y=150)
e3.place(x=200,y=200)
e4.place(x=200,y=250)
e5.place(x=200,y=300)
e6.place(x=200,y=350)
e7.place(x=200,y=400)
        
def close():
    main.destroy()




font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload calories burn Dataset", command=loadProfileDataset)
uploadButton.place(x=300,y=20)
uploadButton.config(font=ff)

processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
processButton.place(x=600,y=20)
processButton.config(font=ff)


predictButton = Button(main, text="Predict Calories burn", command=show_entry)
predictButton.place(x=800,y=20)
predictButton.config(font=ff)

Label(main,text = "Select Gender").place(x=80,y=100)
Label(main,text = "Enter Your Age").place(x=80,y=150)
Label(main,text = "Enter Your Height").place(x=80,y=200)
Label(main,text = "Enter Your Weight").place(x=80,y=250)
Label(main,text = "Duration").place(x=80,y=300)
Label(main,text = "Heart Rate").place(x=80,y=350)
Label(main,text = "Body Temp").place(x=80,y=400)

Button(main,text="Predict",command=show_entry).place(x=200,y=500)

exitButton = Button(main, text="Logout", command=close)
exitButton.place(x=1000,y=20)
exitButton.config(font=ff)

font1 = ('times', 12, 'bold')
outputarea = Text(main,height=30,width=85)
scroll = Scrollbar(outputarea)
outputarea.configure(yscrollcommand=scroll.set)
outputarea.place(x=400,y=100)
outputarea.config(font=font1)


main.config()
main.mainloop()
