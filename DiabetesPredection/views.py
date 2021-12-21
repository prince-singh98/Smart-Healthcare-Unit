from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def home(request):
    return render(request, "home.html")


def prediction_function_diabetes():
    data = pd.read_csv(r"C:\Users\PRINCE\Desktop\Smart Healthcare Unit\diabetes_data.csv")

    x = data.drop(columns=["Outcome"], axis=1)
    y = data['Outcome']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # training the model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model


def prediction_function_heart():
    heart_data = pd.read_csv(r"C:\Users\PRINCE\Desktop\Smart Healthcare Unit\heart_disease_data.csv")

    X = heart_data.drop(columns=['target'], axis=1)
    Y = heart_data['target']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # training the model
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    return model


def diabetes_predict(request):
    return render(request, "diabetes_predict.html")


def heart_predict(request):
    return render(request, "heart_predict.html")


def diabetes_result(request):
    # to read the data from the input fields
    val1 = int(request.POST['pregnancy'])
    val2 = int(request.POST['glucose'])
    val3 = int(request.POST['bp'])
    val4 = int(request.POST['skt'])
    val5 = int(request.POST['insulin'])
    val6 = float(request.POST['bmi'])
    val7 = float(request.POST['dpf'])
    val8 = int(request.POST['age'])

    model = prediction_function_diabetes();

    # Prediction variable will store the result of our model
    prediction = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    if prediction == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request, "diabetes_predict.html", ({"result2": result1, "preg": val1, "glu": val2, "bp": val3,
                                                      "skt": val4, "ins": val5, "bmi": val6, "dpf": val7, "age": val8}))


def heart_result(request):
    # to read the data from the input fields
    val1 = int(request.POST['age'])
    val2 = int(request.POST['sex'])
    val3 = int(request.POST['cp'])
    val4 = int(request.POST['trestbps'])
    val5 = int(request.POST['chol'])
    val6 = int(request.POST['fbs'])
    val7 = int(request.POST['restecg'])
    val8 = int(request.POST['thalach'])
    val9 = int(request.POST['exang'])
    val10 = float(request.POST['oldpeak'])

    model = prediction_function_heart()

    # Prediction variable will store the result of our model
    prediction = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10]])

    if prediction[0] == 1:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request, "heart_predict.html", (
    {"result2": result1, "sex": val2, "age": val1, "cp": val3, "trestbps": val4, "chol": val5, "fbs": val6,
     "restecg": val7, "thalach": val8, "exang": val9, "oldpeak": val10}))