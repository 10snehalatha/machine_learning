from flask import Flask, render_template, request,redirect,url_for,session
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

app = Flask(__name__)
app.secret_key = 'secret_key'


#Nutritional value(high or naormal) prediction based on Water Parameters


# Loading and preprocessing the dataset
df = pd.read_csv('water_analysis.csv')

# Independent variables/features from dataset
x = df.drop('nutrition_values', axis=1)

# Dependent variable from dataset
y = df['nutrition_values']

# Encoding  the target variable 'y'
le = LabelEncoder()
y = le.fit_transform(y)


@app.route('/')
def home():
    return render_template('home.html')
 
@app.route('/home')
def admin():
    #return render_template('home.html')
    return redirect ('/')
@app.route('/training')
def train():
    return render_template('training.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/water')
def water():
    return render_template('water.html')

@app.route('/training', methods=['POST'])
def training():
    selected_model = request.form['training_model']


    if selected_model == "Logistic":
        model = LogisticRegression()
    elif selected_model == "Support Vector Machine":
        model = SVC()
    elif selected_model == "Decision Tree":
        model = DecisionTreeClassifier()
    elif selected_model == "Random Forest":
        model = RandomForestClassifier()
    elif selected_model == "K Nearest Neighbours":
        model = KNeighborsClassifier()
    elif selected_model == "Gradient Boosting":
        model = GradientBoostingClassifier()
    elif selected_model == "Adaboost":
        model = AdaBoostClassifier()
    elif selected_model == "XGboost":
        model = xgb.XGBClassifier()
    else:
        return "Please select a training model from the training page."

    session['selected_model'] = selected_model

    return render_template("training.html")

@app.route('/predictwater', methods=['POST'])
def predictwater():
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)

        if 'selected_model' not in session:
             return "Please select a training model from the training page."

        selected_model = session['selected_model']

        if selected_model == "Logistic":
           model = LogisticRegression()
        elif selected_model == "Support Vector Machine":
           model = SVC()
        elif selected_model == "Decision Tree":
           model = DecisionTreeClassifier()
        elif selected_model == "Random Forest":
           model = RandomForestClassifier()
        elif selected_model == "K Nearest Neighbours":
           model = KNeighborsClassifier()
        elif selected_model == "Gradient Boosting":
           model = GradientBoostingClassifier()
        elif selected_model == "Adaboost":
           model = AdaBoostClassifier()
        elif selected_model == "XGboost":
           model = xgb.XGBClassifier()
        

        model.fit(x_train, y_train)

        pH_range = (6.5, 8.5)
        salinity_range = (0.1, 3)
        toxicity_range = (70, 141)
        alkalinity_range = (30, 60)
        conductivity_range = (0.1, 0.7)

        pH = float(request.form['pH'])
        Salinity = float(request.form['Salinity'])
        Toxicity = float(request.form['Toxicity'])
        Alkalinity = float(request.form['Alkalinity'])
        Conductivity = float(request.form['Conductivity'])
        # Performing  ML prediction

        feature_list = [[pH, Salinity, Toxicity, Alkalinity, Conductivity]]
        # Standardizing the input features
        feature_list = ss.transform(feature_list) 
        prediction = model.predict(feature_list)
        parameters = []

        # Checking which parameters will fall outside the specified range
        if not (pH_range[0] <= pH <= pH_range[1]):
           parameters.append('pH')
        if not (salinity_range[0] <= Salinity <= salinity_range[1]):
           parameters.append('Salinity')
        if not (toxicity_range[0] <= Toxicity <= toxicity_range[1]):
           parameters.append('Toxicity')
        if not (alkalinity_range[0] <= Alkalinity <= alkalinity_range[1]):
            parameters.append('Alkalinity')
        if not (conductivity_range[0] <= Conductivity <= conductivity_range[1]):
            parameters.append('Conductivity')

        if prediction == 1:
           result = f'The water of the crop has normal nutritional value based on water parameters due to {", ".join(parameters)}'
           image="poor.jpg"
   
        else:
          result = 'The water of the crop has high nutritional value based on water parameters.'
          image="rich.jpeg"

        session['water_pH'] = pH
        session['water_Salinity'] = Salinity
        session['water_Toxicity'] = Toxicity
        session['water_Alkalinity'] = Alkalinity
        session['water_Conductivity'] = Conductivity


        return render_template('result.html', prediction=result,model=selected_model)
    


@app.route('/result')
def result():
    return render_template('result.html')

#Nutritional value (high or normal) prediction based on Soil Parameters


# Loading and preprocessing the dataset
df1 = pd.read_csv('soil_data.csv')

# Independent variables/features from dataset
X = df1.drop('nutrition_values', axis=1)

# Dependent variable
Y = df1['nutrition_values']

# Encoding the target variable
lencoder = LabelEncoder()
Y = lencoder.fit_transform(Y)


@app.route('/soil')
def soil():
    return render_template('soil.html')

@app.route('/predictsoil', methods=['POST'])
def predictsoil():
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    
    # Standardizing the features
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    if 'selected_model' not in session:
             return "Please select a training model from the training page."
    
    selected_model=session['selected_model']

    # Training the models
    if selected_model == "Logistic":
        model_soil = LogisticRegression()
    elif selected_model == "Support Vector Machine":
        model_soil = SVC()
    elif selected_model == "Decision Tree":
        model_soil = DecisionTreeClassifier()
    elif selected_model == "Random Forest":
        model_soil = RandomForestClassifier()
    elif selected_model == "K Nearest Neighbours":
        model_soil = KNeighborsClassifier()
    elif selected_model == "Gradient Boosting":
        model_soil = GradientBoostingClassifier()
    elif selected_model == "Adaboost":
        model_soil = AdaBoostClassifier()
    elif selected_model == "XGboost":
        model_soil = xgb.XGBClassifier()
    

    model_soil.fit(X_train, Y_train)


    pH = float(request.form['pH'])
    oc = float(request.form['oc'])
    cec= float(request.form['cec'])
    Nitrogen = float(request.form['Nitrogen'])
    Phosphorous = float(request.form['Phosphorous'])
    Potassium = float(request.form['Potassium'])

    # Performing ML prediction

    pH_soil_range = (7.5, 14)
    oc_range = (0.30, 0.75)
    cec_range = (5, 15)
    nitrogen_range = (4, 10)
    phosphorous_range = (9, 20)
    potassium_range = (120, 172)
    parameters_soil=[]

    feature_list = [[pH, oc, cec, Nitrogen, Phosphorous, Potassium]]
    # Standardizing the input features
    feature_list = ss.transform(feature_list)  
    prediction_soil = model_soil.predict(feature_list)


    # Checking which soil parameters will fall outside the specified range
    if  (pH_soil_range[0] <= pH <= pH_soil_range[1]):
        parameters_soil.append('pH (Soil)')
    if  (oc_range[0] <= oc <= oc_range[1]):
        parameters_soil.append('Organic Carbon (OC)')
    if  (cec_range[0] <= cec <= cec_range[1]):
        parameters_soil.append('CEC')
    if  (nitrogen_range[0] <= Nitrogen <= nitrogen_range[1]):
        parameters_soil.append('Nitrogen')
    if  (phosphorous_range[0] <= Phosphorous <= phosphorous_range[1]):
        parameters_soil.append('Phosphorous')
    if  (potassium_range[0] <= Potassium <= potassium_range[1]):
        parameters_soil.append('Potassium')

    if prediction_soil == 1:
       result_soil = f'The soil of the crop has normal nutritional value based on soil parameters due to {", ".join(parameters_soil)}'
    else:
       result_soil = 'The soil of the crop has high nutritional value based on soil parameters.'

    session['soil_pH'] = pH
    session['soil_OC'] = oc
    session['soil_CEC'] = cec
    session['soil_Nitrogen'] = Nitrogen
    session['soil_Phosphorous'] = Phosphorous
    session['soil_Potassium'] = Potassium
    

    return render_template('result_soil.html', prediction_soil=result_soil,selected_model=selected_model)

@app.route('/overall')
def overall():
    return render_template('overall.html')


@app.route('/result_soil')
def result_soil():
    return render_template('result_soil.html')

# Loading and preprocessing the dataset
df2 = pd.read_csv('water_soil.csv')

# Independent variables/features from dataset
X1 = df2.drop('nutrition_values', axis=1)

# Dependent variable
Y1= df2['nutrition_values']

# Encoding the target variable
lencoder = LabelEncoder()
Y1 = lencoder.fit_transform(Y1)

@app.route('/predictoverall',methods=['POST'])
def predictoverall():
    # Retrieved the stored water and soil parameter values from the session
    
    soil_pH = session.get('soil_pH')
    soil_OC = session.get('soil_OC')
    soil_CEC = session.get('soil_CEC')
    soil_Nitrogen = session.get('soil_Nitrogen')
    soil_Phosphorous = session.get('soil_Phosphorous')
    soil_Potassium = session.get('soil_Potassium')
    water_pH = session.get('water_pH')
    water_Salinity = session.get('water_Salinity')
    water_Toxicity = session.get('water_Toxicity')
    water_Alkalinity = session.get('water_Alkalinity')
    water_Conductivity = session.get('water_Conductivity')
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.25, random_state=0)
    
    # Standardizing the features
    ss = StandardScaler()
    X1_train = ss.fit_transform(X1_train)
    X1_test = ss.transform(X1_test)

    if 'selected_model' not in session:
             return "Please select a training model from the training page."
    
    selected_model=session['selected_model']
    # Training the models
    if selected_model == "Logistic":
        model_soil_water = LogisticRegression()
    elif selected_model == "Support Vector Machine":
        model_soil_water = SVC()
    elif selected_model == "Decision Tree":
        model_soil_water = DecisionTreeClassifier()
    elif selected_model == "Random Forest":
        model_soil_water = RandomForestClassifier()
    elif selected_model == "K Nearest Neighbours":
        model_soil_water = KNeighborsClassifier()
    elif selected_model == "Gradient Boosting":
        model_soil_water = GradientBoostingClassifier()
    elif selected_model == "Adaboost":
        model_soil_water = AdaBoostClassifier()
    elif selected_model == "XGboost":
        model_soil_water = xgb.XGBClassifier()
    

    model_soil_water.fit(X1_train, Y1_train)
    features= [[soil_pH,soil_OC,soil_CEC,soil_Nitrogen,soil_Phosphorous,soil_Potassium,water_pH,water_Salinity,
                     water_Toxicity,water_Alkalinity,water_Conductivity,]]
    # Standardizing the input features
    features= ss.transform(features)  
    prediction_soil_water = model_soil_water.predict(features)
    
    if prediction_soil_water == 1:
       result_soil_water = 'The paddy crop has normal nutritional value '
       image="poor.jpg"
    else:
       result_soil_water = 'The paddy crop has high nutritional value.'
       image="rich.jpeg"

    session.pop('water_pH', None)
    session.pop('water_Salinity', None)
    session.pop('water_Toxicity', None)
    session.pop('water_Alkalinity', None)
    session.pop('water_Conductivity', None)
    session.pop('soil_pH', None)
    session.pop('soil_OC', None)
    session.pop('soil_CEC', None)
    session.pop('soil_Nitrogen', None)
    session.pop('soil_Phosphorous', None)
    session.pop('soil_Potassium', None)

    
    return render_template('result_overall.html', result_soil_water=result_soil_water, image=image,selected_model=selected_model)


if __name__ == '__main__':
    app.run(debug=True)
