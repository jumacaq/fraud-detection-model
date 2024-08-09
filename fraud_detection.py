import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import timeit
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import os



st.title('Detección de Fraude :bank:')
st.write('Aplicación para evaluar el mejor modelo de aprendizaje automático supervisado para detectar transacciones fraudulentas')
#Loading the Data
# Load the dataset from the csv file using pandas
url = 'https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv'
df=st.cache_data(pd.read_csv)(url)
df = df.sample(frac=0.1, random_state = 48)


# Print shape and description of the data
if st.sidebar.checkbox('Mostrar el dataframe'):
    st.write(df.head(100))
    st.write('Tamaño del dataframe: ',df.shape)
    st.write('Descripción del dataframe : \n',df.describe())
# Print valid and fraud transactions
fraud=df[df.Class==1]
valid=df[df.Class==0]
outlier_percentage=(df.Class.value_counts()[1]/df.Class.value_counts()[0])*100
if st.sidebar.checkbox('Mostrar el detalle de transacciones normales y fraudulentas'):
    st.write('Porcentaje de transacciones fraudulentas: %.3f%%'%outlier_percentage)
    st.write('Transacciones Fraudulentas: ',len(fraud))
    st.write('Transacciones Normales: ',len(valid))
 
#Training and Testing Data Split   
#Obtaining X (features) and y (labels)
X=df.drop(['Class'], axis=1)
y=df.Class
#Split the data into training and testing sets
from sklearn.model_selection import train_test_split
size = st.sidebar.slider('Tamaño del dataframe de prueba', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 42)
#Print shape of train and test sets
if st.sidebar.checkbox('Mostrar el tamaño del dataframe de entrenamiento y prueba'):
    st.write('X_train: ',X_train.shape)
    st.write('y_train: ',y_train.shape)
    st.write('X_test: ',X_test.shape)
    st.write('y_test: ',y_test.shape)
 
#Building the Model   
#Import classification models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
logreg=LogisticRegression()
svm=SVC()
knn=KNeighborsClassifier()
etree=ExtraTreesClassifier(random_state=42)
rforest=RandomForestClassifier(random_state=42)
features=X_train.columns.tolist()
    
#Feature Selection
#Feature selection through feature importance
@st.cache_data
#def feature_sort(model,X_train,y_train):@st.cache_data
def feature_sort(_model, X_train,y_train):
    #feature selection
    mod=model
    # fit the model
    mod.fit(X_train, y_train)
    # get importance
    imp = mod.feature_importances_
    return imp
#Classifiers for feature importance
clf=['Extra Trees','Random Forest']
mod_feature = st.sidebar.selectbox('¿Que modelo por importancia de variables?', clf)
start_time = timeit.default_timer()
if mod_feature=='Extra Trees':
    model=etree
    importance=feature_sort(model,X_train,y_train)
elif mod_feature=='Random Forest':
    model=rforest
    importance=feature_sort(model,X_train,y_train)
elapsed = timeit.default_timer() - start_time
#st.write('Tiempo de ejecución para determinar importancias de de variables: %.2f minutos'%(elapsed/60))
#Plot of feature importance

if st.sidebar.checkbox('Gráfico de importancia de variables'):
    fig, ax = plt.subplots()
    ax.bar([x for x in range(len(importance))], importance)
    ax.set_title('Importancia de variables')
    ax.set_xlabel('Variable')
    ax.set_ylabel('Importancia')
    st.pyplot(fig)
    
#Model Training and Performance
feature_imp=list(zip(features,importance))
feature_sort=sorted(feature_imp, key = lambda x: x[1])

n_top_features = st.sidebar.slider('Número de variables importantes a utilizar: ', min_value=5, max_value=20)

top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

if st.sidebar.checkbox('Mostrar variables seleccionadas'):
    st.write('Principales %d variables en orden de importancia son: %s'%(n_top_features,top_features[::-1]))

X_train_sfs=X_train[top_features]
X_test_sfs=X_test[top_features]

X_train_sfs_scaled=X_train_sfs
X_test_sfs_scaled=X_test_sfs

#Import performance metrics, imbalanced rectifiers
from sklearn.metrics import  confusion_matrix,recall_score,classification_report,precision_score,f1_score#matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
np.random.seed(42) #for reproducibility since SMOTE and Near Miss use randomizations
smt = SMOTE()
nr = NearMiss()
def compute_performance(model, X_train, y_train,X_test,y_test):
    start_time = timeit.default_timer()
    st.write('''
    * Un valor de 1 (o 100 %) para cualquiera de estas métricas indica un rendimiento perfecto.
    * Un valor de 0 indica el peor rendimiento posible.
    * Los valores intermedios reflejan distintos niveles de rendimiento; los valores más altos suelen ser mejores.
    ''')
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    'Accuracy: ',scores
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    'Confusion Matrix: ',cm
    rs=recall_score(y_test,y_pred)
    'Recall Score: ',rs
    ps=precision_score(y_test,y_pred)
    'Precision Score: ',ps
    fs=f1_score(y_test,y_pred)
    'F1 Score:',fs
    #cr=classification_report(y_test, y_pred)
    #'Classification Report: ',cr
    #mcc= matthews_corrcoef(y_test, y_pred)
    #'Matthews Correlation Coefficient: ',mcc
#    elapsed = timeit.default_timer() - start_time
#    'Execution Time for performance computation: %.2f minutes'%(elapsed/60)

if st.sidebar.checkbox('Ejecutar un modelo de detección de fraudes con tarjetas de crédito'):
    
    alg=['Extra Trees','Random Forest','k Nearest Neighbor','Support Vector Machine','Logistic Regression']
    classifier = st.sidebar.selectbox('Algoritmo a utilizar', alg)
    rectifier=['SMOTE','Near Miss','Sin rectificador']
    imb_rect = st.sidebar.selectbox('Rectificador de datos desequilibrados', rectifier) 
    
    if classifier=='Logistic Regression':
        model=logreg
        if imb_rect=='Sin rectificador':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Tamaño del dataset de entrenamiento desbalanceado: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Tamaño del dataset de entrenamiento balanceado: ',np.bincount(y_train_bal))
                st.write('Tamaño del dataset de prueba: ', np.bincount(y_test))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Tamaño del dataset de entrenamiento desbalanceado: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
            st.write('Tamaño del dataset de entrenamiento balanceado: ',np.bincount(y_train_bal))
            st.write('Tamaño del dataset de prueba: ', np.bincount(y_test))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
    
        
    elif classifier == 'k Nearest Neighbor':
        model=knn
        if imb_rect=='Sin rectificador':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Tamaño del dataset de entrenamiento desbalanceado: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Tamaño del dataset de entrenamiento balanceado: ',np.bincount(y_train_bal))
                st.write('Tamaño del dataset de prueba: ', np.bincount(y_test))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Tamaño del dataset de entrenamiento desbalanceado: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
            st.write('Tamaño del dataset de entrenamiento balanceado: ',np.bincount(y_train_bal))
            st.write('Tamaño del dataset de prueba: ', np.bincount(y_test))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)    
    
    elif classifier == 'Support Vector Machine':
        model=svm
        if imb_rect=='Sin rectificador':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Tamaño del dataset de entrenamiento desbalanceado: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Tamaño del dataset de entrenamiento balanceado: ',np.bincount(y_train_bal))
                st.write('Tamaño del dataset de prueba: ', np.bincount(y_test))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Tamaño del dataset de entrenamiento desbalanceado: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
            st.write('Tamaño del dataset de entrenamiento balanceado: ',np.bincount(y_train_bal))
            st.write('Tamaño del dataset de prueba: ', np.bincount(y_test))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)    
        
    elif classifier == 'Random Forest':
        model=rforest
        if imb_rect=='Sin rectificador':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Tamaño del dataset de entrenamiento desbalanceado: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Tamaño del dataset de entrenamiento balanceado: ',np.bincount(y_train_bal))
                st.write('Tamaño del dataset de prueba: ', np.bincount(y_test))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Tamaño del dataset de entrenamiento desbalanceado: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
            st.write('Tamaño del dataset de entrenamiento balanceado: ',np.bincount(y_train_bal))
            st.write('Tamaño del dataset de prueba: ', np.bincount(y_test))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)  
            
    elif classifier == 'Extra Trees':
        model=etree
        if imb_rect=='Sin rectificador':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Tamaño del dataset de entrenamiento desbalanceado: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Tamaño del dataset de entrenamiento balanceado: ',np.bincount(y_train_bal))
                st.write('Tamaño del dataset de prueba: ', np.bincount(y_test))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Tamaño del dataset de entrenamiento desbalanceado: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
            st.write('Tamaño del dataset de entrenamiento balanceado: ',np.bincount(y_train_bal))
            st.write('Tamaño del dataset de prueba: ', np.bincount(y_test))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
