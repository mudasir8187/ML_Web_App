import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Header
st.header("Machine learing web app")
st.write("This app will explore different Data sets on ml models")


with st.sidebar.header('Upload your CSV file'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("[Example CSV files](https://www.kaggle.com)")

if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    data = load_csv()
    st.write(data.head())

    st.write("Shape of Data Set: ",data.shape)


    with st.sidebar.header("Select Ml Model"):
        model = st.sidebar.selectbox("Select a Machine Learning Model", ("SVM",'KNN', "Random Forest Classifier"))

    # ' Selecting model parametrers'

    st.sidebar.write("## Model Parameters")
    params = dict()
    if model == "SVM":
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif model == "KNN":
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators


    if model == "SVM":
        clf = SVC(C=params['C'])
    elif model == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)


    # ' Splitting the data set'
    x,y = data.iloc[:,:-1], data.iloc[:,-1]

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)


    # ' Fitting the model'
    clf.fit(x_train, y_train)

    # ' Predicting the model'
    y_pred = clf.predict(x_test)

    # ' Accuracy of the model'
    acc = accuracy_score(y_test, y_pred)

    st.write(f"Model: ", model)
    st.write(f"Model Accuracy: {acc}")

    pca = PCA(2)

    x_project = pca.fit_transform(x)

    x1 = x_project[:, 0]
    x2 = x_project[:,1]


    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(y)

    fig = plt.figure(figsize=(5, 3))

    plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")

    plt.colorbar()

    st.pyplot(plt)


else:
    st.info("Please upload Data Set")
    st.subheader("Note:")
    st.markdown("Columns of Data Set should be numeric except the last one to perform machine"
    "learning algorithms. Otherwise ML models doesn't work.")