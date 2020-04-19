import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

import tensorflow as tf

data = pd.read_csv('creditcard.csv')

# =============================================================================
# EDA
# =============================================================================

# Histogram
data.hist(figsize = (20, 20))
plt.show()

# Correlation matrix
cm = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(cm, vmax = .8, square = True)
plt.show()


# =============================================================================
# Feature Engineering
# =============================================================================

# Split the dataset
X = data.drop(labels='Class',axis=1)
y = data['Class']

ss = StandardScaler()
X['Amount_scaled'] = ss.fit_transform(X['Amount'].values.reshape(-1, 1))
X = X.drop(['Time','Amount'],axis=1)


# =============================================================================
# Data balancing
# =============================================================================

# unbalanced data  0： 284315， 1： 492
y.value_counts()

# Under Sampling

X_US, y_US = NearMiss(version=2).fit_resample(X, y)
pd.Series(y_US).value_counts()

# Over Sampling

X_OS, y_OS = SMOTE().fit_resample(X, y)
pd.Series(y_OS).value_counts()


# =============================================================================
# Model Fitting Function
# =============================================================================

def Fit_Model(X, y, model, params):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
    
    grid = GridSearchCV(estimator = model, param_grid = params, cv = 10, 
                        scoring = 'accuracy', verbose = 1, n_jobs = -1)
    
    grid_result = grid.fit(X_train, y_train)
    best_params = grid_result.best_params_
    pred = grid_result.predict(X_test)
    cm = confusion_matrix(y_test, pred)
   
    print('Best Params :', best_params)
    print('Classification Report :', classification_report(y_test,pred))
    print('Accuracy Score : ' + str(accuracy_score(y_test,pred)))
    print('Confusion Matrix : \n', cm)


# =============================================================================
# Build Models
# =============================================================================

# Logistic Regression

param ={'C': np.logspace(0, 3, 10), 'penalty': ['l1', 'l2', 'elasticnet'] , 'solver' : ['saga']}

Fit_Model(X_US, y_US, LogisticRegression(), param)

Fit_Model(X_OS, y_OS, LogisticRegression(), param)


# SVC 

param ={'C': [0.1, 1, 100, 1000], 'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]}

Fit_Model(X_US, y_US, SVC(), param)

Fit_Model(X_OS, y_OS, SVC(), param)


# Random Forest

param ={'n_estimators': [100, 500, 1000, 2000]}

Fit_Model(X_US, y_US, RandomForestClassifier(), param)

Fit_Model(X_OS, y_OS, RandomForestClassifier(), param)


# XgBoost

param ={'n_estimators': [100, 500, 1000, 2000]}

Fit_Model(X_US, y_US, XGBClassifier(), param)

Fit_Model(X_OS, y_OS, XGBClassifier(), param)


# Neural Network

def Fit_NN(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
    
    # tf model does not accept dataframe
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(2),
                tf.keras.layers.Softmax()
            ])
    
    num_epochs = 5
    batch_size = 200
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])
    
    model.fit(X_train, y_train, batch_size = batch_size, epochs = num_epochs)
    
    print('\nEvaluate on test set:')
    accuracy = model.evaluate(X_test, y_test)
    print(accuracy)


Fit_NN(X_US, y_US)

# test accuracy: 0.9976
Fit_NN(X_OS, y_OS)









