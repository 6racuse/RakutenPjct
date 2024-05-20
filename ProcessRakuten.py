import header
import numpy as np
import pandas as pd

import scipy.io
import random
import math
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,mean_squared_error,accuracy_score,f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.ensemble import RandomForestClassifier
import time
import spacy
from joblib import dump, load

import BankModel
from sklearn.pipeline import Pipeline

def import_data(spacy_nlp,reload):
    
    Y_train_filename = ".\data\Y_train_CVw08PX.csv"
    y_data = pd.read_csv(Y_train_filename,sep=',')
    y = y_data['prdtypecode']
    filename = "X_dt.mat"
    part =20
    
    
    if reload:
        Xtrain_filename = ".\data\X_train_update.csv"

        raw_data = pd.read_csv(Xtrain_filename,sep=',')
        raw_data = raw_data.drop(labels='imageid',axis=1)

        descrip = raw_data['description']
        design = raw_data['designation']
        
        X_data = []
        for k in range(len(design)//part):
            X_data.append(header.raw_to_tokens(design[k],spacy_nlp))
            header.progress_bar(k + 1,len(design)//part, prefix='Preprocessing train:', suffix='Complété', length=50)
        mdic = {"data": X_data}
        scipy.io.savemat('.\data\\' + filename,mdic)
    else:
        X_data = scipy.io.loadmat('.\data\\' + filename)['data']
    
    tfidf = TfidfVectorizer()
    X_tfidf_sample = tfidf.fit_transform(X_data)
    print(X_tfidf_sample.shape)
    return train_test_split(X_tfidf_sample[:len(X_data)//part],y[:len(X_data)//part],test_size=0.25,random_state=42,shuffle=True)
        

def RandomForest(X,Y,n_est):
    
    params = {
    'n_estimators': n_est#[10,50,100,200,300,400]
    }   
    n_folds = 5
    cv = KFold(n_splits=n_folds, shuffle=False)
    
    grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=params,
    return_train_score=True,
    cv=cv,
    ).fit(X, Y)
        
    return grid_search
    
    
    
def main():
    """Point d'entrée principal du programme."""
    
    
    Reload = input("Reload (True/False) : ")
    
    print("Import dictionnary")
    spacy_nlp = spacy.load("fr_core_news_sm")
    X_train,Y_train,X_test,Y_test = [],[],[],[]
    print("Import data")
    X_train,X_test,Y_train,Y_test = import_data(spacy_nlp,Reload)
    
       
    k=1
    n_ests = [200,300,600,900,1200,1500,1600,1800]
    print('setup model')
    
    t = time.time()
    model = RandomForest(X_train,Y_train,n_ests)
    best_params,best_est = model.best_params_,model.best_estimator_
    print(best_est,best_params)
    y_pred = model.predict(X_test)
# rf_filename = ".\models\\RandomForest.joblib"
# dump(model, rf_filename)
# BankModel.split_model(rf_filename,8)
# rfGB_filename = ".\models\\RandomForestGetBack.joblib"
# BankModel.GetBack_model(rfGB_filename,8)
# model = load(rfGB_filename)
    
    F1 = f1_score(Y_test,y_pred,average='weighted')
    
        
    
    
    
    return

if __name__ == '__main__':
    main()