import preprocess as prp
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import preprocess as prp
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import dump, load

from sklearn.model_selection import GridSearchCV
from pactools.grid_search import GridSearchCVProgressBar


#pkr_data
#TODO move into preprocess
pkr_data = prp.pkr_data()
pkr_data.clean()
pkr_data.target = 'hand'
pkr_data.features = pkr_data.all.columns[pkr_data.all.columns != pkr_data.target]

pkr_data.init_model_data(target =['hand'],features = ['suit1','card1','suit2','card2','suit3','card3','suit4','card4'])

#ab data
ab_data = prp.ab_data()
# ab_data.encode = ['room_type']
ab_data.clean()
ab_data.target = 'room_type'
ab_data.features = ab_data.all.columns[ab_data.all.columns != ab_data.target]

ab_data.init_model_data(target=ab_data.target,features=ab_data.features)
print("Models Initiated")

#Validate
# DTree=DecisionTreeClassifier()
# DTree.fit(ab_data.x_train,ab_data.y_train)
# ab_data.y_predict=DTree.predict(ab_data.x_test)
#
# score = metrics.accuracy_score(ab_data.y_test,ab_data.y_predict)
# print("default score: ",score)

params = {'criterion':['gini', 'entropy'],'splitter':['best', 'random'],'min_samples_leaf':[1,2,3,4,5,10,50,100,500,1000,5000,10000]}
ada_params = {'algorithm':['SAMME', 'SAMME.R'], 'learning_rate':np.linspace(0.1,1.0,10),'n_estimators':[1,2,3,4,5,10,50,100]}


def DT_gridSearch(data_obj, params):
    # TODO add progress bar


    clf_grid = make_pipeline(preprocessing.StandardScaler(),
                             GridSearchCV(DecisionTreeClassifier(),
                                          param_grid=params,
                                          cv=5,
                                          refit=True, n_jobs=3, verbose=2,scoring='roc_auc_ovr'))

    # clf_grid = make_pipeline(preprocessing.StandardScaler(),
    #                          GridSearchCV(DecisionTreeClassifier(),
    #                                       param_grid=params,
    #                                       cv=5,
    #                                       refit=True, n_jobs=3, verbose=2))
    #

    clf_grid.fit(data_obj.x_train, data_obj.y_train)
    # score = clf_grid.score(data_obj.x_test, data_obj.y_test)

    y_prob = clf_grid.predict_proba(data_obj.x_test)
    y_predict = clf_grid.predict(data_obj.x_test)

    score = metrics.roc_auc_score(data_obj.y_test,y_prob,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data
    # score = metrics.roc_auc_score(data_obj.y_test, y_predict, multi_class='ovr', average='macro',max_fpr=1.0)  # for pkr data

    print("DT grid search:", score)
    # dump(clf_grid, 'clf_DT_grid' + str(data_obj.title) + '_final_' + str(round(score, 4)) + '.joblib')
    dump(clf_grid, 'clf_DT_grid' + str(data_obj.title)+'_final_roc_'+ str(round(score, 4)) + '.joblib')

def DT_ada_gridSearch(data_obj, params):
    # TODO add progress bar

    clf_grid = make_pipeline(preprocessing.StandardScaler(),
                             GridSearchCV(AdaBoostClassifier(),
                                          param_grid=params,
                                          cv=5,
                                          refit=True, n_jobs=3, verbose=2))

    # clf_grid = make_pipeline(preprocessing.StandardScaler(),
    #                          GridSearchCV(AdaBoostClassifier(),
    #                                       param_grid=params,
    #                                       cv=5,
    #                                       refit=True, n_jobs=3, verbose=2,scoring='roc_auc_ovr'))


    clf_grid.fit(data_obj.x_train, data_obj.y_train)

    y_prob = clf_grid.predict_proba(data_obj.x_test)
    y_predict = clf_grid.predict(data_obj.x_test)

    score = clf_grid.score(data_obj.x_test, data_obj.y_test)
    # score = metrics.roc_auc_score(data_obj.y_test,y_prob,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data
    # score = metrics.roc_auc_score(data_obj.y_test, y_predict, multi_class='ovr', average='macro',max_fpr=1.0)  # for pkr data


    print("DT adaboost grid search:", score)
    dump(clf_grid, 'clf_DT_ada_grid' + str(data_obj.title) +'_final_'+ str(round(score, 4)) + '.joblib')
    # dump(clf_grid, 'clf_DT_ada_grid' + str(data_obj.title) +'_final_roc'+ str(round(score, 4)) + '.joblib')


# DT_gridSearch(ab_data,params)
# DT_gridSearch(pkr_data, params)

# DT_ada_gridSearch(ab_data,ada_params)
DT_ada_gridSearch(pkr_data,ada_params)
