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
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_auc_ovr
from sklearn.model_selection import GridSearchCV
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
#validate classifier ab_data

ab_train_scores = {}
ab_test_scores = {}
pkr_train_scores = {}
pkr_test_scores = {}
pkr_test_auc = {}
pkr_train_auc = {}

for k in range(1,21):
    KNN_ab = KNeighborsClassifier(n_neighbors=k)
    KNN_ab.fit(ab_data.x_train,ab_data.y_train)
    ab_train_score = KNN_ab.score(ab_data.x_train,ab_data.y_train)
    ab_test_score = KNN_ab.score(ab_data.x_test,ab_data.y_test)
    ab_train_scores[k] = ab_train_score
    ab_test_scores[k] = ab_test_score

    KNN_pkr = KNeighborsClassifier(n_neighbors=k)
    KNN_pkr.fit(pkr_data.x_train,pkr_data.y_train)
    pkr_train_score = KNN_pkr.score(pkr_data.x_train,pkr_data.y_train)
    pkr_test_score = KNN_pkr.score(pkr_data.x_test,pkr_data.y_test)
    pkr_train_scores[k] = pkr_train_score
    pkr_test_scores[k] = pkr_test_score
    pkr_test_auc[k] = metrics.roc_auc_score(pkr_data.y_test,KNN_pkr.predict(pkr_data.x_test),average='macro')
    pkr_train_auc[k] = metrics.roc_auc_score(pkr_data.y_train, KNN_pkr.predict(pkr_data.x_train), average='macro')

marker = 1

plt.plot(list(ab_train_scores.keys()),list(ab_train_scores.values()),'o-',label='ab_training_accuracy', color="b")
plt.plot(list(ab_test_scores.keys()),list(ab_test_scores.values()),'x-',label='ab_test_accuracy', color="b")
plt.plot(list(pkr_train_scores.keys()),list(pkr_train_scores.values()),'o-',label='pkr_training_accuracy', color="r")
plt.plot(list(pkr_test_scores.keys()),list(pkr_test_scores.values()),'x-',label='pkr_test_accuracy', color="r")
plt.plot(list(pkr_train_auc.keys()),list(pkr_train_auc.values()),'o--',label='pkr_training_roc', color="r")
plt.plot(list(pkr_test_auc.keys()),list(pkr_test_auc.values()),'x--',label='pkr_test_roc', color="r")


plt.xlabel('# of Neighbors K')
plt.ylabel('Accuracy (%)')
plt.ylim(0,1)
plt.title('KNN Train/Test accuracy vs K')
plt.legend(loc='best')
plt.xticks(list(ab_train_scores.keys()))
# plt.show()
plt.savefig('KNN_vary_K')
plt.close()