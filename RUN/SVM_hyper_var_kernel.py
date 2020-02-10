import preprocess as prp
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

import preprocess as prp
from sklearn import metrics
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from joblib import dump, load


pkr_data = prp.pkr_data()
pkr_data.clean()
pkr_data.target = 'hand'
pkr_data.features = pkr_data.all.columns[pkr_data.all.columns != pkr_data.target]

pkr_data.init_model_data(target =['hand'],features = ['suit1','card1','suit2','card2','suit3','card3','suit4','card4'])

#ab data
ab_data = prp.ab_data(n=100)
# ab_data.encode = ['room_type']
ab_data.clean()
ab_data.target = 'room_type'
ab_data.features = ab_data.all.columns[ab_data.all.columns != ab_data.target]

ab_data.init_model_data(target=ab_data.target,features=ab_data.features)
print("Models Initiated")


ab_train_scores = {}
ab_test_scores = {}
ab_test_auc = {}
ab_train_auc = {}

pkr_train_scores = {}
pkr_test_scores = {}
pkr_test_auc = {}
pkr_train_auc = {}

for krn in ['linear', 'poly', 'rbf', 'sigmoid']:

    SVM_ab = svm.SVC(kernel=krn,verbose=False,probability=True)
    SVM_ab.fit(ab_data.x_train,ab_data.y_train)
    ab_data.y_predict=SVM_ab.predict(ab_data.x_test)
    y_prob = SVM_ab.predict_proba(ab_data.x_test)
    ab_acc_score = metrics.accuracy_score(ab_data.y_test,ab_data.y_predict)
    ab_roc_score = metrics.roc_auc_score(ab_data.y_test,ab_data.y_predict,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data



    # SVM_ab.fit(ab_data.x_train, ab_data.y_train)
    ab_train_score = SVM_ab.score(ab_data.x_train, ab_data.y_train)
    ab_test_score = SVM_ab.score(ab_data.x_test, ab_data.y_test)
    ab_train_scores[krn] = round(ab_train_score,4)
    ab_test_scores[krn] = round(ab_test_score,4)
    ab_test_auc[krn] = metrics.roc_auc_score(ab_data.y_test,SVM_ab.predict(ab_data.x_test),average='macro',multi_class='ovr')
    ab_train_auc[krn] = metrics.roc_auc_score(ab_data.y_train, SVM_ab.predict_proba(ab_data.x_train), average='macro',multi_class='ovr')

    SVM_pkr = svm.SVC(kernel=krn,verbose=False)

    SVM_pkr.fit(pkr_data.x_train, pkr_data.y_train)
    pkr_train_score = round(SVM_pkr.score(pkr_data.x_train, pkr_data.y_train),4)
    pkr_test_score = round(SVM_pkr.score(pkr_data.x_test, pkr_data.y_test),4)
    pkr_train_scores[krn] = pkr_train_score
    pkr_test_scores[krn] = pkr_test_score
    pkr_test_auc[krn] = round(metrics.roc_auc_score(pkr_data.y_test, SVM_pkr.predict(pkr_data.x_test), average='macro'),4)
    pkr_train_auc[krn] = round(metrics.roc_auc_score(pkr_data.y_train, SVM_pkr.predict(pkr_data.x_train), average='macro'),4)

    dump(SVM_ab, 'SVM_ab_'+krn+'.joblib')
    dump(SVM_pkr, 'SVM_pkr_' + krn + '.joblib')

marker = 1


# https://python-graph-gallery.com/11-grouped-barplot/

barWidth = 0.1

# set height of bar
ab_train = list(ab_train_scores.values())
ab_test = list(ab_test_scores.values())
ab_auc_train = list(ab_train_auc.values())
ab_auc_test = list(ab_test_auc.values())

pkr_train = list(pkr_train_scores.values())
pkr_test =  list(pkr_test_scores.values())
pkr_auc_train = list(pkr_train_auc.values())
pkr_auc_test = list(pkr_test_auc.values())

width = 0.1  # the width of the bars
x = np.arange(len(ab_train))
fig, ax = plt.subplots()
rects1 = ax.bar(x - 4*width, ab_train, width, label='ab_train',color='blue')
rects2 = ax.bar(x - 2.5*width, ab_test, width, label='ab_test',color='cyan')
rects3 = ax.bar(x, pkr_train, width, label='pkr_train',color='red')
rects4 = ax.bar(x + width, pkr_test, width, label='pkr_test',color='orange')
rects5 = ax.bar(x + 2.5*width, pkr_auc_train, width, label='pkr_auc_train',color='magenta')
rects6 = ax.bar(x + 4*width, pkr_auc_test, width, label='pkr_auc_test',color='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Score (%)')
ax.set_title('SVM Kernel Performance')
ax.set_xticks(x)
ax.set_xticklabels(list(ab_train_scores.keys()))
ax.legend(loc='lower left')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)


fig.tight_layout()

plt.close()
