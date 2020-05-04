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
from sklearn.dummy import DummyClassifier

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



def dummy_plots(ab_data_obj,pk_data_obj,save_fig=True):
    ab_dummy_vals = {'uniform':0,'most_frequent':0,}
    pk_dummy_vals = {'uniform': 0, 'most_frequent': 0}
    for stratg in ab_dummy_vals.keys():
        dclf_a = DummyClassifier(strategy=stratg,random_state=ab_data_obj.rand_seed)
        dclf_a.fit(ab_data_obj.x_train,ab_data_obj.y_train)
        ab_dummy_vals[stratg] = dclf_a.score(ab_data_obj.x_test,ab_data_obj.y_test)

    for stratg in pk_dummy_vals.keys():
        dclf_p = DummyClassifier(strategy=stratg,random_state=pk_data_obj.rand_seed)
        dclf_p.fit(pk_data_obj.x_train,pk_data_obj.y_train)
        pk_dummy_vals[stratg] = dclf_p.score(pk_data_obj.x_test,pk_data_obj.y_test)
    #https://stackoverflow.com/questions/16010869/plot-a-bar-using-matplotlib-using-a-dictionary


    width = 0.35  # the width of the bars
    pk_dummy_vals.update({'DT_boost':0.9722,'DT':0.9722,'KNN':0.9718,'NN':0.9728,'SVM':0.9722})
    pk_dummy_vals['unif'] = pk_dummy_vals['uniform']
    del pk_dummy_vals['uniform']
    pk_dummy_vals['mst_frq'] = pk_dummy_vals['most_frequent']
    del pk_dummy_vals['most_frequent']

    ab_dummy_vals.update({'DT_boost':0.8894,'DT':0.8277,'KNN':0.8133,'NN':0.8273,'SVM':0})
    ab_dummy_vals['unif'] = ab_dummy_vals['uniform']
    del ab_dummy_vals['uniform']
    ab_dummy_vals['mst_frq'] = ab_dummy_vals['most_frequent']
    del ab_dummy_vals['most_frequent']

    x = np.arange(len(ab_dummy_vals))  # the label locations

    #https: // matplotlib.org / gallery / lines_bars_and_markers / barchart.html  # sphx-glr-gallery-lines-bars-and-markers-barchart-py
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, ab_dummy_vals.values(), width, label='ab_data')
    rects2 = ax.bar(x + width / 2, pk_dummy_vals.values(), width, label='pkr_data')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Test Accuracy')
    ax.set_xlabel('Prediction Method')
    ax.set_title('Model test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(ab_dummy_vals.keys())
    ax.set_ylim(0,1.2)
    ax.set_yticks(np.arange(0, 1.01, step=0.2))
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height,3)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    if save_fig:
        plt.savefig("plt_dummy_vals_accuracy")
    else:
        plt.show()
    return  True


def dummy_plots_auc(ab_data_obj,pk_data_obj,save_fig=True):
    ab_dummy_vals = {'uniform':0,'most_frequent':0,}
    pk_dummy_vals = {'uniform': 0, 'most_frequent': 0}
    for stratg in ab_dummy_vals.keys():
        dclf_a = DummyClassifier(strategy=stratg,random_state=ab_data_obj.rand_seed)
        dclf_a.fit(ab_data_obj.x_train,ab_data_obj.y_train)
        
        y_prob_ab = dclf_a.predict_proba(ab_data_obj.x_test)
        y_predict_ab = dclf_a.predict(ab_data_obj.x_test)
        score_ab = metrics.roc_auc_score(ab_data_obj.y_test,y_prob_ab,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data
        ab_dummy_vals[stratg] = score_ab

    for stratg in pk_dummy_vals.keys():
        dclf_p = DummyClassifier(strategy=stratg,random_state=pk_data_obj.rand_seed)
        dclf_p.fit(pk_data_obj.x_train,pk_data_obj.y_train)
        y_prob_pk = dclf_a.predict_proba(pk_data_obj.x_test)
        y_predict_pk = dclf_a.predict(pk_data_obj.x_test)
        score_pk = metrics.roc_auc_score(pk_data_obj.y_test,y_predict_pk,multi_class='ovr',average='macro',max_fpr=1.0) #for pk data
        pk_dummy_vals[stratg] = score_pk

    #https://stackoverflow.com/questions/16010869/plot-a-bar-using-matplotlib-using-a-dictionary


    width = 0.35  # the width of the bars
    pk_dummy_vals.update({'DT_boost':0.5400,'DT':0.5755,'KNN':0.5282,'NN':0.4855,'SVM':0.5}) #TODO update this
    pk_dummy_vals['unif'] = pk_dummy_vals['uniform']
    del pk_dummy_vals['uniform']
    pk_dummy_vals['mst_frq'] = pk_dummy_vals['most_frequent']
    del pk_dummy_vals['most_frequent']

    ab_dummy_vals.update({'DT_boost':0.8874,'DT':0.884,'KNN':0.8697,'NN':0.6359,'SVM':0})
    ab_dummy_vals['unif'] = ab_dummy_vals['uniform']
    del ab_dummy_vals['uniform']
    ab_dummy_vals['mst_frq'] = ab_dummy_vals['most_frequent']
    del ab_dummy_vals['most_frequent']

    x = np.arange(len(ab_dummy_vals))  # the label locations

    #https: // matplotlib.org / gallery / lines_bars_and_markers / barchart.html  # sphx-glr-gallery-lines-bars-and-markers-barchart-py
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, ab_dummy_vals.values(), width, label='ab_data')
    rects2 = ax.bar(x + width / 2, pk_dummy_vals.values(), width, label='pkr_data')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Test Average ROC_AUC')
    ax.set_xlabel('Prediction Method')
    ax.set_title('Model Average Test ROC_AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(ab_dummy_vals.keys())
    ax.set_ylim(0,1.2)
    ax.set_yticks(np.arange(0, 1.01, step=0.2))
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height,3)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    if save_fig:
        plt.savefig("plt_dummy_vals_roc")
    else:
        plt.show()
    return  True


dummy_plots(ab_data,pkr_data,save_fig=True)
dummy_plots_auc(ab_data,pkr_data,save_fig=True)