import numpy as np
import pandas as pd
from sklearn.ensemble   import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble   import GradientBoostingClassifier
import time
# import itertools




def evaluate_model(predictions, prediction_probs,validLabel):

    # baseline = {}
    # baseline['recall'] = recall_score(validLabel,[1 for _ in range(len(validLabel))])
    # baseline['precision'] = precision_score(validLabel,[1 for _ in range(len(validLabel))])
    # baseline['roc'] = 0.5

    results = {}
    results['recall'] = recall_score(validLabel, predictions)
    results['precision'] = precision_score(validLabel, predictions)
    results['roc'] = roc_auc_score(validLabel, prediction_probs)
    results['f1'] = f1_score(validLabel,predictions)
    results['accuracy'] = accuracy_score(validLabel,predictions)
    # train_results = {}
    # train_results['recall'] = recall_score(trainLabel, train_predictions)
    # train_results['roc'] = roc_auc_score(trainLabel, train_probs)

    for metric in ['recall', 'precision', 'roc', 'f1', 'accuracy']:
        print(
            f' Validation Test - {metric.capitalize()} : {round(results[metric], 2)}')

    # Confusion matrix
    cm = confusion_matrix(validLabel, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels= ['Healthy','Cancer'])
    disp.plot()
    plt.show()

def run_T_V_model(model,trainLabel,trainData,validLabel,validData):
    model.fit(trainData, trainLabel.ravel())
    modelPrediction = model.predict(validData)
    modelProba = model.predict_proba(validData)[:, 1]

    evaluate_model(modelPrediction, modelProba,validLabel)

start = time.time()
rawTrainData    =   pd.read_csv('Data/arcene_train.data', delimiter='\s+',header=None)
rawTrainLabel   =   pd.read_csv('Data/arcene_train.labels', delimiter='\s+',header=None)
rawValidData   =   pd.read_csv('Data/arcene_valid.data', delimiter='\s+',header=None)
rawValidLabel   =   pd.read_csv('Data/arcene_valid.labels', delimiter='\s+',header=None)

trainLabel   =  np.array(rawTrainLabel)
trainData    =  np.array(rawTrainData)
validLabel   =  np.array(rawValidLabel)
validData    =  np.array(rawValidData)


randomForest    =   RandomForestClassifier(n_estimators=60,min_samples_split=5,max_leaf_nodes=11,max_features='sqrt',max_depth=12,bootstrap=False,random_state=50)
run_T_V_model(randomForest,trainLabel,trainData,validLabel,validData)

gradientBoost    =   GradientBoostingClassifier(subsample= 0.94, n_estimators= 192, min_samples_split= 2, min_samples_leaf= 0.24545454545454548, max_features= 'log2', max_depth= 8, loss='deviance', learning_rate= 0.1, criterion= 'friedman_mse',random_state=50)
run_T_V_model(gradientBoost,trainLabel,trainData,validLabel,validData)


print(f'Time:{time.time() - start}')
def lookingForParameters():
    # parameters = {
    #
    #     'n_estimators': np.linspace(10,200).astype(int),
    #     'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    #     'max_features': ['sqrt','log2', None] + list(np.arange(0.5, 1, 0.1)),
    #     'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
    #     'min_samples_split': [2, 5, 10],
    #     'bootstrap': [True, False]
    # }

    # parameters = {
    #
    #     'n_estimators': [56, 60, 126],
    #     'max_depth': [12, 14, 17],
    #     # 'max_features': ['sqrt','log2', None] + list(np.arange(0.5, 1, 0.1)),
    #     'max_leaf_nodes': [11, 18, 42],
    #     'min_samples_split': [2, 5]
    #     # 'bootstrap': [True, False]
    # }
    # estimator = RandomForestClassifier(random_state=50, max_features='sqrt')

    parameters = {
        "loss": ["deviance",'exponential'],
        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        # "min_samples_split": np.linspace(0.1, 0.5, 12),
        # 'min_samples_split': [2, 5],
        'min_samples_split': [2,3,4, 5,6],
        "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        # "min_samples_leaf": np.linspace(0.18, 0.35, 12),
        "max_depth": [3, 5, 8],
        "max_features": ["log2", "sqrt"],
        "criterion": ["friedman_mse", "mse"],
        "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
        # "subsample": [0.85, 0.9, 0.95, 1.0],
        'n_estimators': np.linspace(10, 200).astype(int),
        # 'n_estimators': np.linspace(150, 250).astype(int)
    }

    # parameters = [
    #     {'C': np.linspace(0.1,5,10), 'kernel': ['linear']},
    #     {'C': np.linspace(0.1,5,10), 'gamma': [0.001, 0.0001,0.01], 'kernel': ['rbf','poly','sigmoid'], 'degree': [3,4,5]},
    # ]
    #
    # estimator = SVC(random_state= 50)
    # rs = RandomizedSearchCV(estimator,parameters, n_jobs=-1,scoring='f1',verbose=1,n_iter=50)


    estimator   =   GradientBoostingClassifier(random_state=50)
    rs = RandomizedSearchCV(estimator, parameters, n_jobs = -1,
                           scoring = 'f1',
                            n_iter = 200, verbose = 1)
    # rs = RandomizedSearchCV(estimator, parameters, n_jobs=-1,
    #                   scoring='f1', cv=3,
    #                   verbose=1)




    rs.fit(trainData, trainLabel.ravel())
    print(rs.best_params_)

    bestModel = rs.best_estimator_

    # train_rfProbs = bestModel.predict_proba(trainData)[:, 1]

    rFPrediction = bestModel.predict(validData)
    rfProbs = bestModel.predict_proba(validData)[:, 1]

    # evaluate_model(rFPrediction, rfProbs, train_rFPrediction, train_rfProbs)
    print(f'F1 SCORE: {f1_score(validLabel, rFPrediction)}')
    print(f'ACCURACY : {accuracy_score(validLabel, rFPrediction)}')
    print(f'RECALL: {recall_score(validLabel, rFPrediction)}')
    print(f'Precision: {precision_score(validLabel, rFPrediction)}')

    print(f'Time:{time.time() - start}')

# lookingForParameters()
