from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, accuracy_score
from numpy import savetxt
import matplotlib.pyplot as plt
# import parfit.parfit as pf
# from sklearn.model_selection import RandomizedSearchCV,GridSearchCV, ParameterGrid, PredefinedSplit

def evaluate_model(predictions, prediction_probs,validLabel,modelName):

    results = {}
    results['recall'] = recall_score(validLabel, predictions)
    results['precision'] = precision_score(validLabel, predictions)
    results['roc auc score'] = roc_auc_score(validLabel, prediction_probs)
    results['f1'] = f1_score(validLabel,predictions)
    results['accuracy'] = accuracy_score(validLabel,predictions)

    print(f'\n {modelName}')
    for metric in ['recall', 'precision', 'roc auc score', 'f1', 'accuracy']:
        print(
            f' Validation Test - {metric.capitalize()} : {round(results[metric], 2)}')

    # Confusion matrix
    cm = confusion_matrix(validLabel, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels= ['Healthy','Cancer'])
    disp.plot()
    plt.title(modelName)
    plt.show(block=False)

def run_T_V_model(model, trainLabel, trainData, validLabel, validData, modelName):
    model.fit(trainData, trainLabel.ravel())
    modelPrediction = model.predict(validData)
    modelProba = model.predict_proba(validData)[:, 1]

    evaluate_model(modelPrediction, modelProba, validLabel, modelName)

def run_Test(model, testData, modelName):
    modelPrediction = model.predict(testData)
    savetxt(f'Output/{modelName}.csv', modelPrediction, delimiter=',')

def get_parameters(data,label,trainData, trainLabel, validData, validLabel):
    ''' '''
    '''
    Combining the train and valid arrays as they will be split by RandomizedSearchCV and GridSearchCV
    '''
    #test_fold = [-1] * 100
    #temp = [0] * 100
    #test_fold = test_fold + temp

    #FOR RANDOM SAMPLING

    # parameters = {
    #
    #     'n_estimators': np.linspace(10,200).astype(int),
    #     'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    #     'max_features': ['sqrt','log2', None],
    #     'max_leaf_nodes': [None] + list(np.linspace(10, 50, 40).astype(int)),
    #     'min_samples_split': [2, 5, 10],
    #     'bootstrap': [True, False]
    # }

    '''
    I initially ran RandomizedSearchCV for the Random Forest model on the above parameters and after homing in on the parameters improving the metrics, 
    I ran a GridSearch on the following parameters
     '''

    # parameters = {
    #     'n_estimators': [56, 60, 126],
    #     'max_depth': [12, 14, 17],
    #     'max_features': ['sqrt'],
    #     'max_leaf_nodes': [11, 18, 42],
    #     'min_samples_split': [2, 5],
    #     'bootstrap': [False],
    #     'random_state': [50]
    # }

    # estimator = RandomForestClassifier()

    '''
    In case of Gradient Boosting, a randomized search was run on the following parameters: and then a subsequent GridSearch.
    '''
    #FOR GRADIENT BOOSTING
    # parameters = {
    #     "loss": ["deviance",'exponential'],
    #     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    #     'min_samples_split': [2,3,4, 5,6],
    #     "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    #     "max_depth": [3, 5, 8],
    #     "max_features": ["log2", "sqrt"],
    #     "criterion": ["friedman_mse", "mse"],
    #     "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    #     'n_estimators': np.linspace(10, 200).astype(int),
    # }


    # estimator = GradientBoostingClassifier(random_state=50)

    # rs = GridSearchCV(estimator,parameters,n_jobs=-1,scoring='accuracy',verbose=1, cv = PredefinedSplit(test_fold))
    # # rs = RandomizedSearchCV(estimator,parameters,n_jobs=-1,scoring='accuracy',verbose=1, cv = PredefinedSplit(test_fold),n_iter=10)
    # rs.fit(data,label.ravel())
    # print(rs.best_params_)
    # print(rs.best_score_)


    '''
    For SGD the functionality of the python module parfit was used to tune the hyperparameters as it was faster and easier to use than the functions from scikit-learn
    '''
    # #FOR SGD
    # grid = {
    '''Initially I used the first list for alpha but with time and some tweaks I honed in on values in the 1400's which I then further reduced to just between 1413 and 1414'''
    #     # 'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3,1.5*1e3,2*1e3], # learning rate
    #     # 'alpha':np.linspace(1413,1414,30), # learning rate
    #     'n_iter_no_change': [1000],
    #     'max_iter': np.linspace(1000, 2000, 10), # number of epochs
    '''As modified huber and hinge were lazy it reduced training time, loss= log also produced similar results but I could never get it to cross accuracy threshold of 0.85 '''
    #     'loss': ['modified_huber'],
    #     'random_state': [103]
    # }
    # paramGrid = ParameterGrid(grid)
    #
    # bestModel, bestScore, allModels, allScores = pf.bestFit(SGDClassifier, paramGrid,
    #                                                         trainData, trainLabel, validData, validLabel,
    #                                                         metric=f1_score,
    #                                                         scoreLabel="f1")
    #
    # print(bestModel, bestScore)