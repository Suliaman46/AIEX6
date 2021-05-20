import numpy as np
import pandas as pd
from sklearn.ensemble   import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
import time
import matplotlib.pyplot as plt
from helpers import run_T_V_model, run_Test

start = time.time()
rawTrainData    =   pd.read_csv('Data/arcene_train.data', delimiter='\s+',header=None)
rawTrainLabel   =   pd.read_csv('Data/arcene_train.labels', delimiter='\s+',header=None)
rawValidData   =   pd.read_csv('Data/arcene_valid.data', delimiter='\s+',header=None)
rawValidLabel   =   pd.read_csv('Data/arcene_valid.labels', delimiter='\s+',header=None)

trainLabel   =  np.array(rawTrainLabel)
trainData    =  np.array(rawTrainData)
validLabel   =  np.array(rawValidLabel)
validData    =  np.array(rawValidData)


data = np.concatenate([trainData,validData])
label = np.concatenate([trainLabel,validLabel])

# get_parameters(data,label,trainData, trainLabel, validData, validLabel)

randomForest    =   RandomForestClassifier(n_estimators=126,min_samples_split=2,max_leaf_nodes=18,max_features='sqrt',max_depth=12,bootstrap=False,random_state=50)
run_T_V_model(randomForest,trainLabel,trainData,validLabel,validData,'RandomForest')

gradientBoost    =   GradientBoostingClassifier(subsample= 0.94, n_estimators= 192, min_samples_split= 2, min_samples_leaf= 0.24545454545454548, max_features= 'log2', max_depth= 8, loss='deviance', learning_rate= 0.1, criterion= 'friedman_mse',random_state=50)
run_T_V_model(gradientBoost,trainLabel,trainData,validLabel,validData,'GradientBoost')

sgdModel = SGDClassifier(alpha=1413.7931034482758, loss='modified_huber', max_iter=2000,n_iter_no_change=1000, n_jobs=-1, random_state=103)
run_T_V_model(sgdModel,trainLabel,trainData,validLabel,validData,'SGD')

rawTestData = pd.read_csv('Data/arcene_test.data', delimiter='\s+', header=None)
testData = np.array(rawTestData)

run_Test(randomForest,testData,'RandomForest')
run_Test(gradientBoost,testData,'GradientBoost')
run_Test(sgdModel,testData,'SGD')


print(f'Time:{time.time() - start}')
plt.show()


