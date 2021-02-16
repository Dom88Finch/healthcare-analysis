'''


Task Predict the Length of stay for each patient on case by case basis so that Hospitals can use this information
 for optimal resource allocation and better functioning.

'''

import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np 
import seaborn as sns 
import math 
import matplotlib.pyplot as plt
import sklearn as svm
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
#sns.set_theme(style="ticks", color_codes=True)

df = pd.read_csv('healthcare/train_data.csv')

#print(df.head())
#print(df.columns)


#rename columns names

df.columns = [x.lower().replace(' ','_') for x in df.columns]
print(df.columns)
print(df.head())
print(df.shape)
print(df.info())
print()


print(df.admission_deposit.value_counts())



print(round(((df.isnull().sum())/df.shape[0] *100),4))
# by checking our missing data, we can see that 1.2432%  of the city_code_patient data is missing from the column and 0.0355% of the 'bed_grade' is missing. 
#This numbers are not too high, so instead we will try and preserve this columns by using fillna
# We will decide to drop these two columns for now.

print()
#print(df.nunique().sort_values(ascending=False))



# The following columns are all categorical variables which need to be tranformed
obj_columns = ['hospital_type_code','age',
				'hospital_region_code','department', 'ward_type', 'ward_facility_code', 
				'type_of_admission', 'severity_of_illness', 'stay']


#print(df[obj_columns].info())



#print(df[obj_columns].nunique())
#print(df[obj_columns].head())


#print(df[obj_columns].isnull().sum())
encoder = LabelEncoder()

for i in range(len(obj_columns)):

	df1_encoded= encoder.fit_transform(df[obj_columns[i]])
	name = str(obj_columns[i] + '_mapped')
	df[name] = df1_encoded

#print(df.head())

#print(df.age.unique().sort_values())
#df['age'] = df.age.astype(int)



## DATA EXPLORATION WITH 5-10 DIFFERENT PLOTS ALONG WITH A BRIEF DESCRIPTION.


### Comparison of department and duration of stay 


# plt.scatter(df.department.head(20), df.stay_mapped.head(20))
# plt.show()





M = df[['department', 'stay_mapped']].head(200)

P = M.groupby('department')['stay_mapped'].sum()
P1 = M.groupby('department')['stay_mapped'].value_counts()





# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])

# ax.bar(M.department[M.department == 'radiotherapy'], M.stay_mapped[M.department == 'radiotherapy'])
# plt.show()

p=M[M.department == 'radiotherapy']
# plt.bar(p.stay_mapped.value_counts().index,p.stay_mapped.value_counts().values)
# plt.show()



# CountStatus = pd.value_counts(p.stay_mapped.values, sort=True)
# print(CountStatus)
# CountStatus.plot.barh()
# plt.show()



print(P1)
print(type(P1))









# sns.catplot(x="stay_mapped", y="department", data=M)
# plt.show()


# distribution of duration of stay vs the depertment type
# sns.catplot(x="stay_mapped", y="department",hue="stay_mapped", kind="bar", data=M,legend=True)
# plt.show()

d2  = df.head(200)

# sns.catplot(x="stay_mapped", y="department", kind="bar", data=M,legend=True)
# plt.show()


# sns.catplot(x="admission_deposit", y="age",kind="violin",data=d2)
# plt.title('distribution of admission_deposit vs age')
# plt.show()

# sns.catplot(x="age",kind="bar",hue="visitors_with_patient",data=d2)
# plt.title('distribution of admission_deposit vs age')
# plt.show()





# df = df.drop(columns = obj_columns,axis=1)
# print(df.info())

# df = df.drop(columns = ['bed_grade', 'city_code_patient','case_id'],axis=1)

features =['case_id', 'hospital_code', 'city_code_hospital',
       'available_extra_rooms_in_hospital', 'patientid',
       'visitors_with_patient', 'admission_deposit',
       'hospital_type_code_mapped', 'age_mapped',
       'hospital_region_code_mapped', 'department_mapped', 'ward_type_mapped',
       'ward_facility_code_mapped', 'type_of_admission_mapped',
       'severity_of_illness_mapped', 'stay_mapped']


from sklearn.preprocessing import StandardScaler
print(df.admission_deposit.head())
#df['admission_deposit'] = StandardScaler().fit_transform(df['admission_deposit'])
print(df.admission_deposit.head())
print(df.nunique())






#new_lg = LogisticRegression(clf__C=0.5, clf__penalty='l1', clf__solver= 'liblinear')






## As we can see, the 

# correlations = df.corr()


# print(correlations.stay_mapped.sort_values(ascending=False))
# print(correlations.stay_mapped[(correlations.stay_mapped > 0.05) ])


# print(correlations.stay_mapped[correlations.stay_mapped < -0.05])

## taking a quick glance,we can see a high correlation of 0.53 between visitors_of_patient and the length of stay  

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations,cmap='coolwarm', vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,len(df.columns),1)
# ax.set_xticks(ticks)
# plt.xticks(rotation=90)
# ax.set_yticks(ticks)
# ax.set_xticklabels(df.columns)
# ax.set_yticklabels(df.columns)
# plt.show()




















# model

features = ['visitors_with_patient', 'ward_type_mapped', 'age_mapped', 'hospital_type_code_mapped', 'available_extra_rooms_in_hospital','case_id', 'admission_deposit']

#features = df.drop(columns=['stay_mapped', 'patientid'], axis=1)



labels = ['stay_mapped']



from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import svm 
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier 
#from sklearn.ensemble import DecisionTree


from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics




X = np.array(df[features])
print(X)
y = np.array(df[labels])
print(y)


# traininig


lin_reg_model = LinearRegression()
log_reg_model = LogisticRegression()











rfg_model = RandomForestRegressor()
#suport_v_model = svm()


x_train, x_test, y_train,  y_test = train_test_split(X, y, test_size=0.2, random_state=0,shuffle=False)

lin_reg_model.fit(x_train,y_train)
prediction = np.round(lin_reg_model.predict(x_test),0)
score = metrics.accuracy_score(y_test, prediction)
print('Training Score:', score)
print(prediction[:10])
print(y_test[:10])



from sklearn.metrics import mean_squared_error
rf_mse = mean_squared_error(y_test,prediction)
rf_rmse = np.sqrt(rf_mse)
print(rf_rmse)




# Test to see which model performs best 


rvc_classifier = RandomForestClassifier(n_estimators = 50)







svm_classifier = svm.SVC(C= 1, gamma='auto', kernel= 'linear',probability=True)
gauNB_classifier = GaussianNB()




print()

models_t = [lin_reg_model, log_reg_model,gauNB_classifier,rvc_classifier]

n = ['LinearRegression_cf', 'LogisticRegression_cf', 'GaussianNB_cf', 'RandomForestClassifier']

# models_t = [svm_classifier]
# n = ['SVM_cf']




'''
m = 0
H = []

tr_results = []
tst_accuracy = []


from time import time
for model in models_t:
	
	print('------------',n[m],'---------------')
	m=m+1
	
	start = time()
	model.fit(x_train,y_train)
	end = time()
	
	print("Trained {} in {:.1f} seconds".format(model.__class__.__name__, (end - start)*10))
	predictions = np.round(model.predict(x_test),0)

	# print(predictions[0:20])
	# print(y_test[0:20])
	print('pred : actual')
	for i in range(20):
		print(predictions[i],'   ', y_test[i])

	train_accuracy_score= metrics.accuracy_score(y_test, predictions)
	print('Train Accuracy Score:',train_accuracy_score)



	# predictions_B = model.predict_proba(test_data_X)
	# prediction_class = model.predict(test_data_X)
	# test_accuracy_score= round(metrics.accuracy_score(test_data_y, prediction_class), 3)
	# print('Test Accuracy Score:',test_accuracy_score)

	# H.append(prediction_class.tolist())
	# tr_results.append(train_accuracy_score)
	# tst_accuracy.append(test_accuracy_score)
	# print(prediction_class[0:10])
	# print(test_data_y[0:10])


'''





## performing gridsearch on 



from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from time import time


X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=0,shuffle=False)

pipe_lr = Pipeline([('scl', StandardScaler()),
			('clf', LogisticRegression(random_state=0))])


pipe_rf = Pipeline([('scl', StandardScaler()),
			('clf', RandomForestClassifier(random_state=0))])




param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
param_range_fl = [1.0, 0.5, 0.1]



# grid search parameters
grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
		'clf__C': param_range_fl,
		'clf__solver': ['liblinear']}] 

grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],
		'clf__min_samples_leaf': param_range,
		'clf__max_depth': param_range,
		'clf__min_samples_split': param_range[1:]}]









# Construct grid searches
jobs = -1

gs_lr = GridSearchCV(estimator=pipe_lr,
			param_grid=grid_params_lr,
			scoring='accuracy',
			cv=2)


gs_rf = GridSearchCV(estimator=pipe_rf,
			param_grid=grid_params_rf,
			scoring='accuracy',
			cv=5,
			n_jobs=jobs)



grids = [gs_lr,gs_rf]

grid_dict = {0: 'Logistic Regression',1: 'RandomForestClassifier'}

start =time()
# Fit the grid search objects
print('Performing model optimizations...')
best_acc = 0.0
best_clf = 0
best_gs = ''
for idx, gs in enumerate(grids):
	print('\nEstimator: %s' % grid_dict[idx])	
	# Fit grid search	
	gs.fit(X_train, y_train)
	# Best params
	print('Best params: %s' % gs.best_params_)
	# Best training data accuracy
	print('Best training accuracy: %.3f' % gs.best_score_)
	# Predict on test data with best params
	y_pred = gs.predict(X_test)
	# Test data accuracy of model with best params
	print('Test set accuracy score for best params: %.3f ' % metrics.accuracy_score(y_test, y_pred))
	# Track best (highest test accuracy) model
	if metrics.accuracy_score(y_test, y_pred) > best_acc:
		best_acc = metrics.accuracy_score(y_test, y_pred)
		best_gs = gs
		best_clf = idx
	end = time()
	print("Trained {} in {:.1f} seconds".format(grid_dict[idx], (end - start)*10))



print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])

# Save best grid search pipeline to file
# dump_file = 'best_gs_pipeline.pkl'
# joblib.dump(best_gs, dump_file, compress=1)
# print('\nSaved %s grid search pipeline to file: %s' % (grid_dict[best_clf], dump_file))
























'''
after  our first initial training, we can see that the randomforest classifier performs best with a training score of 0.30, linear with score of 0.26, logistic with score of 0.25





'''


## our current score is 1.727



# Automate this pipeline to be able to test multiple models at once. 

# do cross validation,
# pick better hyperparameters using gridsearch
# pick better features
# create your own freatures.

# do writeup
# include at least 5-10 graphs showing data exploration
# show the avenues that were considered
# write a final conclusion of how you can improve model, write about limitations


