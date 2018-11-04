#z!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pprint
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# %matplotlib inline # uncomment in jupyter to plot inline

import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### here we are also including 'poi' inside the financial_features only since its an imp aspect of our data
financial_features = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', \
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', \
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', \
                      'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',\
                  'shared_receipt_with_poi']
                  
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file) # this would the original copy of the data we are extracting.
df = pd.DataFrame.from_dict(data_dict, orient='index')
df.head()
    
# We're having a look at first few entries in the data dictionary.
count = 1
for i in data_dict.keys()[:15]:
    print count," ", i
    count+=1

pprint.pprint (data_dict['BAXTER JOHN C'])

print 'Number of people:', df['poi'].count()
print 'Number of POIs:', df.loc[df.poi == True, 'poi'].count()
print 'Fraction of examples that are POIs:', \
    float(df.loc[df.poi == True, 'poi'].count()) / df['poi'].count()
print 'Number of features:', df.shape[1]

fpoi = open("poi_names.txt", "r")
rfile = fpoi.readlines()
poi = len(rfile[2:])
print "There were " + str(poi) + " poi's total."

# df.isnull().sum()
df = df.replace('NaN', np.nan)
# df.isnull().sum()

df.info()
df[financial_features] = df[financial_features].fillna(0)
# df.isnull().sum()

df[email_features] = df[email_features].fillna(df[email_features].median())
df.isnull().sum()

features = ["salary", "bonus"]
#data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)
### plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.title("Salary vs Bonus")


### Task 2: Remove outliers
outlier = max(data,key=lambda item:item[1])
print (outlier)

my_dataset = data_dict # we keep our original data as it is for our reference and use a copy for modifications

for person in my_dataset:
    if my_dataset[person]['salary'] == outlier[0] and my_dataset[person]['bonus'] == outlier[1]:
        print "The outlier is : ",person
print "Before removing TOTAL length of our dataset is ",len(my_dataset)
my_dataset.pop('TOTAL',0)
my_dataset.pop('LOCKHART EUGENE E',0)
print "After removing TOTAL length of our dataset is ",len(my_dataset)
data = featureFormat(my_dataset,features) 

for person in data:
    salary = person[0]
    bonus = person[1]
    plt.scatter(salary, bonus)
    
    plt.xlabel("salary")
    plt.ylabel("bonus")
    plt.title('Salary vs Bonus')

(df["salary"]).describe().apply(lambda x: format(x, '.2f'))

for person in my_dataset:
    if my_dataset[person]['salary'] != 'NaN' and my_dataset[person]['bonus'] != 'NaN' \
    and my_dataset[person]['salary'] >= 1000000 and my_dataset[person]['bonus'] >= 5000000:
        print person, 'Salary:', my_dataset[person]['salary'], 'Bonus:', my_dataset[person]['bonus']

pprint.pprint (my_dataset['LAY KENNETH L'])
pprint.pprint (my_dataset['SKILLING JEFFREY K'])
# removing outliers from dataframe object
df = df.drop('TOTAL')               # this is getting removed from the dataframe, earlier we removed from the dict
df = df.drop('LOCKHART EUGENE E')   # this is getting removed from the dataframe, earlier we removed from the dict

# exploring financial data
g = sns.PairGrid(df, vars=['salary','bonus','total_stock_value','total_payments'],
                hue='poi')
g.map_lower(plt.scatter)
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=1)

len(df)

# exloring email data
pois = df[df.poi]
plt.scatter(pois.from_messages, pois.to_messages, c='red');
non_pois = df[~df.poi]
plt.scatter(non_pois.from_messages, non_pois.to_messages, c='blue');

plt.xlabel('From messages')
plt.ylabel('To messages')
plt.legend(['POIs', 'non-POIs'])

# extracting the outlier points
outliers = df[np.logical_or(df.from_messages > 6000, df.to_messages > 10000)]

#plot them in red with the originals
plt.scatter(df.from_messages, df.to_messages, c='blue');
plt.scatter(outliers.from_messages, outliers.to_messages, c='red')
plt.xlabel('From messages')
plt.ylabel('To messages')
plt.legend(['Inliers', 'Potential Outliers'])

# Let's look at our outliers!
outliers

# removing the outlier candidates from the data set
df = df[df.from_messages < 6000] # here the value 6000 is for the value of emailes being sent from
df = df[df.to_messages < 10000]  # here the value 10000 is for the value of emailes being sent to
len(df)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, financial_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Helper Functions
from sklearn.feature_selection import SelectKBest,f_classif
f_scores =[]

def proportion_from_poi(data_dict):
    for k, v in data_dict.iteritems():
    #Assigning value to the feature 'proportion_from_poi'
        if v['from_poi_to_this_person'] != 'NaN' and  v['from_messages'] != 'NaN':
            v['proportion_from_poi'] = float(v['from_poi_to_this_person']) / v['from_messages'] 
        else:    
            v['proportion_from_poi'] = 0.0
    return (data_dict)       
            
def proportion_to_poi(data_dict):
    for k, v in data_dict.iteritems():
        #Assigning value to the feature 'proportion_to_poi'        
        if v['from_this_person_to_poi'] != 'NaN' and  v['to_messages'] != 'NaN':
            v['proportion_to_poi'] = float(v['from_this_person_to_poi'] )/ v['to_messages']   
        else:
            v['proportion_to_poi'] = 0.0
    return (data_dict)

def net_worth (data_dict) :
    features = ['total_payments','total_stock_value']
    for key in data_dict :
        name = data_dict[key]
        is_null = False 
        
        for feature in features:
            if name[feature] == 'NaN':
                is_null = True
        if not is_null:
            name['net_worth'] = name[features[0]] + name[features[1]]
        else:
            name['net_worth'] = 'NaN'
    return data_dict                
            
def select_features(features,labels,features_list,k=10) :
    clf = SelectKBest(f_classif,k)
    new_features = clf.fit_transform(features,labels)
    features_l=[features_list[i+1] for i in clf.get_support(indices=True)]
    f_scores = zip(features_list[1:],clf.scores_[:])
    f_scores = sorted(f_scores,key=lambda x: x[1],reverse=True)
    return new_features, ['poi'] + features_l, f_scores

data_dict = net_worth(data_dict)
data_dict = proportion_from_poi(data_dict)
data_dict = proportion_to_poi(data_dict)
pprint.pprint(data_dict['ALLEN PHILLIP K'])

# we will add these features to our financial_features 
financial_features+=['net_worth','proportion_from_poi','proportion_to_poi']

my_dataset = data_dict

data = featureFormat(my_dataset, financial_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

for point in data_dict:
    salary = data_dict[point]["proportion_from_poi"]
    bonus = data_dict[point]["proportion_to_poi"]
    plt.scatter( salary, bonus)
plt.xlabel("proportion_from_poi")
plt.ylabel("proportion_to_poi")
plt.show()




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

features,financial_features,f_scores=select_features(features,labels,financial_features,k=6)
# call the function with uses selectkbest
print ("features_list---" ,financial_features)
print("feature scores")
for i in f_scores:
    print (i)
data = featureFormat(my_dataset, financial_features, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf,my_dataset,financial_features)


from sklearn import tree
clf1=tree.DecisionTreeClassifier()
test_classifier(clf1,my_dataset,financial_features)


from sklearn.ensemble import AdaBoostClassifier
# clf2 = AdaBoostClassifier()
# test_classifier(clf2,my_dataset,financial_features)


# from sklearn.neighbors import KNeighborsClassifier 
# clf3=KNeighborsClassifier(n_neighbors = 4)
# test_classifier(clf3,my_dataset,financial_features)

from sklearn.neighbors.nearest_centroid import NearestCentroid
clf4 = NearestCentroid()
test_classifier(clf4,my_dataset,financial_features)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

'''
OUR FINAL ALGORITHM
'''
from time import time
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

t0 = time()

pipe1 = Pipeline([('pca',PCA()),('classifier',GaussianNB())])
param = {'pca__n_components':[4,5,6]}
gsv = GridSearchCV(pipe1, param_grid=param,n_jobs=2,scoring = 'f1',cv=2)
gsv.fit(features_train,labels_train)
clf = gsv.best_estimator_
print("GausianNB with PCA fitting time: %rs" % round(time()-t0, 3))
pred = clf.predict(features_test)

t0 = time()
test_classifier(clf,my_dataset,financial_features,folds = 1000)
print("GausianNB  evaluation time: %rs" % round(time()-t0, 3))

'''
Adaboost tuned for comparision with final algorithm
'''
from sklearn.tree import DecisionTreeClassifier
abc = AdaBoostClassifier(random_state=40)
data = featureFormat(my_dataset, financial_features, sort_keys = True)
labels, features = targetFeatureSplit(data)
dt = []
for i in range(6):
    dt.append(DecisionTreeClassifier(max_depth=(i+1)))
ab_params = {'base_estimator': dt,'n_estimators': [60,45, 101,10]}
t0 = time()
abt = GridSearchCV(abc, ab_params, scoring='f1',)
abt = abt.fit(features_train,labels_train)
print("AdaBoost fitting time: %rs" % round(time()-t0, 3))
abc = abt.best_estimator_
t0 = time()
test_classifier(abc, data_dict, financial_features, folds = 100)
print("AdaBoost evaluation time: %rs" % round(time()-t0, 3))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, financial_features)