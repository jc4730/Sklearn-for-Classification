# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:57:33 2017

@author: jc9730; Jiada Chen; HW3_q3
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, linear_model, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

def main():
    inp_path = sys.argv[1]
    out_path = sys.argv[2]
    data = []
    labels= []
    output = []
    
    with open(inp_path,"rU") as f:
        reader = csv.reader(f, delimiter=',')
        
        reader.next()
        
        for row in reader:
            data.append(row[0:2])
            labels.append(row[2])
    
    
    data = np.array(data, dtype = float)
    labels = np.array(labels, dtype = float)
    n = len(data)
    
    ### Scaling Features
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0,ddof=1)
    data = (data - mu)/std
    
    
    ### Splitting Data
    x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=0.4, random_state=1,stratify=labels)
    
    
    # SVM with Linear Kernel
    print '### SVM with Linear Kernel ###'
    parameters = {'C':[0.1,0.5,1.,5.,10.,50.,100.]}
    svr = svm.SVC(kernel='linear')
    clf = GridSearchCV(svr,parameters,cv=5,scoring='accuracy')
    clf.fit(x_train,y_train)
    scores = clf.cv_results_['mean_test_score']
    #print max(scores)
    print clf.best_params_
    #print clf.score(x_test,y_test)
    output_row = ['svm_linear',max(scores),clf.score(x_test,y_test)]
    output.append(output_row)
    
    
    # SVM with Polynomial Kernel
    print '### SVM with Polynomial Kernel ###'
    parameters = {'C':[0.1,1.,3.],'degree':[4,5,6],'gamma':[.1,1.]}
    svr = svm.SVC(kernel='poly')
    clf = GridSearchCV(svr,parameters,cv=5,scoring='accuracy')
    clf.fit(x_train,y_train)
    scores = clf.cv_results_['mean_test_score']
    #print max(scores)
    print clf.best_params_
    #print clf.score(x_test,y_test)
    output_row = ['svm_polynomial',max(scores),clf.score(x_test,y_test)]
    output.append(output_row)
    
    # SVM with RBF Kernel
    print '### SVM with RBF Kernel ###'
    parameters = {'C':[0.1,0.5,1.,5.,10.,50.,100.],'gamma':[.1,.5,1.,3.,6.,10.]}
    svr = svm.SVC(kernel='rbf')
    clf = GridSearchCV(svr,parameters,cv=5,scoring='accuracy')
    clf.fit(x_train,y_train)
    scores = clf.cv_results_['mean_test_score']
    #print max(scores)
    print clf.best_params_
    #print clf.score(x_test,y_test)
    output_row = ['svm_rbf',max(scores),clf.score(x_test,y_test)]
    output.append(output_row)
    
    # Logistic Regression
    print '### Logistic Regression ###'
    parameters = {'C':[0.1,0.5,1.,5.,10.,50.,100.]}
    logistic = linear_model.LogisticRegression()
    clf = GridSearchCV(logistic,parameters,cv=5,scoring='accuracy')
    clf.fit(x_train,y_train)
    scores = clf.cv_results_['mean_test_score']
    #print max(scores)
    print clf.best_params_
    #print clf.score(x_test,y_test)
    output_row = ['logistic',max(scores),clf.score(x_test,y_test)]
    output.append(output_row)
    
    
    # k-Nearest Neighbors
    print '### k-Neareat Neighbors ###'
    parameters = {'n_neighbors':range(1,51),'leaf_size':range(5,61,5)}
    nbrs = KNeighborsClassifier()
    clf = GridSearchCV(nbrs,parameters,cv=5,scoring='accuracy')
    clf.fit(x_train,y_train)
    scores = clf.cv_results_['mean_test_score']
    #print max(scores)
    print clf.best_params_
    #print clf.score(x_test,y_test)
    output_row = ['knn',max(scores),clf.score(x_test,y_test)]
    output.append(output_row)
    
    # Decision Trees
    print '### Decision Trees ###'
    parameters = {'max_depth':range(1,51),'min_samples_split':range(2,11)}
    dtree = tree.DecisionTreeClassifier()
    clf = GridSearchCV(dtree,parameters,cv=5,scoring='accuracy')
    clf.fit(x_train,y_train)
    scores = clf.cv_results_['mean_test_score']
    #print max(scores)
    print clf.best_params_
    #print clf.score(x_test,y_test)
    output_row = ['decision_tree',max(scores),clf.score(x_test,y_test)]
    output.append(output_row)
    
    # Random Forest
    print '### Random Forest ###'
    parameters = {'max_depth':range(1,51),'min_samples_split':range(2,11)}
    rfc = RandomForestClassifier(n_estimators=10)
    clf = GridSearchCV(rfc,parameters,cv=5,scoring='accuracy')
    clf.fit(x_train,y_train)
    scores = clf.cv_results_['mean_test_score']
    #print max(scores)
    print clf.best_params_
    #print clf.score(x_test,y_test)
    output_row = ['random_forest',max(scores),clf.score(x_test,y_test)]
    output.append(output_row)
    
    with open(out_path,'w') as f:
        writer = csv.writer(f,delimiter = ',')
        writer.writerows(output)
    
    
if __name__ == "__main__":
    main()