#! /home/songwei/anaconda3/bin/python
# -*- coding: utf-8 -*-

#print(__doc__)

import time

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from io import StringIO
from sklearn import tree
#import pydot
# import random
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#import svmMLiA
from sklearn.externals import joblib
import numpy as np
## now you can save it to a file
# joblib.dump(clf, 'filename.pkl')
## and later you can load it
#

def loadDataSet():
    dataMat = []; labelMat = []
#    fr = open('qqxd-ali.csv')
#    qqxd - ali.csv
#    fr = open('An_off.csv')
#    fr = open('SEdel.csv')
#    fr = open('Sh.csv')
    fr = open('SW_OldOD.csv')
#    k=0
    for line in fr.readlines():
#        k+=1
#        print (line)
#        print ('\n', k)
        lineArr = line.strip().split('\t')
#        print (lineArr)
        linelist=[]
        for i in range(len(lineArr)-1):
            b = lineArr[i].strip()
            #linelist.append( float(b) )
            linelist.append(b)
 #       print (linelist, len(lineArr))
        dataMat.append(linelist)

        labelMat.append(int(lineArr[-1]))
#        print (labelMat)
    fr.close()
    return dataMat,labelMat

def deleteZeroCol(dataCol):
    zero_ratio = 1
    a = np.array(dataCol)
    print (a.shape)
    zeronum = zero_ratio*(a.shape[0]-96)
    print (zeronum)
    columns = (a == '0').sum(0)
#    columns = (a == 0).sum(0)

#    rows    = (a == 0).sum(1)

    print (columns)
    print (len(columns))
#    print (rows)

    deletecol=[]
    for i in range(79, len(columns)) :
        if columns[i]>zeronum :
            deletecol.append(i)
    print()
    delMate = np.delete(a, deletecol, axis=1)
    print (40*'-')
    print ('delete:',deletecol,' ', len(deletecol))
    print (delMate.shape)
    return delMate



def loadDataSetForOneClass(file):
    dataMat = [];
#    fr = open('qqtest.csv')
    fr = open(file)
#    k=0
    for line in fr.readlines():
#        k+=1
#        print (line)
#        print ('\n', k)
        lineArr = line.strip().split()
#        print (lineArr)
        linelist=[]
        for i in range(len(lineArr)):
            b = lineArr[i].strip()
            linelist.append( float(b) )
 #       print (linelist, len(lineArr))
        dataMat.append(linelist)

#        print (labelMat)
#    print ( dataMat )
#    print (len(dataMat))
    return dataMat


def useSVM(X, y, testX, testy, testid):
    num=0
    goodf=1
    badf = 1-goodf
#    print (y)
    for i in range(len(y)):
#        if ( y[i] == -1 ):
        if ( y[i] == goodf ):
            num+=1
#    print (num)
#    if ( goodf == 1 ):
#    weight = int(len(y)/num+0.5)
#    weight = int((len(y)-num)/num+0.5)
    weight = float((len(y)-num)/num)

#    else:
#        weight = 1/int(len(y) / num + 0.5)
#    weight = 1/

    print ('weight - ',len(y)-num, num, weight)

    kf = 'rbf'
#    kf = 'linear'

#    clf = svm.SVC(kernel='linear', C=1.0, random_state=0)  # 用线性核，你也可以通过kernel参数指定其它的核。
#    clf = svm.SVC(kernel='linear', C=1.0, random_state=0, class_weight={1: 1/weight})  # 用线性核，你也可以通过kernel参数指定其它的核。
#    clf = svm.SVC(class_weight={1: weight})
#    clf = svm.LinearSVC(class_weight={weight: 1})

#    '''0
#    clf = svm.SVC(kernel=kf, C=1.0, random_state=0, class_weight={1: 1/weight})  # 用线性核，你也可以通过kernel参数指定其它的核。
    '''
    if (goodf == 1):
        clf = svm.SVC(kernel=kf, C=1.0, random_state=0, class_weight={1 : 1/weight})
    else:
        clf = svm.SVC(kernel=kf, C=100, random_state=0, class_weight={1 : weight})
    '''
    clf = svm.SVC(kernel=kf, C=100, random_state=0, class_weight={0: weight})

    clf.fit(X, y)

    joblib.dump(clf, 'qqml_svm.pkl')
    '''
    clf = joblib.load('qqml_svm.pkl')
   '''

    tp=0; fn=0; fp=0; tn=0

    print(clf.predict(X[0].reshape(1,-1)))


#    print(clf.get_params())

    print(clf.score(testX,testy))
#    print ('importance:', clf.feature_importances_)
    truep = []
    falsep = []
    trueneg = []
    falseneg = []

    truepstr = []
    falsepstr = []
    truenegstr = []
    falsenegstr = []


    err = 0
    for i in range(len(testy)):
#        ret = clf.predict(testX[i])
        tX = testX[i].reshape(1, -1)
        tid = testid[i]
        ret = clf.predict(tX)
        decis = clf.decision_function(tX)[0]

#        if ( np.sign(ret) != np.sign(clf.decision_function(testX[i]) ) ):
#            print(ret[0], 'distance:', clf.decision_function(testX[i])[0], testy[i] )
        if ( ret != testy[i] ):
            err += 1

        if ( testy[i] == goodf ):
            if ( ret == goodf ):
                #print(testy[i], ret[0], 'distance:', clf.decision_function(testX[i])[0])
                truepstr.append('%s\n'%tid)
                truep.append('%s, %f\n'%(tid,decis))
                tp+=1
            else:
                #print(testy[i], ret[0], 'distance:', clf.decision_function(testX[i])[0])
                falsenegstr.append('%s\n' % tid)
                falseneg.append('%s, %f\n'%(tid,decis))
                fn+=1
        else:
            if ( ret == goodf ):
#                print(testy[i], ret[0], 'distance:', clf.decision_function(testX[i])[0])
                falsepstr.append('%s\n' % tid)
                falsep.append('%s, %f\n'%(tid,decis))
                fp+=1
            else:
#                print(testy[i], ret[0], 'distance:', clf.decision_function(testX[i])[0])
                truenegstr.append('%s\n' % tid)
                trueneg.append('%s, %f\n'%(tid,decis))
                tn+=1

#    print ( 'error : %d in %d test cases. the err percent: %d'%(err, trainnum, (err/trainnum)) )
    print (kf, "goodflag:", goodf)
    print(err, len(testy), 100-(err/len(testy))*100)

    print ('tp:', tp, 'fn:', fn, 'fp:', fp, 'tn:', tn)

    str = 'Precision(TruePositibeRate tp/(tp+fp) )=%f' % (tp / (tp + fp))
    print(str)

    str = 'Recall tp/(tp + fn) =%f' % (tp / (tp + fn))
    print(str)

    str = 'FalsePositiveRate fp/(fp + tn)=%f' % (fp / (fp + tn))
    print(str)

#    print (truep)

#    print ('tp', max(truep), min(truep))
#    print ('fn', max(falseneg), min(falseneg))
#    print ('tn', max(trueneg), min(trueneg))
#    print ('fp', max(falsep), min(falsep))

#    print ('tp', max(truep), min(truep))
    if ( truep != [] ):
        print ('tp', max(truep), min(truep))
#    print ('fn', max(falseneg), min(falseneg))
    if ( falseneg != [] ):
        print ('fn', max(falseneg), min(falseneg))
#    print ('tn', max(trueneg), min(trueneg))
    if ( trueneg != [] ):
        print ('tn', max(trueneg), min(trueneg))
#    print ('fp', max(falsep), min(falsep))
    if ( falsep != [] ):
        print ('tn', max(trueneg), min(trueneg))



#'''
    fp = open ('SVtruep.txt', 'w')
    fp.writelines(truep)
    fp.close()

    fp = open ('SVfalsep.txt', 'w')
    fp.writelines(falsep)
    fp.close()

    fp = open ('SVtrueneg.txt', 'w')
    fp.writelines(trueneg)
    fp.close()

    fp = open ('SVfalseneg.txt', 'w')
    fp.writelines(falseneg)
    fp.close()
#'''

def useRForest(X, y, testX, testy, testid):
    num=0
    goodf=1
    badf = 1-goodf
#    print (y)
    for i in range(len(y)):
#        if ( y[i] == -1 ):
        if ( y[i] == badf ):
            num+=1
#    print (num)
#    if ( goodf == 1 ):
    weight = int(len(y)/num+0.5)
#    else:
#        weight = 1/int(len(y) / num + 0.5)
#    weight = 1/

    print ('weight - ',len(y), num, weight)

#    kf = 'rbf'
#    kf = 'linear'

#    clf = svm.SVC(kernel='linear', C=1.0, random_state=0)  # 用线性核，你也可以通过kernel参数指定其它的核。
#    clf = svm.SVC(kernel='linear', C=1.0, random_state=0, class_weight={1: 1/weight})  # 用线性核，你也可以通过kernel参数指定其它的核。
#    clf = svm.SVC(class_weight={1: weight})
#    clf = svm.LinearSVC(class_weight={weight: 1})

#    '''0
#    clf = svm.SVC(kernel=kf, C=1.0, random_state=0, class_weight={1: 1/weight})  # 用线性核，你也可以通过kernel参数指定其它的核。
    '''
    if (goodf == 1):
        clf = svm.SVC(kernel=kf, C=1.0, random_state=0, class_weight={1 : 1/weight})
    else:
        clf = svm.SVC(kernel=kf, C=1.0, random_state=0, class_weight={1 : weight})
    '''
#    clf = svm.SVC(kernel=kf, C=1.0, random_state=0, class_weight={1: 1/weight})

    clf = RandomForestClassifier(n_estimators=30)

    clf.fit(X, y)

    joblib.dump(clf, 'qqml_rf.pkl')



    '''
    clf = joblib.load('qqml_rf.pkl')
   '''

    importances = clf.feature_importances_
#    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


    tp=0; fn=0; fp=0; tn=0

    print(clf.predict(X[0].reshape(1,-1)))
#    print(clf.predict(X[0]))


#    print(clf.get_params())

    print(clf.score(testX,testy))
#    print ('importance:', clf.feature_importances_)
    truep = []
    falsep = []
    trueneg = []
    falseneg = []

    truepstr = []
    falsepstr = []
    truenegstr = []
    falsenegstr = []


    err = 0
    for i in range(len(testy)):
        tX = testX[i].reshape(1,-1)
        tid = testid[i]
        ret = clf.predict(tX)
        prob = clf.predict_proba(tX)
#        print(ret, prob, prob[0][0], prob[0][1])
#        if ( np.sign(ret) != np.sign(clf.decision_function(testX[i]) ) ):
#            print(ret[0], 'distance:', clf.decision_function(testX[i])[0], testy[i] )
        if ( ret != testy[i] ):
            err += 1

        if ( testy[i] == goodf ):
            if ( ret == goodf ):
                #print(testy[i], ret[0], 'distance:', clf.decision_function(testX[i])[0])
                truepstr.append('%s\n'%tid)
                truep.append('%s, %f\n'%(tid,prob[0][goodf]))
                tp+=1
            else:
                #print(testy[i], ret[0], 'distance:', clf.decision_function(testX[i])[0])
                falsenegstr.append('%s\n'%tid)
                falseneg.append('%s, %f\n'%(tid,prob[0][badf]))
                fn+=1
        else:
            if ( ret == goodf ):
#                print(testy[i], ret[0], 'distance:', clf.decision_function(testX[i])[0])
                falsepstr.append('%s\n'%tid)
                falsep.append('%s, %f\n'%(tid,prob[0][goodf]))
                fp+=1
            else:
#                print(testy[i], ret[0], 'distance:', clf.decision_function(testX[i])[0])
                truenegstr.append('%s\n'%tid)
                trueneg.append('%s, %f\n'%(tid,prob[0][badf]))
                tn+=1

#    print ( 'error : %d in %d test cases. the err percent: %d'%(err, trainnum, (err/trainnum)) )
    print ("goodflag:", goodf)
    print(err, len(testy), 100-(err/len(testy))*100)

    print ('tp:', tp, 'fn:', fn, 'fp:', fp, 'tn:', tn)

    str = 'Precision(TruePositibeRate tp/(tp+fp) )=%f' % (tp / (tp + fp))
    print(str)

    str = 'Recall tp/(tp + fn) =%f' % (tp / (tp + fn))
    print(str)

    str = 'FalsePositiveRate fp/(fp + tn)=%f' % (fp / (fp + tn))
    print(str)

#    print (truep)

#    print ('tp', max(truep), min(truep))
    if ( truep != [] ):
        print ('tp', max(truep), min(truep))
#    print ('fn', max(falseneg), min(falseneg))
    if ( falseneg != [] ):
        print ('fn', max(falseneg), min(falseneg))
#    print ('tn', max(trueneg), min(trueneg))
    if ( trueneg != [] ):
        print ('tn', max(trueneg), min(trueneg))
#    print ('fp', max(falsep), min(falsep))
    if ( falsep != [] ):
        print ('tn', max(trueneg), min(trueneg))

#'''
    fp = open ('RFtruep.txt', 'w')
    fp.writelines(truep)
    fp.close()

    fp = open ('RFfalsep.txt', 'w')
    fp.writelines(falsep)
    fp.close()

    fp = open ('RFtrueneg.txt', 'w')
    fp.writelines(trueneg)
    fp.close()

    fp = open ('RFfalseneg.txt', 'w')
    fp.writelines(falseneg)
    fp.close()
#'''

def useOneClassSVM(Xt, Xs, outlier):
#    '''
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='auto')
    clf.fit(Xt)

    joblib.dump(clf, 'qqml.pkl')
    '''
    clf = joblib.load('qqml.pkl')
   '''
    y_pred_train = clf.predict(Xt)
    y_pred_test = clf.predict(Xs)
    y_pred_outliers = clf.predict(outlier)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    print("error train: %d/%d ; errors novel regular: %d/%d ; "
        "errors novel abnormal: %d/%d"
        % (n_error_train, y_pred_train.size, n_error_test, y_pred_test.size, n_error_outliers, y_pred_outliers.size))


def plot_dt(model, filename):
    str_buffer = StringIO()
#    tree.export_graphviz(model, out_file=str_buffer)
    tree.export_graphviz(model, out_file ='tree.dot')

#    graph = pydot.graph_from_dot_data(str_buffer.getvalue())
#    graph.write_jpg(filename)

def useDecisionTree(X, y, testX, testy):
    num = 0
    for i in range(len(y)):
        if (y[i] == 0):
            num += 1
    weight = int(len(y) / num + 0.5)
    #    weight = 1/

    print('weight - ', len(y), num, weight)

    dt = DecisionTreeClassifier(max_depth=5,class_weight={1: 1/weight})
#     plot_dt(dt, "myfile.png")

    tp=0; fn=0; fp=0; tn=0

    print(dt.score(testX, testy))
    print(dt.predict(X[0]))

    #    print(clf.get_params())

    print(dt.score(testX, testy))

    err = 0
    for i in range(len(testy)):
        ret = dt.predict(testX[i])
        if (ret != testy[i]):
            err += 1

        if (testy[i] == 1):
            if (ret == 1):
                tp += 1
            else:
                fn += 1
        else:
            if (ret == 1):
                fp += 1
            else:
                tn += 1

                #    print ( 'error : %d in %d test cases. the err percent: %d'%(err, trainnum, (err/trainnum)) )
    print(err, len(testy), 100 - (err / len(testy)) * 100)

    print('tp:', tp, 'fn:', fn, 'fp:', fp, 'tn:', tn)

    str = 'Precision(TruePositibeRate tp/(tp+fp) )=%f' % (tp / (tp + fp))
    print(str)

    str = 'Recall tp/(tp + fn) =%f' % (tp / (tp + fn))
    print(str)

    str = 'FalsePositiveRate fp/(fp + tn)=%f'%(fp / (fp + tn))
    print(str)


def main():
#    boston = datasets.load_boston()
#    print(boston.DESCR)

#    X, y = datasets.make_classification()
#    print (X, y)
#    base_svm = SVC()
#    base_svm.fit(X, y)
    dataMat, labelMat = loadDataSet()
#    print (labelMat)
#    print (dataMat)
#    exit(0)
#    trainnum = int(len(labelMat)*0.5)
#    trainArr=np.array(dataMat)
    #trainLab = np.array(labelMat)

    __comment='''
    if ( trainnum == 0 ):
        trainnum = len(labelMat)
        X=np.mat(trainArr[:trainnum])
        y=labelMat[0:trainnum]
        testX = X
        testy = y
    else:
        X = np.mat(trainArr[:trainnum])
        y = labelMat[0:trainnum]
        testX = np.mat(trainArr[trainnum:])
        testy = labelMat[trainnum:]

    '''
    __comment='''
    trainx = []
    y = []
    if (trainnum == 0):
        trainnum = len(labelMat)
        X = np.mat(trainArr[:trainnum])
        y = labelMat[0:trainnum]
        testX = X
        testy = y
    else:
        for i in range(trainnum):
            index = random.randint(0, len(labelMat)-1 )
#            print (len(labelMat) - 1,index)
            trainx.append(dataMat.pop(index))
            y.append(labelMat.pop(index))
    X = np.mat(trainx)
    testX = np.mat(dataMat)
    testy = labelMat
    '''


#    X = np.mat(dataMat)
#    y = labelMat

    X_train, X_test, y_train, y_test = train_test_split(dataMat, labelMat, test_size=0.5,
                                                         random_state=0)  # 为了看模型在没有见过数据集上的表现，随机拿出数据集中50%的部分做测试


    print(len(X_train), len(X_test))
#    print(X_train[0])
    # 为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
    #from sklearn.preprocessing import StandardScaler

#    '''
    sc = StandardScaler()
    sc.fit(X_train)  # 估算每个特征的平均值和标准差
    joblib.dump(sc, 'qqml_sc.pkl')

#    sc = joblib.load('qqml_sc.pkl')
#    print(X_train[0])

    X_train_std = sc.transform(X_train)

    # 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性
    X_test_std = sc.transform(X_test)

    #    X, y = datasets.make_classification(1000, 20, n_informative=3)
#'''
#   X_train_std = X_train
#    X_test_std = X_test

    __comment='''
    pca = PCA(n_components=2)

    X_train_reduced = pca.fit_transform(X_train_std)
    print (X_train_reduced[0])
#    X_test_reduced = pca.transform(X_test_std)
    '''


#    print(X_train[0])

#    X_train = np.mat(X_train)
#    X_test = np.mat(X_test)

#    X_train_std = np.mat(X_train_std)
#    X_test_std = np.mat(X_test_std)
#    X_train_std = np.mat(X_train)
#    X_test_std = np.mat(X_test)


#    X_train_reduced = np.mat(X_train_reduced)
#    X_test_reduced = np.mat(X_test_reduced)


#    print ("Train: ", X_train_std.shape, X_train.shape, set(y_train), len (y_train))
#    print ("Test: ", X_test_std.shape, X_test.shape, set(y_test), len (y_test))

    print ("Train: ", X_train_std.shape, set(y_train), len (y_train))
    print ("Test: ", X_test_std.shape, set(y_test), len (y_test))

    print (X_train[0])
    print (X_train_std[0])
#    print (X[0])
    print (y_train[0])
#    print (y_train)d

#    print(X_train_reduced.shape)
#    print(X_test_reduced.shape)



#    useDecisionTree(X_train_std, y_train, X_test_std, y_test)
#    useDecisionTree(X_train_reduced, y_train, X_test_reduced, y_test)
#    useSVM(X_train_reduced, y_train, X_test_reduced, y_test)
#    useSVM(X_train, y_train, X_test, y_test)
#    useSVM(X_train_std, y_train, X_test_std, y_test)
    useSVM(X_train_std, y_train, X_test_std, y_test)

def mainRForestSVM():

    dataMat, labelMat = loadDataSet()
#    dataMat = deleteZeroCol( dataMat )


#    X_outlier = loadDataSetForOneClass('qqOut.csv')

    X_tr, X_test, y_train, y_test = train_test_split(dataMat, labelMat, test_size=0.5,
                                                        random_state=0)  # 为了看模型在没有见过数据集上的表现，随机拿出数据集中50%的部分做测试

    X_trArr = np.array(X_tr)
    print(len(X_tr), len(X_test))
#    print(X_trArr)
#    print("==========================")
    print('shape:', X_trArr.shape)

    X_train = X_trArr[:, 1:]


    print('shape:', X_train.shape)

#    print (type(X_train))
#    print(len(X_train))
#    print(X_train)

    X_train = X_train.astype(float)
    print ('X_train[0]\n', X_train[0])

#    print (X_test)
    X_testArr = np.array(X_test)
    XtestId = X_testArr[:, :1]
    X_test = X_testArr[:, 1:]
    print('XtestId[0]\n', XtestId[0])
#    print(X_test)

    X_test = X_test.astype(float)
    print ('X_test[0]\n', X_test[0])
#    X_test = float(X_test)
#    print (X_test)
    # 为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
    #from sklearn.preprocessing import StandardScaler

#'''
    sc = StandardScaler()
    sc.fit(X_train)  # 估算每个特征的平均值和标准差
    joblib.dump(sc, 'qqml_sc.pkl')

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    print ('X_train_std[0]\n', X_train_std[0])
    print ('X_test_std[0]\n', X_test_std[0])

    print ('shape:', X_train_std.shape)
#'''

    # sc = joblib.load('qqml_sc.pkl')

#    X, y = datasets.make_classification(1000, 20, n_informative=3)
#    X_train_std = X_train
#    X_test_std = X_test


#    print(X_train[0])

#    X_train = np.mat(X_train)
#    X_test = np.mat(X_test)
#    X_outlier = np.mat( X_outlier )


#    X_train_std = np.mat(X_train_std)
#    X_test_std = np.mat(X_test_std)

#    X_train_reduced = np.mat(X_train_reduced)


#    print (X_train_std.shape, X_train.shape)
#    print (X_test_std.shape, X_test.shape)

#    print (X_train[0])
#    print (X_train_std[0])

#    print (X[0])
#    print (y_train[0])
#    print (y_train)d

#    print(X_train_reduced.shape)
#    print(X_test_reduced.shape)



#    useDecisionTree(X_train_std, y_train, X_test_std, y_test)
#    useDecisionTree(X_train_reduced, y_train, X_test_reduced, y_test)
#    useSVM(X_train_reduced, y_train, X_test_reduced, y_test)
#    useSVM(X_train, y_train, X_test, y_test)
#    useSVM(X_train_std, y_train, X_test_std, y_test)

    useRForest(X_train_std, y_train, X_test_std, y_test, XtestId)

    useSVM(X_train_std, y_train, X_test_std, y_test, XtestId)


def mainRForest():

    dataMat, labelMat = loadDataSet()
#    X_outlier = loadDataSetForOneClass('qqOut.csv')

    X_train, X_test, y_train, y_test = train_test_split(dataMat, labelMat, test_size=0.5,
                                                        random_state=0)  # 为了看模型在没有见过数据集上的表现，随机拿出数据集中50%的部分做测试

    print(len(X_train), len(X_test))
    # 为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
    #from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.fit(X_train)  # 估算每个特征的平均值和标准差
    joblib.dump(sc, 'qqml_sc.pkl')

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)


    # sc = joblib.load('qqml_sc.pkl')

#    X, y = datasets.make_classification(1000, 20, n_informative=3)
#'''
#    X_train_std = X_train
#    X_test_std = X_test


#    print(X_train[0])

#    X_train = np.mat(X_train)
#    X_test = np.mat(X_test)
#    X_outlier = np.mat( X_outlier )


#    X_train_std = np.mat(X_train_std)
#    X_test_std = np.mat(X_test_std)

#    X_train_reduced = np.mat(X_train_reduced)


#    print (X_train_std.shape, X_train.shape)
#    print (X_test_std.shape, X_test.shape)

    print (X_train[0])
    print (X_train_std[0])
#    print (X[0])
#    print (y_train[0])
#    print (y_train)d

#    print(X_train_reduced.shape)
#    print(X_test_reduced.shape)



#    useDecisionTree(X_train_std, y_train, X_test_std, y_test)
#    useDecisionTree(X_train_reduced, y_train, X_test_reduced, y_test)
#    useSVM(X_train_reduced, y_train, X_test_reduced, y_test)
#    useSVM(X_train, y_train, X_test, y_test)
#    useSVM(X_train_std, y_train, X_test_std, y_test)
    useRForest(X_train_std, y_train, X_test_std, y_test)

#    useSVM(X_train_std, y_train, X_test_std, y_test)


def mainForOne():

    dataMat = loadDataSetForOneClass('qqOne.csv')
    X_outlier = loadDataSetForOneClass('qqOut.csv')

    test_size = 0.7
    splitNo = round(test_size * len(dataMat))
    X_train = dataMat[:splitNo]
    X_test = dataMat[splitNo:]


    # 为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
    #from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.fit(X_train)  # 估算每个特征的平均值和标准差
    joblib.dump(sc, 'qqml_sc.pkl')

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_outlier_std = sc.transform(X_outlier)


    # sc = joblib.load('qqml_sc.pkl')

#    X, y = datasets.make_classification(1000, 20, n_informative=3)
#'''
#    X_train_std = X_train
#    X_test_std = X_test


#    print(X_train[0])

    X_train = np.mat(X_train)
    X_test = np.mat(X_test)
    X_outlier = np.mat( X_outlier )


    X_train_std = np.mat(X_train_std)
    X_test_std = np.mat(X_test_std)
    X_outlier_std = np.mat(X_outlier_std)

#    X_train_reduced = np.mat(X_train_reduced)
#    X_test_reduced = np.mat(X_test_reduced)


    print (X_train_std.shape, X_train.shape)
    print (X_test_std.shape, X_test.shape)
    print (X_outlier_std.shape, X_outlier.shape)

    print (X_train[0])
    print (X_train_std[0])
#    print (X[0])
#    print (y_train[0])
#    print (y_train)d

#    print(X_train_reduced.shape)
#    print(X_test_reduced.shape)



#    useDecisionTree(X_train_std, y_train, X_test_std, y_test)
#    useDecisionTree(X_train_reduced, y_train, X_test_reduced, y_test)
#    useSVM(X_train_reduced, y_train, X_test_reduced, y_test)
#    useSVM(X_train, y_train, X_test, y_test)
#    useSVM(X_train_std, y_train, X_test_std, y_test)
    useOneClassSVM(X_train_std, X_test_std, X_outlier_std)

'''
def dydotTest():
    g = pydot.Dot('mygraph', g_type='dig')  # 创建有向图

    node = pydot.Node(1)  # 创建节点'1'，与pydot.Node('1')等价
    g.add_node(node)  # 添加节点
    print
    node.get_name()  # 输出为字符串1

    node = pydot.Node(2, label='bbb')  # 如果设置label属性，那么画图时节点显示为'bbb'
    g.add_node(node)
    print(node.get_name())  # 输出为字符串2

    node = pydot.Node(3, label='ccc')
    g.add_node(node)

    e = pydot.Edge('1', '2')  # 创建边1->2
    g.add_edge(e)  # 添加边

    e = pydot.Edge('1', '3')  # 创建边1->3
    g.add_edge(e)  # 添加边

    print(g.to_string())  # 打印整个图
    g.write_jpg("mygraph.jpg")  # 保存图形到文件

def mainmy():
    dataArr, labelArr = loadDataS et()
    X_train, X_test, y_train, y_test = train_test_split(dataArr, labelArr, test_size=0.75, random_state=0)
#    b, alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)

    sc = StandardScaler()
    sc.fit(X_train)  # 估算每个特征的平均值和标准差
    X_train_std = sc.transform(X_train)
# 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性
    X_test_std = sc.transform(X_test)

    print( np.mat(X_train).shape, set(y_train), len(y_train))
    print( np.mat(X_test).shape, set(y_test), len(y_test))

    print(X_train[0])
    print(X_train_std[0])
    #    print (X[0])
    print(y_train[0])
    k1 = 1.3
#    kf = 'lin'
    kf = 'rbf'
    C=50
    toler = 0.0001
    b, alphas = svmMLiA.smoP(X_train_std, y_train, C, toler, 200, (kf, k1))
    print (b, alphas[alphas>0])
#    ws = svmMLiA.calcWs(alphas, X_train_std, y_train)
#    print (ws)


    datMat = np.mat(X_train_std);
    labelMat = np.mat(y_train).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]  # get matrix of only support vectors
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = svmMLiA.kernelTrans(sVs, datMat[i, :], (kf, k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = svmMLiA.kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))

#    X_train_stdMat = np.mat(X_train_std)
    kernelEval = svmMLiA.kernelTrans(sVs, datMat[0, :], (kf, k1))
    predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
    print ('predict=', predict, ' ------ ', y_train[0])

    X_test_stdMat = np.mat(X_test_std)
    kernelEval = svmMLiA.kernelTrans(sVs, X_test_stdMat[0, :], (kf, k1))
    predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
    print('predict=', predict, ' ------ ', y_test[0])

    ttp=0; tfn=0; tfp=0; ttn=0

    for i in range(m):
#        if ( labelArr[i] ) < 0 :
            kernelEval = svmMLiA.kernelTrans(sVs, datMat[i, :], (kf, k1))
            predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
            print(i, ' : ', labelArr[i], ' : ', predict)
            if np.sign(predict) != np.sign(labelArr[i]):
                if ( np.sign(labelArr[i]) >0 ):
                    tfp += 1
                else:
                    tfn += 1
            else:
                if ( np.sign(labelArr[i]) >0 ):
                    ttp += 1
                else:
                    ttn += 1

    l, k = np.shape(X_test_stdMat)
    tp=0; fn=0; fp=0; tn=0
    for i in range(l):
#        if ( labelArr[i] ) < 0 :
            kernelEval = svmMLiA.kernelTrans(sVs, X_test_stdMat[i, :], (kf, k1))
            predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
            print ( i, ' : ', y_test[i], ' : ', predict)
            if np.sign(predict) != np.sign(labelArr[i]):.
                if (np.sign(labelArr[i]) > 0):
                    fp += 1
                else:
                    fn += 1
            else:
                if (np.sign(labelArr[i]) > 0):
                    tp += 1
                else:
                    tn += 1
    print ('kf:', kf, '  C:', C, ' toler:', toler)
    print ('train : tp:', ttp, ' fp:', tfp, ' tn:', ttn, ' fn:', tfn, ' total:', ttp+tfp+ttn+tfn)
    print ('test : tp:', tp, ' fp:', fp, ' tn:', tn, ' fn:', fn, ' total:', tp+fp+tn+fn)
'''
if __name__ == '__main__':
    stime =  time.time();
#    main()
#    mainRForest()
    mainRForestSVM()

#    mainForOne()
#    mainmy()

    etime =  time.time();
    print ('\nExecute %d sec, %d min, %d hou' %(etime-stime, (etime-stime)/60, (etime-stime)/3600))
