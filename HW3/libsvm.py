import numpy as np
from svmutil import *
import pandas as pd
import operator
import time

def start_time():
    start_time = time.time()
    return start_time

def end_time(start_time):
    print "--- %s seconds ---" % (time.time() - start_time)
    return time.time() - start_time

def distinct_values(x):
    values=[]
    for j in range(0, x.shape[1]):
        values.append({})
        for i in range(0, x.shape[0]):
            if x[i][j] not in values[j]:
                values[j][x[i][j]] = 1
            else:
                values[j][x[i][j]] = values[j][x[i][j]] + 1
    return values

def predict_RBF(gamma,C):
    print 'predicting best'
    print '\n gamma=', gamma, 'C=',C,
    y = y_train.tolist()
    x = x_new_train.tolist()
    y_t = y_test.tolist()
    x_t = x_new_test.tolist()
    prob = svm_problem(y, x)
    param = svm_parameter('-t 2 -h 0 -c ' + str(C) + ' -g ' + str(gamma))
    m = svm_train(prob, param)
    # svm_save_model('linear_scale.model', m)
    # m = svm_load_model('linear_scale.model')
    print 'Training Accuracy'
    p_label, p_acc, p_val = svm_predict(y, x, m)
    ACC, MSE, SCC = evaluations(y, p_label)
    print ACC, MSE, SCC
    print 'Testing Accuracy'
    p_label, p_acc, p_val = svm_predict(y_t, x_t, m)
    ACC, MSE, SCC = evaluations(y_t, p_label)
    print ACC, MSE, SCC

def predict_poly(degree,C):
    print 'predicting best'
    y = y_train.tolist()
    x = x_new_train.tolist()
    y_t = y_test.tolist()
    x_t = x_new_test.tolist()
    prob = svm_problem(y, x)
    param = svm_parameter('-t 2 -h 0 -c ' + str(C) + ' -d ' + str(degree))
    m = svm_train(prob, param)
    # m = svm_model(prob, param)
    # svm_save_model('linear_scale.model', m)
    # m = svm_load_model('linear_scale.model')
    p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
    ACC, MSE, SCC = evaluations(y, p_label)
    print '\n\n\n', gamma, C,


def process_data(x):
    print 'Data preprosessing', x.shape, x.shape[1]
    del_col = []
    x_new_train = x.copy()
    values = distinct_values(x)
    for j in range(0, x.shape[1]):
        count = len(values[j])
        if count == 3:
            del_col.append(j)
        #print j, count, values[j]
        arr_0 = np.zeros(x.shape[0])
        arr_1 = np.zeros(x.shape[0])
        arr_2 = np.zeros(x.shape[0])
        for i in range(0, x.shape[0]):
            if count == 2:
                # print ' value \t',i,j,x[i][j],x_new_train[i][j]
                if x[i][j] == -1:
                    x_new_train[i][j] = 0

            if count == 3:
                if x[i][j] == -1:
                    arr_0[i] = 1
                if x[i][j] == 0:
                    arr_1[i] = 1
                if x[i][j] == 1:
                    arr_2[i] = 1
        if count == 3:
            #print 'before', x_new_train.shape, arr_0.shape, arr_1.shape, arr_2.shape
            arr_0 = np.vstack((arr_0, arr_1, arr_2))
            arr_0 = arr_0.T
            #print 'after', x_new_train.shape, arr_0.shape, arr_1.shape, arr_2.shape
            x_new_train = np.hstack((x_new_train, arr_0))
            #print 'after', x_new_train.shape, arr_0.shape, arr_1.shape, arr_2.shape
            # np.hstack((x_new_train, arr_2))
    #print del_col
    #print x_new_train.shape
    x_new_train = np.delete(x_new_train, del_col, axis=1)
    #print x_new_train.shape
    values = distinct_values(x_new_train)
    #print values
    return x_new_train

def linear_SVM():
    CV_ACC={}
    for a in range(-6,3):
        C=np.power(4.0,a)
        y=y_train.tolist()
        x=x_new_train.tolist()
        y_t=y_test.tolist()
        x_t=x_new_test.tolist()
        prob = svm_problem(y,x)
        param = svm_parameter('-t 0 -v 3 -h 0 -c '+str(C))
        m = svm_train(prob,param)
        #m = svm_model(prob, param)
        #svm_save_model('linear_scale.model', m)
        #m = svm_load_model('linear_scale.model')
        #p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
        #ACC, MSE, SCC = evaluations(y, p_label)
        print '\n\n\n', a, C,
        print 'CV_ACC1', m
        CV_ACC[C]=m
        #print p_label
    return CV_ACC

def poly_SVM():
    CV_ACC ={}
    for degree in range(1,4):
        for a in range(-3,8):
            C=np.power(4.0,a)
            y=y_train.tolist()
            x=x_new_train.tolist()
            y_t=y_test.tolist()
            x_t=x_new_test.tolist()
            prob = svm_problem(y,x)
            param = svm_parameter('-t 1 -v 3 -h 0 -c '+str(C) +' -d '+str(degree))
            m = svm_train(prob,param)
            #m = svm_model(prob, param)
            #svm_save_model('linear_scale.model', m)
            #m = svm_load_model('linear_scale.model')
            #p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
            #ACC, MSE, SCC = evaluations(y, p_label)
            print '\n\n\n', a, C,
            print 'CV_ACC2', m
            CV_ACC[(degree,C)] = m
    return CV_ACC


def rbf_SVM():
    CV_ACC={}
    for b in range(-7,0):
        gamma=np.power(4.0,b)
        for a in range(-3,8):
            C=np.power(4.0,a)
            y=y_train.tolist()
            x=x_new_train.tolist()
            y_t=y_test.tolist()
            x_t=x_new_test.tolist()
            prob = svm_problem(y,x)
            param = svm_parameter('-t 2 -v 3 -h 0 -c '+str(C)+' -g '+str(gamma))
            m = svm_train(prob,param)
            #m = svm_model(prob, param)
            #svm_save_model('linear_scale.model', m)
            #m = svm_load_model('linear_scale.model')
            #p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
            #ACC, MSE, SCC = evaluations(y, p_label)
            print '\n\n\n', a, C,
            print 'CV_ACC3', m,
            #print p_label
            CV_ACC[(gamma, C)] = m
    return CV_ACC
#cwd=os.getcwd()

x_train=np.genfromtxt('./data/phishing-train-features.txt',delimiter='\t')
y_train=np.genfromtxt('./data/phishing-train-label.txt', delimiter='\t')
x_test=np.genfromtxt('./data/phishing-test-features.txt',delimiter='\t')
y_test=np.genfromtxt('./data/phishing-test-label.txt',delimiter='\t')
print x_train.shape ,y_train.shape, x_test.shape, y_test.shape

x_new_train=process_data(x_train)
x_new_test=process_data(x_test)
print x_new_train.shape,x_new_test.shape
start_time1=start_time()
CV1=linear_SVM()
time1=end_time(start_time1)

best_CV1 = sorted(CV1.iteritems(),key=operator.itemgetter(1),reverse=True)
print best_CV1[0],best_CV1[0][1]

start_time2=start_time()
CV2=poly_SVM()
time2=end_time(start_time2)

start_time3=start_time()
CV3=rbf_SVM()
time3=end_time(start_time3)

best_CV2 = sorted(CV2.iteritems(),key=operator.itemgetter(1),reverse=True)
best_CV3 = sorted(CV3.iteritems(),key=operator.itemgetter(1),reverse=True)
print '******************'
print 'CV1',CV1
print 'CV2',CV2
print 'CV3',CV3

print '******************'
best=0
hyperparam=[]
kernel_type=None
if best_CV1[0][1]>best:
    best=best_CV1[0][1]
    hyperparam=best_CV1[0][0]
    kernel_type='Linear'
if best_CV2[0][1]>best:
    best=best_CV2[0][1]
    hyperparam=best_CV2[0][0]
    kernel_type='Poly'
if best_CV3[0][1]>best:
    best=best_CV3[0][1]
    hyperparam=best_CV3[0][0]
    kernel_type='RBF'

print best,kernel_type,hyperparam
print best_CV1[0],best_CV1[0][1]
print best_CV2[0],best_CV2[0][1]
print best_CV3[0],best_CV3[0][1]

start_time1=start_time()
if kernel_type=='RBF':
    gamma=hyperparam[0]
    C=hyperparam[1]
    predict_RBF(gamma,C)
if kernel_type=='Poly':
    degree=hyperparam[0]
    C=hyperparam[1]
    predict_poly(degree,C)
end_time(start_time1)

print time1,time2,time3