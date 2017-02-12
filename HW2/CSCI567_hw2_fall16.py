import pandas as pd
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
import sys
import read_data

read_data.main()
theta=None
features=None
column_names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

def mean(array):
    return float(sum(array)/len(array))

def stdev(numbers):
	meanVal = mean(numbers)
	variance = sum([pow(x-meanVal,2) for x in numbers])/float(len(numbers)-1)
	return float(math.sqrt(variance))

def variance(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return float(variance)
def correlation(arr1,arr2):
    x_mean = mean(arr1)
    y_mean= mean(arr2)
    sigma_x=math.sqrt(variance(arr1))
    sigma_y=math.sqrt(variance(arr2))
    x1_sum=0
    for i in range(len(arr1)):
        x1_sum=x1_sum+((arr1[i] - x_mean)*(arr2[i] - y_mean))
    covariance=(x1_sum)/float(len(arr1)-1)
    #print '=>',x_mean,y_mean,sigma_x,sigma_y,covariance
    return float(covariance/(sigma_x*sigma_y))

def plot_histograms(x):
    for i in range(0, 13, 1):
        a = x[i].tolist()
        plt.hist(x[i], bins=10)
        plt.title(column_names[i])
        plt.plot()
        plt.show()

def theta_calculation(x,y):
    X = x
    xTx = X.T.dot(X)  # 14x433 dot 433x14

    #iXtX = np.linalg.inv(xTx)
    iXtX_xT = np.linalg.solve(xTx,X.T)
    #print xTx.shape, X.T.shape, x.shape, iXtX_xT.shape, y.shape
    theta = iXtX_xT.dot(y)
    return theta


def theta_calculation2(x,y):
    X = x
    xTx = X.T.dot(X)  # 14x433 dot 433x14
    print xTx[0][0], X.T[0][0], x[0][0],y[0][0]
    iXtX = np.linalg.inv(xTx)
    iXtX_xT = iXtX.dot(X.T)
    print iXtX_xT[0][0]
    print xTx.shape, X.T.shape, x.shape, iXtX_xT.shape, y.shape
    theta = iXtX_xT.dot(y)
    return theta

def cv_fold_data(x,filename):
    temp_test, temp_train = {}, {}
    for i in range(0, 10, 1):
        # temp_test[i]=[]
        temp_train[i] = []
    for j in range(len(x)):
        temp_train[j % 10].append(x.loc[j])
    for i in range(0, 10, 1):
        np.savetxt(filename + str(i) + '.csv', temp_train[i], delimiter=',')

def standardization(x):
    x_mean = x.mean()
    x_stdev = x.std()
    x_z = (x - x_mean) / x_stdev
    return x_z

def ridge_regression_train():
    IdentitySize = x.shape[1]
    IdentityMatrix = np.zeros((IdentitySize, IdentitySize))
    np.fill_diagonal(IdentityMatrix, 1)
    lamb = 0.0001
    for i in range(0, 5):
        print 'lambda=', lamb
        XtX_lamb = x.T.dot(x) + lamb * IdentityMatrix
        XtY = x.T.dot(y)
        # i_XtX_lamb=np.linalg.inv(XtX_lamb)
        # theta2=i_XtX_lamb.dot(XtY)
        theta2 = np.linalg.solve(XtX_lamb, XtY)
        y_pred = x.dot(theta2)
        y_pred = (y_pred * y_stdev) + y_mean
        e = (y_pred - train_target)
        se = e ** 2
        MSE_ridge_train = se.sum() / se.shape[0]
        print '\tlambda ', lamb, 'MSE_train ', MSE_ridge_train.values[0]
        y1_pred = x1.dot(theta2)
        y1_pred = (y1_pred * y1_stdev) + y1_mean
        e = (y1_pred - test_target)
        se = e ** 2
        MSE_ridge_test = se.sum() / se.shape[0]
        print '\tlambda ', lamb, 'MSE_test ', MSE_ridge_test.values[0]
        lamb = lamb * (10)


def linear_regression_train():
    x.insert(0, 'x0', 1)
    theta = theta_calculation(x, y)
    theta = theta_calculation(x, y)
    y_pred = x.dot(theta)
    y_pred = (y_pred * y_stdev) + y_mean
    e = (y_pred - train_target)
    se = e ** 2
    MSE_train = se.sum() / se.shape[0]
    print 'Linear MSE_train', MSE_train.values[0]
    return theta

def linear_regression_test():
    x1.insert(0, 'x0', 1)
    y_pred_test = x1.dot(theta)
    y_pred_test = (y_pred_test * y1_stdev) + y1_mean
    e = y_pred_test - test_target
    se = e ** 2
    MSE_test = se.sum() / se.shape[0]
    print ' Linear MSE_test', MSE_test.values[0]


def CV_ridge_regression():
    cv_fold_data(trainSet, 'test_')
    cv_fold_data(train_target, 'target_')
    lambda_cv_mse_train={}
    lambda_cv_mse_test = {}
    for i in range(0, 10, 1):
        cv_test = pd.read_csv('test_' + str(i) + '.csv', header=None)
        cv_test_target = pd.read_csv('target_' + str(i) + '.csv', header=None)
        cv_train_frames = []
        cv_train_target_frames = []
        for j in range(0, 10, 1):
            if i != j:
                cv_train_temp = pd.read_csv('test_' + str(j) + '.csv', header=None)
                cv_train_frames.append(cv_train_temp)
                cv_train_target_temp = pd.read_csv('target_' + str(j) + '.csv', header=None)
                cv_train_target_frames.append(cv_train_target_temp)
        cv_train = pd.concat(cv_train_frames)
        cv_train_target = pd.concat(cv_train_target_frames)
        # print i,len(cv_train),len(cv_test)
        cv_x_mean = cv_train.mean()
        cv_y_mean = cv_train_target.mean()
        cv_x_stdev = cv_train.std()
        cv_y_stdev = cv_train_target.std()
        cv_x = (cv_train - cv_x_mean) / cv_x_stdev
        cv_y = (cv_train_target - cv_y_mean) / cv_y_stdev

        cv_x1_mean = cv_test.mean()
        cv_y1_mean = cv_test_target.mean()
        cv_x1_stdev = cv_test.std()
        cv_y1_stdev = cv_test_target.std()
        cv_x1 = (cv_test - cv_x1_mean) / cv_x1_stdev
        cv_y1 = (cv_test_target - cv_y1_mean) / cv_y1_stdev

        cv_x.insert(0, 'x0', 1)
        cv_x1.insert(0, 'x0', 1)
        print '\n\n',i,' => ',cv_x.shape,cv_y.shape,cv_x1.shape,cv_y1.shape
        IdentitySize = cv_x.shape[1]
        IdentityMatrix = np.zeros((IdentitySize, IdentitySize))
        np.fill_diagonal(IdentityMatrix, 1)
        lamb = 0.0001
        min_lambda = None

        for k in range(0, 10000):
            # print 'lambda=', lamb
            cv_XtX_lamb = cv_x.T.dot(cv_x) + lamb * IdentityMatrix
            cv_XtY = cv_x.T.dot(cv_y)
            cv_theta = np.linalg.solve(cv_XtX_lamb, cv_XtY)
            cv_y_pred = cv_x.dot(cv_theta)
            cv_y_pred = (cv_y_pred * cv_y_stdev) + cv_y_mean
            e = (cv_y_pred - cv_train_target)
            se = e ** 2
            cv_MSE_ridge_train = se.sum() / se.shape[0]
            cv_y1_pred = cv_x1.dot(cv_theta)
            cv_y1_pred = (cv_y1_pred * cv_y1_stdev) + cv_y1_mean
            e = (cv_y1_pred - cv_test_target)
            se = e ** 2
            cv_MSE_ridge_test = se.sum() / se.shape[0]
            print 'Run', i, '\t lambda ', lamb, 'CV MSE_train ', cv_MSE_ridge_train.values[0], 'CV MSE_test ', cv_MSE_ridge_test.values
            if lamb in lambda_cv_mse_train:
                lambda_cv_mse_train[lamb]= lambda_cv_mse_train[lamb] + cv_MSE_ridge_train.values[0]
                lambda_cv_mse_test[lamb] = lambda_cv_mse_test[lamb] + cv_MSE_ridge_test.values[0]
            else:
                lambda_cv_mse_train[lamb]=cv_MSE_ridge_train.values[0]
                lambda_cv_mse_test[lamb] =  cv_MSE_ridge_test.values[0]
            if lamb == 0:
                break
            if lamb==10:
                lamb=0
            lamb = (lamb) + lamb*(0.5)
            if lamb>10:
                lamb=10


    for k in lambda_cv_mse_train.keys():
        lambda_cv_mse_train[k]=lambda_cv_mse_train[k]/10
        lambda_cv_mse_test[k] = lambda_cv_mse_test[k]/10

    min_lambda_train=sorted(lambda_cv_mse_train.iteritems(), key=operator.itemgetter(1))
    min_lambda_test = sorted(lambda_cv_mse_test.iteritems(), key=operator.itemgetter(1))
    print 'Average lamdba',min_lambda_train
    lamb=min_lambda_train[0][0]
    # on test set
    XtX_lamb = x.T.dot(x) + lamb * IdentityMatrix
    XtY = x.T.dot(y)
    theta3 = np.linalg.solve(XtX_lamb, XtY)
    y_pred = x.dot(theta3)
    y_pred = (y_pred * y_stdev) + y_mean
    e = (y_pred - train_target)
    se = e ** 2
    MSE_ridge_train = se.sum() / se.shape[0]
    print '\t best lambda_train lambda=', lamb, 'MSE_train ', MSE_ridge_train.values[0]

    # on test set
    XtX_lamb = x1.T.dot(x1) + lamb * IdentityMatrix
    XtY = x1.T.dot(y1)
    y1_pred = x1.dot(theta3)
    y1_pred = (y1_pred * y1_stdev) + y1_mean
    e = (y1_pred - test_target)
    se = e ** 2
    MSE_ridge_test = se.sum() / se.shape[0]
    print '\t lambda_test lambda=', lamb, 'MSE_test ', MSE_ridge_test.values[0]

def top4_features():
    b = y[0].tolist()
    pearson = {}
    abs_pearson = {}
    for i in range(0, 13, 1):
        a = x[i].tolist()
        pearson[i] = correlation(a, b)
        abs_pearson[i] = abs(correlation(a, b))

    features = sorted(abs_pearson.iteritems(), key=operator.itemgetter(1), reverse=True)
    selected_trainSet = None
    selected_testSet = None
    for i in range(0, 4, 1):
        k = features[i][0]

        if i == 0:
            selected_trainSet = x[k]
            selected_testSet = x1[k]
        else:
            selected_trainSet = pd.concat([selected_trainSet, x[k]], axis=1)
            selected_testSet = pd.concat([selected_testSet, x1[k]], axis=1)
        #print selected_trainSet.shape, selected_testSet.shape
    x3 = selected_trainSet
    x3_1 = selected_testSet
    selected_trainSet.insert(0, 'x0', 1)
    selected_testSet.insert(0, 'x0', 1)
    theta = theta_calculation(x3, y)
    y_pred = x3.dot(theta)
    y_pred = (y_pred * y_stdev) + y_mean
    e = (y_pred - train_target)
    se = e ** 2
    MSE_train = se.sum() / se.shape[0]
    feature_list=[]
    for col  in x3.columns:
        if col!='x0':
            col=int(col)
            feature_list.append(column_names[col])
    print 'top4 features MSE_train', MSE_train.values[0] ,feature_list
    y_pred_test = x3_1.dot(theta)
    y_pred_test = (y_pred_test * y1_stdev) + y1_mean
    e = y_pred_test - test_target
    se = e ** 2
    MSE_test = se.sum() / se.shape[0]
    print 'top4 features MSE_test', MSE_test.values[0]

def residue_feature():
    b = y[0].tolist()
    pearson = {}
    abs_pearson = {}
    for i in range(0, 13, 1):
        a = x[i].tolist()
        pearson[i] = correlation(a, b)
        abs_pearson[i] = abs(correlation(a, b))
    features = sorted(abs_pearson.iteritems(), key=operator.itemgetter(1), reverse=True)
    residue_trainSet = None
    residue_testSet = None
    temp_x = x.copy()
    list = []
    for i in range(0, 4, 1):
        k = features[i][0]
        list.append(k)
        if i == 0:
            residue_trainSet = temp_x[k]
            residue_testSet = x1[k]
        else:
            residue_trainSet = pd.concat([residue_trainSet, temp_x[k]], axis=1)
            residue_testSet = pd.concat([residue_testSet, x1[k]], axis=1)
        print 'feature selected : ', k, residue_testSet.shape, residue_trainSet.shape, temp_x.shape
        # del temp_x[k]
        x4 = pd.DataFrame(residue_trainSet)
        x4_1 = pd.DataFrame(residue_testSet)
        if 'x0' not in x4.columns:
            x4.insert(0, 'x0', 1)
            x4_1.insert(0, 'x0', 1)
        #print x4.columns
        theta = theta_calculation(x4, y)
        y_pred = x4.dot(theta)
        y_pred = (y_pred * y_stdev) + y_mean
        e = (y_pred - train_target)
        se = e ** 2
        MSE_train = se.sum() / se.shape[0]
        print 'residue features MSE_train', MSE_train.values[0]
        y_pred_test = x4_1.dot(theta)
        y_pred_test = (y_pred_test * y1_stdev) + y1_mean
        e1 = y_pred_test - test_target
        se1 = e1 ** 2
        MSE_test = se1.sum() / se1.shape[0]
        print 'residue features MSE_test', MSE_test.values[0]
        b = e[0].tolist()
        pearson1 = {}
        abs_pearson1 = {}
        for j in range(0, 13, 1):
            a = temp_x[j].tolist()
            if (j in list):
                pearson1[j] = 0
            else:
                pearson1[j] = correlation(a, b)
                abs_pearson1[j] = abs(correlation(a, b))
        features = sorted(abs_pearson1.iteritems(), key=operator.itemgetter(1), reverse=True)

def brute_force_feature():
    min_trainig_set=sys.maxint
    min_test_set = sys.maxint
    min_features={}
    for i in range(0,13,1):
        for j in range(i+1 ,13, 1):
            for p in range(j+1, 13, 1):
                for q in range(p+1, 13, 1):
                    b_trainSet = pd.concat([trainSet[i],trainSet[j],trainSet[p],trainSet[q]], axis=1)
                    b_testSet = pd.concat([testSet[i],testSet[j],testSet[p],testSet[q]], axis=1)
                    x4 = pd.DataFrame(b_trainSet)
                    x4_1 = pd.DataFrame(b_testSet)
                    if 'x0' not in x4.columns:
                        x4.insert(0, 'x0', 1)
                        x4_1.insert(0, 'x0', 1)
                    b_theta = theta_calculation(x4, y)
                    y_pred = x4.dot(b_theta)
                    y_pred = (y_pred * y_stdev) + y_mean
                    e = (y_pred - train_target)
                    se = e ** 2
                    MSE_train = se.sum() / se.shape[0]
                    y_pred_test = x4_1.dot(b_theta)
                    y_pred_test = (y_pred_test * y1_stdev) + y1_mean
                    e1 = y_pred_test - test_target
                    se1 = e1 ** 2
                    MSE_test = se1.sum() / se1.shape[0]
                    print 'brute force features ',x4.columns,'\t MSE_train', MSE_train.values[0],'MSE_test ',  MSE_test.values[0]
                    if min_trainig_set >MSE_train.values[0]:
                        min_trainig_set = MSE_train.values[0]
                        min_features =x4.columns
                        min_test_set=MSE_test.values[0]
    print 'Best Brute Force, MSE_test',min_test_set,' MSE_train',min_trainig_set , 'For features',  min_features
def polynomial_feature_expansion():
    p_trainSet = trainSet.copy()
    p_testSet = testSet.copy()
    count = len(trainSet.columns)

    for i in range(0, 13, 1):
        for j in range(i, 13, 1):
            p_trainSet = pd.concat([p_trainSet, trainSet[i] * trainSet[j]], axis=1)
            #p_trainSet.insert( 0, trainSet[i] * trainSet[j]], 1)
            count += 1
            #print p_trainSet.shape

    for i in range(0, 13, 1):
        for j in range(i, 13, 1):
            p_testSet = pd.concat([p_testSet, testSet[i] * testSet[j]], axis=1)
            #print p_testSet.shape

    #np.savetxt('poly_test.csv', p_testSet, delimiter=',')
    px_mean = p_trainSet.mean()
    py_mean = train_target.mean()
    px_stdev = p_trainSet.std()
    py_stdev = train_target.std()
    px = (p_trainSet - px_mean) / px_stdev
    py = (train_target - y_mean) / y_stdev

    px1_mean = p_testSet.mean()
    py1_mean = test_target.mean()
    px1_stdev = p_testSet.std()
    py1_stdev = test_target.std()
    px1 = (p_testSet - px1_mean) / px1_stdev
    py1 = (test_target - py1_mean) / py1_stdev

    px.insert(0, 'x0', 1)
    px1.insert(0, 'x0', 1)
    print count, px.shape, px1.shape
    p_theta = theta_calculation(px, py)
    py_pred = px.dot(p_theta)
    y_pred = (py_pred * py_stdev) + py_mean
    e = (y_pred - train_target)
    se = e ** 2
    p_MSE_train = se.sum() / se.shape[0]
    print 'MSE_train', p_MSE_train.values[0]

    py1_pred_test = px1.dot(p_theta)
    y_pred_test = (py1_pred_test * y1_stdev) + y1_mean
    e = y_pred_test - test_target
    se = e ** 2
    p_MSE_test = se.sum() / se.shape[0]
    print 'MSE_test', p_MSE_test.values[0]


trainSet = pd.read_csv('train.csv', header=None)
testSet = pd.read_csv('test.csv', header=None)
train_target = pd.read_csv('train_target.csv', header=None)
test_target = pd.read_csv('test_target.csv', header=None)

x_mean = trainSet.mean()
y_mean = train_target.mean()
x_stdev = trainSet.std()
y_stdev = train_target.std()
x = (trainSet - x_mean) / x_stdev
y = (train_target - y_mean) / y_stdev

x1_mean = testSet.mean()
y1_mean = test_target.mean()
x1_stdev = testSet.std()
y1_stdev = test_target.std()
x1 = (testSet - x1_mean) / x1_stdev
y1 = (test_target - y1_mean) / y1_stdev

b = y[0].tolist()
pearson = {}
abs_pearson = {}
for i in range(0, 13, 1):
    a = x[i].tolist()
    pearson[i] = correlation(a, b)
    abs_pearson[i] = abs(correlation(a, b))
print 'Pearson coefficient : \n',pearson

print '\nPlotting histograms :'
plot_histograms(trainSet)
print '\nLinear Regression'
theta=linear_regression_train()
linear_regression_test()
print '\nRidge Regression'
ridge_regression_train()
print '\nRidge Regression with Cross Validation'
CV_ridge_regression()
print '\nFeature Selection Top Pearson Coefficient based'
top4_features()
print '\nFeature Selection Residue Iterative based Pearson'
residue_feature()
print '\nFeature Selection Brute Force'
brute_force_feature()
print '\nPolynomial Expansion'
polynomial_feature_expansion()


