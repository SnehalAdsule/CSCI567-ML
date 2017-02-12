import math
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

def sample_error():
    error = np.random.normal(0, 0.1, 10)
    print np.mean(error)

def plot_histograms(x,k,mean):
        plt.hist(x,bins=10)
        plt.title('g'+str(k)+' mean('+str(mean)+')')
        plt.plot()
        plt.show()

def fx(x):
    return 2*x*x

def theta_calculation(x,y):
    X = x
    xTx = np.array(X.T.dot(X))
    iXtX=np.linalg.pinv(xTx)
    #iXtX_xT = np.linalg.solve(xTx,X.T)
    iXtX_xT = iXtX.dot(X.T)
    #print X.T.shape, x.shape,'=',xTx.shape, iXtX_xT.shape, y.shape
    theta = iXtX_xT.dot(y)
    return theta

def linear_regression(p,q):
    X1, Y1 = [], []
    err = []
    efx=[]
    for k in range(0,6):
        efx.append(np.zeros(q))
    avg_g_x=np.zeros(6)
    FX=[]
    GX=[[],[],[],[],[],[]]
    mse = []
    bias = []
    variance = []

    for i in range(0, p):
        x1 = []
        for j in range(0, q):
            x1.append(random.uniform(-1, 1))
        error = np.random.normal(0, 0.1, q)
        err.append(error)
        x = np.array(x1)
        x = np.sort(x)
        f_x = 2 * (x * x)
        y = f_x + error
        FX.append(f_x)
        #FX.append(y)
        X1.append(x)
        Y1.append(y)
        g_x = np.ones(len(x))
        MSE = []
        BIAS = []
        VAR = []
        for k in range(0, 6):
            GX[k].append([])
            if k < 1:
                theta = np.ones(len(x))
            else:
                b = np.power(x, k - 1)
                g_x = np.vstack((g_x, b))
                theta = theta_calculation(g_x.T, y)
            if k > 0:
                y_pred = g_x.T.dot(theta)
            else:
                y_pred = np.ones(len(x)).T
            GX[k][i].append(y_pred)
            mse_i = np.mean(np.power((y_pred - y), 2))  # se.sum() / se.shape[0]
            if MSE == None:
                MSE = mse_i
            else:
                MSE.append(mse_i)
            avg_g_x[k]=avg_g_x[k]+np.mean(y_pred)
            efx[k]= efx[k] + (y_pred)

        mse.append(MSE)
        bias.append(BIAS)
        variance.append(VAR)
    arr_mse = np.array(mse)

    for k in range(0, 6):
        efx[k]=efx[k]/p
    '''
    print 'arr mse',arr_mse.shape
    print len(GX),len(GX[0])
    print len(FX), len(FX[0])
    print len(efx),len(efx[0])
    '''
    plot_mse = arr_mse.T

    for k in range(0, 6):
        plot_histograms(plot_mse[k],k+1,np.mean(plot_mse[k]))
        FX=np.array(FX)
        GX=np.array(GX)
        '''
        bias2=np.zeros(q)
        var2=np.zeros(q)
        avg_g_x=(avg_g_x/p)
        for i in range(0, p):
            bias2=bias2+np.power((efx[k] - (FX[i]) ),2)
            var2=var2 + np.power((GX[k][i]-efx[k]),2)
        bias2=np.mean(bias2/p)
        var2=np.mean(var2/p)
        '''
        print 'g'+str(k+1),
        print 'mse',np.mean(plot_mse[k]),
        print 'bias',np.mean(np.power((efx[k] - (FX )),2)),
        print 'vr',np.mean(np.power((GX[k]-efx[k]),2))

def ridge_theta_calculation(x,y,lamb):
    IdentitySize = x.shape[1]
    IdentityMatrix = np.zeros((IdentitySize, IdentitySize))
    np.fill_diagonal(IdentityMatrix, 1)
    XtX_lamb = x.T.dot(x) + lamb * IdentityMatrix
    XtY = x.T.dot(y)
    # i_XtX_lamb=np.linalg.inv(XtX_lamb)
    # theta2=i_XtX_lamb.dot(XtY)
    theta2 = np.linalg.solve(XtX_lamb, XtY)
    return theta2

def regularized_regression(p,q):
    lamb=0.0003
    mul_3=False
    for row in range(0,7):
        if (mul_3==False):
            lamb = (lamb/3) * 10
            mul_3=True
        else:
            lamb = (lamb * 3)
            mul_3 = False
        X1, Y1 = [], []
        err = []
        efx = []
        for k in range(0, 1):
            efx.append(np.zeros(q))
        avg_g_x=np.zeros(6)
        FX = []
        GX = [[]]
        mse = []
        bias = []
        variance = []

        for i in range(0, p):
            x1 = []
            for j in range(0, q):
                x1.append(random.uniform(-1, 1))
            error = np.random.normal(0, 0.1, q)
            err.append(error)
            x = np.array(x1)
            x = np.sort(x)
            f_x = 2 * (x * x)
            y = f_x + error
            FX.append(f_x)
            X1.append(x)
            Y1.append(y)
            g_x = np.ones(len(x))
            MSE = []
            BIAS = []
            VAR = []
            for k in range(3, 4):
                GX[k-3].append([])
                if k < 1:
                    theta = np.ones(len(x))
                else:
                    b = np.power(x, k - 1)
                    g_x = np.vstack((g_x, b))
                    theta = ridge_theta_calculation(g_x.T, y,lamb)
                if k > 0:
                    y_pred = g_x.T.dot(theta)
                else:
                    y_pred = np.ones(len(x)).T
                GX[k-3][i].append(y_pred)
                mse_i = np.mean(np.power((y_pred - f_x), 2))  # se.sum() / se.shape[0]
                if MSE == None:
                    MSE = mse_i
                else:
                    MSE.append(mse_i)
                avg_g_x[k-3]=avg_g_x[k-3]+np.mean(y_pred)
                efx[k-3] = efx[k-3] + (y_pred)
                '''
                #bias_i=np.mean(y_pred - f_x)
                bias_i = np.mean(np.power((efx - f_x) ,2))
                BIAS.append(bias_i)
                e_x=np.mean(np.power(y_pred ,2))
                e_x2=np.power(np.mean(y_pred),2)
                #variance_i= e_x - e_x2
                variance_i=np.mean(np.power(y_pred-e_fx ,2))
                VAR.append(variance_i)
                '''
            mse.append(MSE)
            bias.append(BIAS)
            variance.append(VAR)
        arr_mse = np.array(mse)
        for k in range(3, 4):
            efx[k-3] = efx[k-3] / p
        '''
        print 'arr mse', arr_mse.shape,
        print len(GX), len(GX[0]),
        print len(FX), len(FX[0]),
        print len(efx), len(efx[0])
        '''
        plot_mse = arr_mse.T

        for k in range(3, 4):
            #plot_histograms(plot_mse[k-3], k, np.mean(plot_mse[k-3]))
            bias2 = np.zeros(q)
            var2 = np.zeros(q)
            FX = np.array(FX)
            GX = np.array(GX)

            '''
            for i in range(0, p):
                bias2=bias2+np.power((efx[k] - (FX[i]) ),2)
                var2=var2 + np.power((GX[k][i]-efx[k]),2)
            bias2=np.mean(bias2/p)
            var2=np.mean(var2/p)
            '''

            print 'lambda',lamb,
            print 'mse', np.mean(plot_mse[k-3]),
            print 'bias', np.mean(np.power((efx[k-3] - (FX)), 2)),
            print 'vr', np.mean(np.power((GX[k-3] - efx[k-3]), 2))

print '************************1a********************'
linear_regression(100,10)
print '************************1b********************'
linear_regression(100,100)
print '************************1d********************'
regularized_regression(100,100)
#sample_error()


