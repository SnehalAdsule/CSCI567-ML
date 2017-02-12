import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from sympy.simplify.simplify import sum_add
import datetime
from scipy.stats import multivariate_normal

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def plot_scatter(data,cluster):
    colors = ['red', 'green', 'blue','orange']
    plt.scatter(data[:, 0], data[:, 1], c=cluster,alpha=0.5,s=100)
    plt.legend()
    plt.grid(True)
    plt.show()

def not_converged(mu,n_mu):
    print len(mu),len(mu[0])
    for i in range(len(mu)):
        for j in range(len(mu[i])):
            if mu[i][j]==n_mu[i][j]:
               # print mu[i][j], n_mu[i][j],'\t',
                continue
            else:
                return True
        #print '\n'
    return False

def no_change(old,new):
    if len(old)==0:
        return True
    for i in range(len(old)):
        if new[i] == old[i]:
            continue
        else:
            print len(old),new[i] , old[i],'\t',
            return True
    return False

def k_means(data,k):
    means=(random.sample(data, k))
    new_means=(random.sample(data, k))
    #print 'start',means,new_means
    old_cluster = []
    cluster = []
    while no_change(old_cluster,cluster):
        means=new_means
        old_cluster=cluster
        clusters={}
        cluster = []
        for i in range(data.shape[0]):
            dist=[]
            for j in range(k):
                dist.append((euclideanDistance(data[i],means[j],len(data[i])),j))
            dist.sort()
            if dist[0][1] in clusters.keys():
                clusters[dist[0][1]].append(data[i])
            else:
                clusters[dist[0][1]] = [data[i]]
            cluster.append(dist[0][1])
        new_means=[np.mean(clusters[j],axis=0) for j in range(k)]
        #print len(clusters[0]), len(clusters[1])
        #print "old means",means,"new_means",new_means
    #print "old ",np.sum(cluster),"new_",np.sum(old_cluster)
    #plot_scatter(data, cluster)
    return cluster,new_means

def rbf_kernel_means(data,k):
    means=[]
    new_means=(random.sample(data, k))
    #print 'start',means,new_means
    old_cluster = []
    cluster = []
    nk=1
    while no_change(old_cluster,cluster):
        means=new_means
        old_cluster=cluster
        clusters={}
        #cluster = []
        dist=[]
        print len(cluster)
        for ki in range(len(data)):
            for j in range(k):
                #print 'len clusters', len(clusters)
                if len(cluster) == 0:
                    cluster=np.random.randint(2, size=len(data))
                    #cluster=init_cluster.in_cluster()
                if j==0:
                    nk = len(cluster) - np.sum(cluster)
                else:
                    nk= np.sum(cluster)

                Kii = data[ki][ki]
                sum_Kij=0
                for kj in range(len(data)):
                    i_cluster=[1 if cluster[kj] ==j else 0]
                    #print 'ukj',i_cluster
                    sum_Kij+=i_cluster[0] *data[kj][ki]
                Kij = sum_Kij
                sum_Kjk = 0
                for kj in range(len(data)):
                    for kk in range(len(data)):
                        i_cluster_kj = [1 if cluster[kj] == j else 0]
                        i_cluster_kk = [1 if cluster[kk] == j else 0]
                        sum_Kjk += i_cluster_kj[0]*i_cluster_kk[0]* data[kj][kk]
                Kjk = sum_Kjk

                #print j,'Kii',Kii,Kij,Kjk,nk
                distances = Kii - 2*((Kij)/nk) + ( Kjk / math.pow(nk , 2))
                #print ki,j, 'Kii', Kii, Kij, Kjk, nk,distances ,'\t',
                dist.append((distances,j))
            dist.sort()
            if dist[0][1] in clusters.keys():
                clusters[dist[0][1]].append(data[ki])
            else:
                clusters[dist[0][1]] = [data[ki]]
            cluster[ki]=(dist[0][1])
            #print 'cluster', np.sum(cluster), dist[0]
        #new_means=[np.mean(clusters[j],axis=0) for j in clusters]

        #print "old means",means,"new_means",new_means
    #print "old ",np.sum(cluster),"new_",np.sum(old_cluster)
    return cluster

def kernel_means(data,k):
    means=[]
    new_means=(random.sample(data, k))
    #print 'start',means,new_means
    old_cluster = []
    cluster = []
    while no_change(old_cluster,cluster):
        means=new_means
        old_cluster=cluster
        clusters={}
        old_clusters={}
        cluster = np.zeros(len(data))
        print data
        for ki in range(len(data)):
            dist = []
            for j in range(k):
                nk=np.sum([len(clusters[j]) if j in clusters.keys() else 1])
                print 'len clusters', len(clusters),nk
                Kjj=len(clusters[j])*len(clusters[j])
                #Kii=data.diagonal()
                Kii=data[ki][ki]
                Kij=np.sum(data[ki][:])
                Kjk=np.sum(np.sum(data))
                nk2 = math.pow(nk, 2)
                print ki,j,Kjj,Kii,Kij,Kjk,nk2

                distances = (Kii - ((2 * Kij)/nk) + ( Kjj*Kjk / (nk2)))
                print distances
                dist.append((distances,j))
            dist.sort()
            print 'dist',dist
            if dist[0][1] in clusters.keys():
                clusters[dist[0][1]].append(data[ki])
            else:
                clusters[dist[0][1]] = [data[ki]]
            cluster[ki]=(dist[0][1])
            print len(cluster),'== ',len(data)
        new_means=[np.mean(clusters[j],axis=0) for j in clusters.keys()]
        print "old means",means,"new_means",new_means
    return cluster



    #recaluclate the means

def exec_k_means():
    k_values = [2, 3, 5]
    # k_values=[2]
    for j in k_values:
        cluster,new_means=k_means(blob, j)
        plot_scatter(blob, cluster)
    for j in k_values:
        cluster,new_means=k_means(circle, j)
        plot_scatter(circle, cluster)

def kernalize_input(data):
    feature_space=[]
    for i in range(data.shape[0]):
        feature_space.append([])
        for j in range(data.shape[0]):
            math.exp(-math.pow((euclideanDistance(data[i],data[j],len(data[j]))),2)/2)
    feature_space=np.array(feature_space)
    print feature_space.shape
    return feature_space

def rbf(data,sigma,gamma):
    feature_space=[]
    for i in range(data.shape[0]):
        feature_space.append([])
        for j in range(data.shape[0]):
            sq_eucliden=math.pow((euclideanDistance(data[i],data[j],len(data[j]))),2)
            sq_sigma=(2*sigma*sigma)
            #feature_space[i].append(math.exp(-sq_eucliden/sq_sigma))
            feature_space[i].append(math.exp(-sq_eucliden*gamma))
    feature_space=np.array(feature_space)
    print feature_space.shape
    return feature_space

def poly(data):
    feature_space = []
    for i in range(len(data)):
        feature_space.append([])
        for j in range(len(data)):
            c = 1
            for kj in range(2):
                c += math.pow(data[i][kj], 2) + math.pow(data[j][kj], 2)  + math.pow((math.pow(data[i][kj], 2) * math.pow(data[j][kj], 2)),2)
                #c +=  2*math.pow(data[i][kj], 2) * math.pow(data[j][kj], 2)  + math.pow((math.pow(data[i][kj], 4) + math.pow(data[j][kj], 4)),2)
                #c = 1+ math.pow(data[i][kj], 2)*math.pow(data[j][kj], 2)  + (2*data[i][kj]*data[j][kj])
                # c+=data[i][kj]*data[j][kj] * data[i][kj]*data[j][kj]

            feature_space[i].append(c)
    feature_space = np.array(feature_space)
    print feature_space.shape
    return feature_space

def test():
    cluster = np.random.randint(2, size=500)
    for jj in range(0,2):
        if jj == 0:
            nk = 500 - np.sum(cluster)
        else:
            nk = np.sum(cluster)
        print jj,nk

def poly_kernel(data):
    feature_space=1+data.dot(data.T)*data.dot(data.T)
    print feature_space.shape
    return feature_space


def gaussian(data, mean, cov):
    norm = multivariate_normal.pdf(data, mean, cov)
    return norm


def EM():
    LL_iter_converged = {}
    clusters_assigned = {}
    most_likely_mus = {}
    most_likely_vars = {}
    LL_best = {}
    for run in range(0, 5):
        if (run == 5):
            cluster, mu = k_means(blob, 3)
            prior = np.zeros(3)
            for ki in range(len(cluster)):
                for t in range(0, 3):
                    if cluster[ki] == t:
                        prior[t] += 1.0
            prior = prior / len(cluster)
        else:
            cluster = np.random.randint(3, size=len(blob))
            r = []
            for i in range(3):
                r.append(random.randint(1, 3))
            print r, np.sum(r)
            sum = np.sum(r)
            prior = np.array(r) / float(sum)
        print prior, np.sum(prior)
        if (run == 5):
            cluster, mu = k_means(blob, 3)
        else:
            mu = random.sample(blob, 3)

        mu = []
        variances = []
        for j in range(3):
            clusters = []
            for i in range(len(blob)):
                if (cluster[i] == j):
                    clusters.append(blob[i])
            clusters = np.array(clusters)
            var = np.cov(clusters.T)
            mean = np.mean(clusters, axis=0)
            # print clusters.shape
            variances.append(var)
            mu.append(mean)
            # print mu[j],variances[j],'\n'
        newmu = mu
        mu = []
        old_cluster = []
        LL_iter = []
        iter = 0
        loglik = -999999
        old_loglik = -99999999
        while (loglik != old_loglik):
            iter = iter + 1
            old_loglik = loglik
            old_cluster = cluster
            mu = newmu
            p = []
            for ki in range(len(blob)):
                p.append([])
                for j in range(3):
                    # (x_n|z_n=k)
                    p[ki].append(gaussian(blob[ki], mu[j], variances[j]))
                    # print p[ki][j]
            responsibility = []
            sum_pdf = 0
            n = len(blob)
            k = 3
            responsibility = np.zeros((k, n))
            for ki in range(len(blob)):
                for j in range(3):
                    responsibility[j, ki] = prior[j] * p[ki][j]
            responsibility = responsibility / np.sum(responsibility, axis=0)
            # print sum_pdf
            # print responsibility
            m_c = np.sum(responsibility, axis=1)
            mu = []
            sigma = []
            for i in range(k):
                mu_i = np.dot(responsibility[i, :], blob) / m_c[i]
                sigma_i = np.zeros((2, 2))
                for j in range(n):
                    sigma_i += responsibility[i, j] * np.outer(blob[j, :] - mu_i, blob[j, :] - mu_i)
                sigma_i = sigma_i / m_c[i]
                prior[i] = m_c[i] / np.sum(m_c)  # normalize the new priors
                mu.append(mu_i)
                sigma.append(sigma_i)
                #print '\t mu_sigma', mu,
                #print '\t', sigma,
            print '\t', m_c, prior

            sum_log_prob = 0
            cluster = np.zeros(n)
            for ki in range(len(blob)):
                sum_prob = 0
                max_idx = 0
                max_val = -999999
                for j in range(3):
                    sum_prob += prior[j] * p[ki][j]
                    if p[ki][max_idx] >= p[ki][j]:
                        max_idx = j
                        # max_val=p[ki][j]
                sum_log_prob += math.log(sum_prob)
                cluster[ki] = max_idx

            loglik = sum_log_prob
            #if (iter > 250): break
            LL_iter.append(loglik)
            print '(', run, iter, ')', loglik, old_loglik, len(LL_iter)
        LL_iter_converged[run] = LL_iter
        LL_best[run] = loglik
        most_likely_mus[run] = mu
        most_likely_vars[run] = sigma
        clusters_assigned[run] = cluster
    colors = []
    for run in range(5):
        x = np.arange(len(LL_iter_converged[run]))
        # print x,LL_iter_converged[run]
        plt.plot(x, LL_iter_converged[run])
    plt.legend(['1', '2', '3', '4', '5'], loc='upper right')
    plt.show()

    best_idx = 0
    for run in range(5):
        if LL_best[run] >= LL_best[best_idx]:
            best_idx = run
            print best_idx
    plot_scatter(blob, clusters_assigned[best_idx])
    print most_likely_mus[best_idx]
    print most_likely_vars[best_idx]

print 'Loading Data'
blob = np.loadtxt('hw5_blob(1).csv',delimiter=',')
circle = np.loadtxt('hw5_circle(1).csv',delimiter=',')
print blob.shape
print circle.shape
print 'K- means'
exec_k_means()
new_data = poly(circle)
print new_data.shape
k_values=[2]
#poly kernel matrix
print 'Polynomial kernel matrix'
print datetime.datetime.now().time()
for j in k_values:
    cluster,new_means=k_means(new_data, j)
    plot_scatter(circle, cluster)
print datetime.datetime.now().time()
new_data = rbf(circle,0.1,20)
k_values=[2]
print 'EM Algorithm'
EM()
print 'RBF kernel matrix'
#RBF kernel
print datetime.datetime.now().time()
for j in k_values:
    #cluster,new_means=k_means(new_data, j)
    cluster,new_means=rbf_kernel_means(new_data, j)
    plot_scatter(circle, cluster)
print datetime.datetime.now().time()
