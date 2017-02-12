import csv
import math

def prob_glass(dataset):
    prob={}
    for row in dataset:
        try:
            prob[str(row[-1])] = prob[str(row[-1])]+1
        except:
            prob[str(row[-1])]=1
    for x,y in prob.iteritems():
        prob[x]=y/float(len(dataset))
    return prob

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
    x1=0
    for i in range(len(arr1)):
        x1=x1+((arr1[i] - x_mean)*(arr2[i] - y_mean))
    covariance=(x1)/float(len(arr1)-1)
    #print '=>',x_mean,y_mean,sigma_x,sigma_y,covariance
    return float(covariance/(sigma_x*sigma_y))

def loadData(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def conditionalProbability(x, mean1, var):
    #if x==0:
     #   return 1
    if var==0:
        return 0
    exp = math.exp(-(math.pow(x-mean1,2)/(2*var)))
    return (1 / (math.sqrt(2*math.pi*var))) * exp

def nb():
    dataset=loadData("train.txt")
    attribute={}
    for i in range(len(dataset[0])):
        attribute[i]=[]
    for x in dataset:
        for i in range(len(x)):
            attribute[i].append(x[i])
            #print attribute
    mu_attr={}
    sigma2_attr={}
    corr_attr={}
    print 'Attribute wise Summary \n attr\tmean\t var\t correlation'
    for i in range(len(dataset[0])-2):
        mu_attr[i]=mean(attribute[i+1])
        sigma2_attr[i]=variance(attribute[i+1])
        corr_attr[i]=correlation(attribute[i+1], attribute[len(dataset[0])-1])
        print i,' \t',mu_attr[i] ,'\t', sigma2_attr[i],'\t', corr_attr[i]


    glass={} # k classes
    for k in range(len(dataset)):
        row=dataset[k]
        if (str(row[-1]) not in glass):
            glass[str(row[-1])] = []
        glass[str(row[-1])].append(row) # array of array

    mu= {}
    sigma2={}
    corr={}

   # print 'Glass Type \n (Class|Attr) \t mean\t var\t'
    for k in glass:
        className=glass[k]
        #print 'Glass = ',len(glass[k]),'\n', len(glass[k][0])
        mu[k]={}
        sigma2[k]={}
        #print 'j=',len(glass[k][0])
        for j in range(len(glass[k][0])-2):
            collect_col=[]
            for row in range(len(glass[k])):
                collect_col.append(glass[k][row][j+1])
            mu[k][j]=mean(collect_col)
            sigma2[k][j]=variance(collect_col)
          #  print '{(',k,':',j,'),mean=',mu[k][j] ,',var=', sigma2[k][j],'},\t',

    prob_glassType=prob_glass(dataset)
    #print 'probability of glass type \n',prob_glassType

    test_dataset=loadData('test.txt')
    # every sample in test
    accuracy=[]
    count=0
    for row in test_dataset:
        marginalProbabilities = {}
        #every glass type compute posterior
        for k,val in prob_glassType.iteritems():
            marginalProbabilities[k] = 1.0
            #print 'prob(',k,')=',val,len(mu[k])
            for j in range(len(mu[k])):
                # every attribute
                mean1, var = mu[k][j],sigma2[k][j]
                x = row[j+1]
                condProb=conditionalProbability(x, mean1, var)
                marginalProbabilities[k]*= condProb
                #print '\t',j, x, mean1, var,condProb,marginalProbabilities,prob_glassType[k]
            marginalProbabilities[k]=marginalProbabilities[k]*prob_glassType[k]
        #print marginalProbabilities

        # compare the posterior
        maxClass='Unknown'
        maxProb=-1
        for glassType in (marginalProbabilities):
            # arg max of P(glassType|row)
            postProb=marginalProbabilities[glassType]
            if maxClass == 'Unknown' or postProb > maxProb:
                maxProb = postProb
                maxClass = glassType
        if(maxClass==str(row[-1])):
            accuracy.append(1)
        else:
            accuracy.append(0)
        count=count+1
        #print row[0],row[-1],maxClass#'\n\t',marginalProbabilities

    correct=0
    #print accuracy
    for i in range(len(accuracy)):
        if accuracy[i]== 1:
            correct=correct+1
            #print correct
    print '\n\n Naive Bayes : Test Accuracy :', (correct/float(len(accuracy)))*100

    train_dataset=loadData('train.txt')
    # every sample in test

    accuracy=[]
    count=0

    for row in train_dataset:

        marginalProbabilities = {}
        #every glass type compute posterior
        for k,val in prob_glassType.iteritems():
            marginalProbabilities[k] = 1.0
            #print 'prob(',k,')=',val,len(mu[k])
            for j in range(len(mu[k])):
                # every attribute
                mean1, var = mu[k][j],sigma2[k][j]
                x = row[j+1]
                condProb=conditionalProbability(x, mean1, var)
                marginalProbabilities[k]*= condProb
                #print '\t',j, x, mean1, var,condProb,marginalProbabilities,prob_glassType[k]
            marginalProbabilities[k]=marginalProbabilities[k]*prob_glassType[k]
        #print marginalProbabilities

        # compare the posterior
        maxClass='Unknown'
        maxProb=-1
        for glassType in (marginalProbabilities):
            # arg max of P(glassType|row)
            postProb=marginalProbabilities[glassType]
            if maxClass == 'Unknown' or postProb > maxProb:
                maxProb = postProb
                maxClass = glassType
        if(maxClass==str(row[-1])):
            accuracy.append(1)
        else:
            accuracy.append(0)
        count=count+1
        #print row[0],row[-1],maxClass#'\n\t',marginalProbabilities

    correct=0
    #print accuracy
    for i in range(len(accuracy)):
        if accuracy[i]== 1:
            correct=correct+1
            #print correct
    print '\n Naive Bayes : Train Accuracy',(correct/float(len(accuracy)))*100
