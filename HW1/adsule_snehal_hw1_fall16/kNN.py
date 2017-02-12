import pandas as pd
import math
import operator

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def manhattanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += abs(instance1[x] - instance2[x])
	return distance

def knn_test(df_test_norm,df_train_norm,k):
	L1_accuracy=0
	L2_accuracy=0
	for i  in range(len(df_test_norm)):
		x1=df_test_norm.iloc[i,1:10]
		list_x1 = list(x1)
		#print i, list_x1
		dist1=[]
		dist2=[]
		k_neighbour_L1=[]
		k_neighbour_L2=[]
		for j  in range(len(df_train_norm)):
			y1= df_train_norm.iloc[j,1:10]
			list_y1 = list(y1)
			#print list_x1,list_y1
			temp1=manhattanDistance(list_x1,list_y1,len(list_y1))
			temp2=euclideanDistance(list_x1,list_y1,len(list_y1))
			list_y1.append(df_train_norm.iloc[j,0])
			list_y1.append(df_train_norm.iloc[j,10])
			dist1.append((list_y1,temp1))
			dist2.append((list_y1,temp2))

		dist1.sort(key=operator.itemgetter(1))
		dist2.sort(key=operator.itemgetter(1))
		#print dist1
		for x in range(k):
			k_neighbour_L1.append((dist1[x][0][-1],dist1[x][1]))
			k_neighbour_L2.append((dist2[x][0][-1],dist2[x][1]))
		#print k_neighbour_L1

		kClass = {}
		for x in range(len(k_neighbour_L1)):
			#print k_neighbour_L1[x][0],k_neighbour_L1[x][1]
			glass_type = k_neighbour_L1[x][0]
			if glass_type in kClass:
				kClass[glass_type] += 1
			else:
				kClass[glass_type] = 1

		sortedClass = sorted(kClass.iteritems(), key=operator.itemgetter(1), reverse=True)
		tie=[]
		for x in kClass:
			if kClass[x]==sortedClass[0][1]:
			  tie.append(x)
		ans=None
		if len(tie)>1:
			#print 'tie',tie,k_neighbour_L1
			for i in range(len(k_neighbour_L1)):
				for x in tie:
					#print i,x,k_neighbour_L1[i][0]
					if x==k_neighbour_L1[i][0]:
						#print 'answer',k_neighbour_L1[i]
						ans=i
						break
				if(ans!=None):
					break

			L1_predicted=k_neighbour_L1[ans][0]
		else:
			L1_predicted=sortedClass[0][0]
		kClass = {}
		for x in range(len(k_neighbour_L2)):
			glass_type = k_neighbour_L2[x][0]
			if glass_type in kClass:
				kClass[glass_type] += 1
			else:
				kClass[glass_type] = 1
		sortedClass = sorted(kClass.iteritems(), key=operator.itemgetter(1), reverse=True)
		tie=[]
		for x in kClass:
			if kClass[x]==sortedClass[0][1]:
			  tie.append(x)
		ans=None
		if len(tie)>1:
			#print 'tie',tie,k_neighbour_L2
			for i in range(len(k_neighbour_L2)):
				for x in tie:
					#print i,x,k_neighbour_L1[i][0]
					if x==k_neighbour_L2[i][0]:
						#print 'answer',k_neighbour_L2[i]
						ans=i
						break
				if(ans!=None):
					break

			L2_predicted=k_neighbour_L2[ans][0]
		else:
			L2_predicted=sortedClass[0][0]
		actual=df_test_norm.iloc[i,10]
		#print ' L1  ',L1_predicted ,' L2  ',L2_predicted ,'Actual',actual
		if(L1_predicted==actual):
			L1_accuracy=L1_accuracy+1
		if(L2_predicted==actual):
			L2_accuracy=L2_accuracy+1
	#accuracy
	total=len(df_test_norm)
	#print L1_accuracy,L2_accuracy,total
	print 'k=',k,'\t\tL1 :' ,(L1_accuracy/float(total))*100,'\t L2 :' ,(L2_accuracy/float(total))*100



def knn_LOO(df_test_norm):
	L1_accuracy=[]
	L2_accuracy=[]
	for i  in range(len(df_test_norm)):
		x1=df_test_norm.iloc[i,1:10]
		list_x1 = list(x1)
		dist1=[]
		dist2=[]
		k_neighbour_L1=[]
		k_neighbour_L2=[]
		rem_id=df_test_norm.iloc[i,0]
		df_train_norm=df_test_norm[df_test_norm[0]!=rem_id]
		#print len(df_train_norm),rem_id,df_train_norm

		for j  in range(len(df_train_norm)):
			y1= df_train_norm.iloc[j,1:10]
			list_y1 = list(y1)
			#print list_x1,list_y1
			temp1=manhattanDistance(list_x1,list_y1,len(list_y1))
			temp2=euclideanDistance(list_x1,list_y1,len(list_y1))
			list_y1.append(df_train_norm.iloc[j,0])
			list_y1.append(df_train_norm.iloc[j,10])
			dist1.append((list_y1,temp1))
			dist2.append((list_y1,temp2))

		dist1.sort(key=operator.itemgetter(1))
		dist2.sort(key=operator.itemgetter(1))
		for k in range(8):
			if k%2>0:
				idx=(k-1)/2
				#print k,idx
				if (len(L1_accuracy)<idx+1):
					L1_accuracy.append(0)
					L2_accuracy.append(0)
				#print dist1
				for x in range(k):
					k_neighbour_L1.append((dist1[x][0][-1],dist1[x][1]))
					k_neighbour_L2.append((dist2[x][0][-1],dist2[x][1]))
				kClass = {}
				for x in range(len(k_neighbour_L1)):
					glass_type = k_neighbour_L1[x][0]
					if glass_type in kClass:
						kClass[glass_type] += 1
					else:
						kClass[glass_type] = 1
				sortedClass = sorted(kClass.iteritems(), key=operator.itemgetter(1), reverse=True)
				#print sortedClass
				tie=[]
				for x in kClass:
					if kClass[x]==sortedClass[0][1]:
					  tie.append(x)
				ans=None
				if len(tie)>1:
					#print 'tie',tie,k_neighbour_L1
					for i in range(len(k_neighbour_L1)):
						for x in tie:
							#print i,x,k_neighbour_L1[i][0]
							if x==k_neighbour_L1[i][0]:
								#print 'answer',k_neighbour_L1[i]
								ans=i
								break
						if(ans!=None):
							break
					L1_predicted=k_neighbour_L1[ans][0]
				else:
					L1_predicted=sortedClass[0][0]
				kClass = {}
				for x in range(len(k_neighbour_L2)):
					glass_type = k_neighbour_L2[x][0]
					if glass_type in kClass:
						kClass[glass_type] += 1
					else:
						kClass[glass_type] = 1
				sortedClass = sorted(kClass.iteritems(), key=operator.itemgetter(1), reverse=True)
				tie=[]
				for x in kClass:
					if kClass[x]==sortedClass[0][1]:
					  tie.append(x)
				ans=None
				if len(tie)>1:
					#print 'tie',tie,k_neighbour_L2
					for i in range(len(k_neighbour_L2)):
						for x in tie:
							#print i,x,k_neighbour_L1[i][0]
							if x==k_neighbour_L2[i][0]:
								#print 'answer',k_neighbour_L2[i]
								ans=i
								break
						if(ans!=None):
							break

					L2_predicted=k_neighbour_L2[ans][0]
				else:
					L2_predicted=sortedClass[0][0]
				actual=df_test_norm.iloc[i,10]
				if(L1_predicted==actual):
					L1_accuracy[idx]=L1_accuracy[idx]+1
				if(L2_predicted==actual):
					L2_accuracy[idx]=L2_accuracy[idx]+1
		#print 'k=',k, df_test_norm.iloc[i,0],' L1  ',L1_predicted ,' L2  ',L2_predicted ,'Actual',actual,L1_accuracy,L2_accuracy,'\n'

	#accuracy
	total=len(df_test_norm)
	#print L1_accuracy,L2_accuracy,total
	print 'k=1\t\tL1 :' ,(L1_accuracy[0]/float(total))*100,'L2 :' ,(L2_accuracy[0]/float(total))*100
	print 'k=3\t\tL1 :' ,(L1_accuracy[1]/float(total))*100,'L2 :' ,(L2_accuracy[1]/float(total))*100
	print 'k=5\t\tL1 :' ,(L1_accuracy[2]/float(total))*100,'L2 :' ,(L2_accuracy[2]/float(total))*100
	print 'k=7\t\tL1 :' ,(L1_accuracy[3]/float(total))*100,'L2 :' ,(L2_accuracy[3]/float(total))*100

def knn():
	trainingSet=pd.read_csv('train.txt',header=None)
	testSet=pd.read_csv('test.txt',header=None)
	#df_train=pd.DataFrame(trainingSet,columns=[ 'Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type_glass'])
	df_train=pd.DataFrame(trainingSet)
	df_test=pd.DataFrame(testSet)
	# calculate mu, sigma
	mean=df_train.mean()
	stdev=df_train.std()
	variance=df_train.var()
	#print mean ,stdev
	# normalize the data
	df_train_norm=df_train.copy()
	df_test_norm=df_test.copy()
	df_train_norm=(df_train_norm-mean)/stdev
	df_test_norm=(df_test_norm-mean)/stdev
	df_train_norm[[0]]=df_train[[0]]
	df_test_norm[[0]]=df_test[[0]]
	df_train_norm[[10]]=df_train[[10]]
	df_test_norm[[10]]=df_test[[10]]

	'''# check 100 % without leave-one-out
	knn_test(df_test,df_train,1)
	knn_test(df_test,df_train,3)
	knn_test(df_test,df_train,5)
	knn_test(df_test,df_train,7)'''

	#print df_train_norm
	print 'kNN Test Accuracy '
	knn_test(df_test_norm,df_train_norm,1)
	knn_test(df_test_norm,df_train_norm,3)
	knn_test(df_test_norm,df_train_norm,5)
	knn_test(df_test_norm,df_train_norm,7)

	print 'kNN Training LOO Accuracy '
	knn_LOO(df_train_norm)
