import naive_bayes as nb
import kNN
from datetime import datetime

start_time = datetime.now()
#print start_time
nb.nb()
end_time = datetime.now()
#start_time=end_time
#print 'Duration: {}'.format(end_time - start_time)
print '=========================================='
kNN.knn()
end_time = datetime.now()
#print 'Duration: {}'.format(end_time - start_time)


