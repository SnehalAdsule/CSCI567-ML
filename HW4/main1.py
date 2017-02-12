import hw_utils
import time

def start_time():
    start_time = time.time()
    return start_time

def end_time(start_time):
    print "--- %s seconds ---" % (time.time() - start_time)
    return time.time() - start_time


print ' 3 (c) 1 Load Data'

X_tr,y_tr,X_te,y_te = hw_utils.loaddata('min_MiniBooNE_PID.txt')
print X_tr.shape ,X_te.shape, y_tr.shape, y_te.shape

print ' 3 (c) 2 Normalization'
X_tr, X_te=hw_utils.normalize(X_tr, X_te)
print X_tr.shape ,X_te.shape

din=50
dout=2

arch_list1=[[din, dout], [din, 50, dout], [din, 50, 50, dout], [din, 50,50, 50, dout]]
print arch_list1
'''
print ' 3 (d) 1 Linear Activation'
time_d1 = start_time()
hw_utils.testmodels(X_tr, y_tr, X_te, y_te, arch_list1, actfn='linear', last_act='softmax', reg_coeffs=[0.0],
				num_epoch=30, batch_size=1000, sgd_lr=1e-5, sgd_decays=[0.0], sgd_moms=[0.0],
					sgd_Nesterov=False, EStop=False, verbose=0)
end_time(time_d1)

print ' 3 (d) 2 Linear Activation'
time_d2=start_time()
arch_list2=[[din, 50, dout], [din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout],
[din, 800, 800, 500, 300, dout]]
hw_utils.testmodels(X_tr, y_tr, X_te, y_te, arch_list2, actfn='linear', last_act='softmax', reg_coeffs=[0.0],
				num_epoch=30, batch_size=1000, sgd_lr=1e-5, sgd_decays=[0.0], sgd_moms=[0.0],
					sgd_Nesterov=False, EStop=False, verbose=0)
end_time(time_d2)


print ' 3 (e) 1 Sigmoid Activation'
time_e=start_time()
arch_list_e=[[din, 50, dout],[din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]

hw_utils.testmodels(X_tr, y_tr, X_te, y_te, arch_list_e, actfn='sigmoid', last_act='softmax', reg_coeffs=[0.0],
                    num_epoch=30, batch_size=1000, sgd_lr=1e-5, sgd_decays=[0.0], sgd_moms=[0.0],sgd_Nesterov=False, EStop=False, verbose=0)
end_time(time_e)

print ' 3 (f) 1'
time_e=start_time()
arch_list_e=[[din, 50, dout],[din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]

hw_utils.testmodels(X_tr, y_tr, X_te, y_te, arch_list_e, actfn='relu', last_act='softmax', reg_coeffs=[0.0],
				num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0],sgd_Nesterov=False, EStop=False, verbose=0)
end_time(time_e)

print ' 3 (g) 1'
time_g=start_time()
arch_list_e=[[din, 800, 500, 300, dout]]
param_list=[10e-7, 50e-7, 10e-6, 50e-6, 10e-5]
hw_utils.testmodels(X_tr, y_tr, X_te, y_te, arch_list_e, actfn='relu', last_act='softmax', reg_coeffs=param_list,
				num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0],
					sgd_Nesterov=False, EStop=False, verbose=0)
end_time(time_g)
'''

print ' 3 (h) Early Stopping and L2-regularization'
time_g=start_time()
arch_list_e=[[din, 800, 500, 300, dout]]
param_list=[10e-7, 50e-7, 10e-6, 50e-6, 10e-5]
best_h = hw_utils.testmodels(X_tr, y_tr, X_te, y_te, arch_list_e, actfn='relu', last_act='softmax', reg_coeffs=param_list,
				num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0],
					sgd_Nesterov=False, EStop=True, verbose=0)
end_time(time_g)

print ' 3 (i) SGD with weight decay'
time_g=start_time()
arch_list_e=[[din, 800, 500, 300, dout]]
param_list=[50e-7]
decay=[10e-5, 50e-5, 10e-4, 30e-4, 70e-4, 10e-3]
best_i =hw_utils.testmodels(X_tr, y_tr, X_te, y_te, arch_list_e, actfn='relu', last_act='softmax', reg_coeffs=param_list,
				num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=[decay], sgd_moms=[0.0],
					sgd_Nesterov=False, EStop=False, verbose=0)
end_time(time_g)

print ' 3 (j) Momentum'
time_g=start_time()
arch_list_e=[[din, 800, 500, 300, dout]]
decay=best_i[2]
best_j=hw_utils.testmodels(X_tr, y_tr, X_te, y_te, arch_list_e, actfn='relu', last_act='softmax', reg_coeffs=[0.0],
				num_epoch=50, batch_size=1000, sgd_lr=1e-4, sgd_decays=[decay], sgd_moms=[0.99, 0.98, 0.95, 0.9,0.85],
                sgd_Nesterov=True, EStop=False, verbose=0)
end_time(time_g)

print ' 3 (k) Combining the above'
time_g=start_time()
arch_list_e=[[din, 800, 500, 300, dout]]
reg_co=best_h[1]
decay=best_i[2]
mom_co=best_j[3]
hw_utils.testmodels(X_tr, y_tr, X_te, y_te, arch_list_e, actfn='relu', last_act='softmax', reg_coeffs=[reg_co],
				num_epoch=100, batch_size=1000, sgd_lr=1e-4, sgd_decays=[decay], sgd_moms=[mom_co],
                sgd_Nesterov=True, EStop=True, verbose=0)
end_time(time_g)


print ' 3 (l) Grid search with cross-validation'
time_g=start_time()
arch_list_e=[[din, 800, 500, 300, dout]]
reg_cos=[10e-7,50e-7, 10e-6, 50e-6, 10e-5]
decays=[10e-5, 50e-5, 10e-4]
mom_cos=[0.99]
hw_utils.testmodels(X_tr, y_tr, X_te, y_te, arch_list_e, actfn='relu', last_act='softmax', reg_coeffs=reg_cos,
				num_epoch=100, batch_size=1000, sgd_lr=1e-4, sgd_decays=decays, sgd_moms=mom_cos,
                sgd_Nesterov=True, EStop=True, verbose=0)
end_time(time_g)
[din, 50, dout], [din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800,
500, 300, dout];
print 'done'
