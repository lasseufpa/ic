import numpy as np
lr = 0.1 #learning rate
w0 = np.array([0.2, -0.2])
g1=np.array([1,-3])
g2=np.array([2,0.5])
g3=np.array([-0.3,0.3])
g4=np.array([8,5])
g5=np.array([9,-3])
g6=np.array([1,8])

def apply_gradient(current_value, gradient):
    new_value = current_value - lr*gradient    
    print("new_value=", new_value, "from current_value=", current_value, "-lr*gradient=", -lr*gradient)
    return new_value

print("a) B=1 example, SGD case:")
w1 = apply_gradient(w0, g1) #w1 = w0 - lr*g1
w2 = apply_gradient(w1, g2) #w2 = w1 - lr*g2
w3 = apply_gradient(w2, g3)
w4 = apply_gradient(w3, g4)
w5 = apply_gradient(w4, g5)
w6 = apply_gradient(w5, g6)
wa = w6 #final weight vector

print("b) mini batch with B=2 examples:")
grad_first_batch = 0.5*(g1+g2)
w1 = apply_gradient(w0, grad_first_batch) #w1 = w0 - lr*(g1+g2)/2
grad_second_batch = 0.5*(g3+g4)
w2 = apply_gradient(w1, grad_second_batch) #w2 = w1 - lr*(g3+g4)/2
grad_third_batch = 0.5*(g5+g6)
w3 = apply_gradient(w2, grad_third_batch) #w3 = w2 - lr*(g5+g6)/2
wb = w3 #final weight vector

print("c) mini batch with B=3 examples:")
grad_first_batch = (g1+g2+g3)/3.0
w1 = apply_gradient(w0, grad_first_batch) #w1 = w0 - lr*(g1+g2+g3)/3
grad_second_batch = (g4+g5+g6)/3.0
w2 = apply_gradient(w1, grad_second_batch) #w2 = w1 - lr*(g4+g5+g6)/3
wc = w2 #final weight vector

print("d) full batch with B=6 examples:")
grad_first_batch = (g1+g2+g3+g4+g5+g6)/6.0
w1 = apply_gradient(w0, grad_first_batch) #w1 = w0 - lr*(g1+g2+g3+g4+g5+g6)/6
wd = w1 #final weight vector

#assuming the optimum value is w_opt = [4, 1], we have the following Euclidean norms
w_opt = [4, 1]
print("Distance from a): ", np.sqrt( np.sum( (wa-w_opt)**2 ) ))
print("Distance from b): ", np.sqrt( np.sum( (wb-w_opt)**2 ) ))
print("Distance from c): ", np.sqrt( np.sum( (wc-w_opt)**2 ) ))
print("Distance from d): ", np.sqrt( np.sum( (wd-w_opt)**2 ) ))