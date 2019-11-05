import numpy as np                                             
import matplotlib.pyplot as pl                                 

# define the sigmoid function f and its derivative df
def  f(h,beta): return 1/(1+np.exp(-beta*h)) 
def df(h,beta): return np.divide(beta*np.exp(-beta*h),np.power(1+np.exp(-beta*h),2))

# setup
configs = np.matrix([[0.,0.,1.,1.],[0.,1.,0.,1.]])             # a predefined matrix of the four configurations (from which we can sample several examples at once)
T       = np.matrix([1.,0.,0.,1.])                             # the desired output for each configuration
Wz      = np.random.randn(2,3)                                 # random weights from input to hidden layer
Wy      = np.random.randn(1,3)                                 # random weights from hidden to output layer
eta     =    0.5                                               # the learning rate
beta    =    2.0                                               # a large steepness parameter to make the sigmoid behave like a binary function
steps   =    2500                                              # number of iterations in the learning procedure
b       =    10
error   = np.zeros(steps)                                      # a vector for storing the error at each iteration

# training
for i in range(steps):                                         # a for-loop over the training iterations
    s        = np.random.randint(4,size=b)                     # randomly select one configuration
    x        = np.vstack([configs[:,s],np.ones(b)])            # the input vector x corresponding to a number of randomly chosen examples
    hz       = Wz*x                                            # net input to hidden layer
    z        = np.vstack([f(hz,beta),np.ones(b)])              # activation of hidden layer neurons
    z_prime  = df(hz,beta)                                     # derivative of activation function (hidden layer)
    hy       = Wy*z                                            # net input to output layer
    y        = f(hy,beta)                                      # activation of output neuron
    y_prime  = df(hy,beta)                                     # derivative of activation function (hidden layer)
    
    delta    = np.multiply((T[0,s]-y),y_prime)                 # calculate the delta for the hidden layer
    dWy      = eta*delta*np.transpose(z)                       # calculate desired change for weights from hidden to output neurons
    delta    = np.transpose(Wy)*delta                          # calculate the delta for the input layer
    dWz      = (eta*np.multiply(delta[[0,1]],z_prime)*         # calculate desired change for weights from input to hidden neurons
                np.transpose(x)) 
    Wy       = Wy+dWy                                          # adjusting weights from hidden to output neurons
    Wz       = Wz+dWz                                          # adjusting  weights from input to hidden neurons
    error[i] = .5*np.sum(np.power(T[0,s]-y,2))                 # measurement of error at each iteration
    
pl.plot(error)                                                 # plot the development of the error over iterations 
pl.ylim([0,2.5])
pl.xlabel('iteration')
pl.ylabel('error')

#testing
for i in range(4):                                             # a for-loop over the 4 configurations
    x        = np.vstack([configs[:,i],1.])                    # the input vector x corresponding to the ith configuration and the bias unit
    hz       = Wz*x                                            # net input
    z        = np.vstack([f(hz,beta),1.])
    hy       = Wy*z
    y        = f(hy,beta) 
    print("%d. x1 = %d, x2 = %d, y = %.2f" % (i+1, x[0],x[1],y))# printing the results
    
pl.show()