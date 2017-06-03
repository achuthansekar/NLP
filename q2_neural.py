import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    
    

    ### YOUR CODE HERE: forward propagation
    i = np.ones([data.shape[0],1])
    h = sigmoid(data.dot(W1)+i.dot(b1))
    y_hat = softmax(h.dot(W2)+ i.dot(b2)) 
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    gradW2 = (h.T.dot((y_hat-labels)))
    gradb2 = np.reshape(np.sum(y_hat-labels,axis=0),[1,Dy])
    gradW1 = data.T.dot(((y_hat-labels).dot(W2.T))*(h*(1-h)))
    gradb1 = np.reshape(np.sum(((y_hat-labels).dot(W2.T))*h*(1-h),axis=0),[1,H])
    cost = np.sum(labels*np.log(y_hat))*-1  
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print ("Running sanity check...")

    N = 100
    dimensions = [15, 10, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    #print(labels)
    for i in range(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
    
    #print(forward_backward_prop(data,labels,params,dimensions)[1].shape)
    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()