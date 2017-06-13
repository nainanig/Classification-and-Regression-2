import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import pickle


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))



def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """


    train_data, labeli = args
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    #ERROR

    bias=np.ones((n_data,1));
    train_data=np.concatenate((bias,train_data),axis=1)
    #print(tr_data.shape)
    #print("w_trans",w_trans.shape)

    #x_trans=np.transpose(tr_data);
    #print("x_tran",x_trans.shape);

  
    temp1= np.dot(train_data,initialWeights);
    theta=sigmoid(temp1);
    #print(theta)
    theta=theta.reshape(theta.shape[0],1)

    #temp2=1-theta;
    log1=np.log(1-theta)
    temp3=1-labeli;
    part_1=np.multiply(temp3,log1)
   
    #temp3=temp3.reshape(n_data,1)
    
    #print("calculated step 1")
    
    log2=np.log(theta)
    part_2=np.multiply(labeli,log2)
    #temp5=np.transpose(labeli)

    #temp6=labeli*log2
    #temp7=temp6+temp4
    #print("calculated step 2")
    
    mid=-np.sum(part_2+part_1)
    error=mid/n_data
    #print("calculated step 3")

    print("error",error)


    #GRADIENT

    #print("theta",theta.shape)
    #print("label",labeli.shape)
    #print("train",train_data.shape)
    train_data_t=np.transpose(train_data)
    temp_cal=np.multiply((theta-labeli),train_data);
    temp_cal_1=np.sum(temp_cal,axis=0);
    error_grad=temp_cal_1/n_data;

    print("complted blr obj function")
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    #print("started blr prdict")
    label = np.zeros((data.shape[0], 1))

    bias=np.ones((data.shape[0],1));
    x=np.concatenate((bias,data),axis=1)

    w_x=np.dot(x,W)
    sigmoid_w_x=sigmoid(w_x)
    label=np.argmax(sigmoid_w_x, axis=1)
    label=label.reshape(label.shape[0],1)

    #print("complted blr predict function")

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data ,labeli=args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    print("entering mlr obj")
    weight_mat= params.reshape((n_feature+1,n_class))
    bias2= np.ones((n_data,1))
    x_data=np.concatenate((bias2,train_data),axis=1)
    
    wt_trans=np.transpose(weight_mat)
    xTr= np.transpose(x_data)
    
    #wt_x=np.dot(wt_trans,xTr)
    
    wt_x=np.dot(x_data,weight_mat)
    numerator=np.exp(wt_x)
    
    #part2=np.sum(part1)
    exp_sum=np.sum(numerator,axis=1)
    #exp_sum = np.tile(np.sum(numerator, axis = 1), (10,1)).T
    exp_sum = exp_sum.reshape(exp_sum.shape[0],1)
    #print(exp_sum.shape)
    theta2=numerator/exp_sum
    #theta2=part1/part2


    log_theta=np.log(theta2)
    #print("log theta ",log_theta.shape)
    #labeli=labeli.reshape(labeli.shape[0],n_class)
    #yTheta=np.dot(labeli,log_theta) 
    labeli = np.array(labeli)
    labeli=labeli.reshape(labeli.shape[0],n_class)
    yTheta=labeli*log_theta
    error=-(np.sum(yTheta))/n_data
    
    #print("theta2",theta2.shape)
    #print("labeli",labeli.shape)

    #GRADIENT

    labeli_tran=np.transpose(labeli)
    #print("labeli transpose",labeli_tran.shape)
    #print("x_data",x_data.shape)
    #print("x_Data_transpsed",x_data_tran.shape)
    theta2 = np.array(theta2)
    theta2 = theta2.reshape(theta2.shape[0],n_class)
    error_grad=np.dot(xTr,theta2-labeli)/n_data
    error_grad=error_grad.flatten()
    #print(error_grad_t.shape)
    #error_grad= np.sum(error_grad_t)/n_data
    print("finished mlr obj")



    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data print("entering the mlr predict")

    print("started mlr predict")
    size=data.shape[0]

    bias3= np.ones((size,1))
    x=np.concatenate((bias3,data),axis=1)
    x_w=np.dot(x,W)
    part3=np.exp(x_w)
    part4=np.sum(part3,axis=1)
    part4=part4.reshape(part4.shape[0],1)
    part=part3/part4
    label=np.argmax(part,axis=1)
    label=label.reshape((label.shape[0],1))

    print("finished the mlr predict")

    return label

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################


print('\n\n-----Linear-----\n\n')
print("started")
clf = SVC(kernel='linear')
clf.fit(train_data,train_label.ravel())

output_TR=clf.predict(train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((output_TR == train_label.ravel()).astype(float))) + '%')

output_V=clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((output_V == validation_label.ravel()).astype(float))) + '%')

output_TS=clf.predict(test_data)
print('\n Test set Accuracy:' + str(100 * np.mean((output_TS == test_label.ravel()).astype(float))) + '%')

print("successful")

print('\n\n-----RBF GAMMA=1-----\n\n')
print("started")
clf_rbf= SVC(kernel='rbf',gamma=1)
clf_rbf.fit(train_data,train_label.ravel())

output_TR_rbf=clf_rbf.predict(train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((output_TR_rbf == train_label.ravel()).astype(float))) + '%')

output_V_rbf=clf_rbf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((output_V_rbf == validation_label.ravel()).astype(float))) + '%')

output_TS_rbf=clf_rbf.predict(test_data)
print('\n Test set Accuracy:' + str(100 * np.mean((output_TS_rbf == test_label.ravel()).astype(float))) + '%')


print('\n\n-----RBF DEFAULT-----\n\n')
print("started")
clf_rbf_d= SVC(kernel='rbf')
clf_rbf_d.fit(train_data,train_label.ravel())

output_TR_rbf_d=clf_rbf_d.predict(train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((output_TR_rbf_d == train_label.ravel()).astype(float))) + '%')

output_V_rbf_d=clf_rbf_d.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((output_V_rbf_d == validation_label.ravel()).astype(float))) + '%')

output_TS_rbf_d=clf_rbf_d.predict(test_data)
print('\n Test set Accuracy:' + str(100 * np.mean((output_TS_rbf_d == test_label.ravel()).astype(float))) + '%')
print("successful")


print('\n\n-----RBF now for various values of C-----\n\n')


print("for value C=1")
clf_rbf_c1= SVC(kernel='rbf',C=1)
clf_rbf_c1.fit(train_data,train_label.ravel())
    
output_TR_rbf_c1=clf_rbf_c1.predict(train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((output_TR_rbf_c1 == train_label.ravel()).astype(float))) + '%')

output_V_rbf_c1=clf_rbf_c1.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((output_V_rbf_c1 == validation_label.ravel()).astype(float))) + '%')

output_TS_rbf_c1=clf_rbf_c1.predict(test_data)
print('\n Test set Accuracy:' + str(100 * np.mean((output_TS_rbf_c1 == test_label.ravel()).astype(float))) + '%')


for i in range(10,110,10):
    print("Value of C",i)
    clf_rbf_c= SVC(kernel='rbf',C=i)
    clf_rbf_c.fit(train_data,train_label.ravel())
    
    output_TR_rbf_c=clf_rbf_c.predict(train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((output_TR_rbf_c == train_label.ravel()).astype(float))) + '%')

    output_V_rbf_c=clf_rbf_c.predict(validation_data)
    print('\n Validation set Accuracy:' + str(100 * np.mean((output_V_rbf_c == validation_label.ravel()).astype(float))) + '%')

    output_TS_rbf_c=clf_rbf_c.predict(test_data)
    print('\n Test set Accuracy:' + str(100 * np.mean((output_TS_rbf_c == test_label.ravel()).astype(float))) + '%')


"""
#Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')


f1 = open('params.pickle', 'wb') 
pickle.dump(W, f1) 
f1.close()

f2 = open('params_bonus.pickle', 'wb')
pickle.dump(W_b, f2)
f2.close()
