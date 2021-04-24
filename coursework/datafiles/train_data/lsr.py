import os
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


#####################################################################################
'''Least Squares formula
    A = (X.T . X)^-1 . X.T . Y'''

def squared_error(y, y_hat):
    return np.sum((y- y_hat) ** 2)

def fit(X, y_1):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_1)

def lsr_linear(x_1,y_1):
    # x_1 = np.matrix(x_1).reshape(20,1)
    # y_1 = np.matrix(y_1).reshape(20,1)

    X = np.c_[np.ones(x_1.shape), x_1]
    A = fit(X,y_1)

    #reconstruction error
    error = 0
    for i in range(0,len(x_1)):
        error += squared_error((A[0]+A[1]*x_1[i]), y_1[i])
    return A, error

#Function used to find best polynomial order
def poly_order_finder(x_1, y_1, order):
    X = np.c_[np.ones(x_1.shape), x_1]
    for i in range(2, order + 1):
        X = np.c_[X, x_1**i]

    A = fit(X,y_1)

    error = squared_error((X.dot(A)), y_1)
    return A, error


def lsr_cubic(x_1,y_1):

    # x_1 = np.matrix(x_1).reshape(20,1)
    # y_1 = np.matrix(y_1).reshape(20,1)

    X = np.c_[np.ones(x_1.shape), x_1, np.power(x_1,2), np.power(x_1,3)]
    A = fit(X,y_1)
    #reconstruction error
    error = 0
    for i in range(0,len(x_1)):
        #dx^3+cx^2+bx+a
        error += squared_error((A[0] + A[1]*x_1[i] + A[2]*np.power(x_1[i],2) + A[3]*np.power(x_1[i],3)), y_1[i])
    return A, error

def lsr_sine(x_1,y_1):
    # x_1 = np.matrix(x_1).reshape(20,1)
    # y_1 = np.matrix(y_1).reshape(20,1)

    X = np.c_[np.ones(x_1.shape), np.sin(x_1)]
    A = A = fit(X,y_1)
    #reconstruction error
    error = 0
    for i in range(0,len(x_1)):
        error += squared_error((A[0]+A[1]*np.sin(x_1[i])), y_1[i])
    return A,error


#TODO:
#1.Shuffle the dataset randomly. X
#Split the dataset into k groups X
#For each unique group:
    #Take the group as a hold out or test data set X
    #Take the remaining groups as a training data set X
    #Fit a model on the training set and evaluate it on the test set X
    #Retain the evaluation score and discard the model X
#Summarize the skill of the model using the sample of model evaluation scores X

def k_fold(xs, ys):
    """Implementing the K_fold algorithm to resolve overfitting the data
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        minimum error
    """
    k = 10
    lin_error = []
    poly_error = []
    sine_error =[]
    #Shuffling data randomly whilst preseving the pairs of the data
    arr = list(zip(xs,ys))
    np.random.seed(6)
    np.random.shuffle(arr)

    x_shuffled, y_shuffled = zip(*arr)
    x = np.array_split(np.array(x_shuffled),k)
    y = np.array_split(np.array(y_shuffled),k)
    x = np.array(x)
    y = np.array(y)
    #Split dataset into K groups
    #splitting into test and training sets
    for i in range(k):
        x_test = x[i]
        x_train_set = np.concatenate((x[0:i], x[i+1:]))
        y_test = y[i]
        y_train_set = np.concatenate((y[0:i], y[i+1:]))

        # X_train_set,x_test = (remove_this_element(x,i))
        # Y_train_set,y_test = (remove_this_element(y,i))

        x_train = np.concatenate(x_train_set)
        y_train = np.concatenate(y_train_set)
        
        # train_test,error_test = poly_order_finder(x_train,y_train,2)


        train_lin, error_line = lsr_linear(x_train,y_train)
        train_poly, error_ploy = lsr_cubic(x_train,y_train)
        train_sine, error_sine = lsr_sine(x_train,y_train)

        # poly2 = fit(poly_order_finder(x_train,2),y_train)
        # poly3 = fit(poly_order_finder(x_train,3),y_train)
        # poly4 = fit(poly_order_finder(x_train,4),y_train)
        # poly5 = fit(poly_order_finder(x_train,5),y_train)

        # test_poly = np.c_[np.ones(x_test.shape), x_test, np.power(x_test,2), np.power(x_test,3), np.power(x_test,4)].dot(poly4)
        test_lin  = np.c_[np.ones(x_test.shape), x_test].dot(train_lin)
        test_poly = np.c_[np.ones(x_test.shape), x_test, np.power(x_test,2), np.power(x_test,3)].dot(train_poly)
        test_sine = np.c_[np.ones(x_test.shape), np.sin(x_test)].dot(train_sine)

        lin_error.append(squared_error(y_test, test_lin))
        poly_error.append(squared_error(test_poly,y_test))

        sine_error.append(squared_error(test_sine,y_test))

    
    return np.mean(lin_error), np.mean(poly_error), np.mean(sine_error)




if __name__ == "__main__":
    is_plot = False
    filename = str(sys.argv[1])
    if len(sys.argv)>2:
        if sys.argv[2] == "--plot":
            is_plot = True
        else:
            is_plot = False

    xs, ys = load_points_from_file(filename)
    xs = np.array(xs)
    ys = np.array(ys)
    num_segments =len(xs)//20
    total_error = 0 
    for i in range(num_segments):
        xs_1 = xs[i*20:(i*20)+20]
        ys_1 = ys[i*20:(i*20)+20]
        l_error, p_error, s_erro = k_fold(xs_1,ys_1)
        A, error_linear = lsr_linear(xs_1,ys_1)
        B, error_ploy = poly_order_finder(xs_1,ys_1,4)
        # B, error_ploy = lsr_cubic(xs_1,ys_1)
        C, error_sine = lsr_sine(xs_1,ys_1)

        #checking to see which has min error for each segment
        min_error = min(l_error, p_error,s_erro)
        if min_error == l_error:
            print("Linear")
            total_error += error_linear
        elif min_error == p_error:
            print("Poly")
            total_error += error_ploy
        elif min_error == s_erro:
            print("Sine")
            total_error += error_sine
    print("total")
    print(total_error)
    if is_plot:
        view_data_segments(xs,ys)

    plt.show()


