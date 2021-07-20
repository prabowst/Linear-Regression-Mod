import numpy as np
import sys

from collections import deque

class PacLinearRegression():   
    """
    A class used to fit a linear regression to a dataset, given features X and 
    label y provided that both are in the form of numpy arrays.

    ...

    Attributes
    ----------
    learning_rate : float
        learning rate of the linear regression for gradient descent
    cost_function : str
        type of cost function to be used (mae or mse)
    max_iter : int
        number of gradient descent iteration (10x higher for mae)
    X : array of float/int
        the feature array of the dataset
    y : array of float/int
        the label array of the dataset
    theta0 : arrray of float
        slope parameter(s) of linear regression
    theta1 : float
        intercept/bias parameter of linear regression
    m : int
        number of data points
    cost_hist: list of float
        the historical cost function values
    coeff_: list of float
        the list representation of theta0 and theta1, respectively

    Methods
    -------
    fit(X, y):
        Fit a linear regression to a data by calling gradientDescent method.
    computeCostFunction(y_pred):
        Return cost value per specified cost function.
    gradientCalculation(y_pred):
        Calculate gradients for theta0 and theta1 provided the type of cost
        function.
    gradientDescent():
        Create a loop of gradient descent method, appending cost_hist per iter.
    predict(X):
        Predict the label given feature X.
    score(y_val, y_pred):
        Return the R-squared metric of the model.
    """

    def __init__(self, max_iter=1e4, learning_rate=1e-4, 
                cost_function='mse', early_stop=False):
        """
        Constructs necessary attributes for the PacLinearRegression object.
        Attributes are introduced within fit method for a more concise
        structure.

        Parameters
        ----------
            learning_rate : float
                learning rate of the linear regression for gradient descent
            cost_function : str
                type of cost function to be used (mae or mse)
            max_iter : int
                number of gradient descent iteration (10x higher for mae)
        """

        self.learning_rate = learning_rate
        self.cost_function = cost_function
        if cost_function == 'mae':
            self.max_iter = int(max_iter * 10)
        else:
            self.max_iter = int(max_iter)
        self.early_stop = early_stop

    def fit(self, X, y):
        """
        Fit a linear regression to a data by calling gradientDescent method.

        Parameters
        ----------
            X : array of float/int
                the feature array of the dataset
            y : array of float/int
                the label array of the dataset
            theta0 : arrray of float
                slope parameter(s) of linear regression
            theta1 : float
                intercept/bias parameter of linear regression
            m : int
                number of data points
            cost_hist: list of float
                the historical cost function values
            coeff_: list of float
                the list representation of theta0 and theta1, respectively
        
        Returns
        -------
            self : fitted PacLinearRegression class
        """

        try:
            self.X = X.reshape(len(X), X.shape[1])
            self.theta0 = np.zeros(X.shape[1]).reshape(X.shape[1], -1)
        except:
            self.X = X.reshape(len(X), 1)
            self.theta0 = 0
        self.y = y.reshape(len(y), 1)
        self.theta1 = 0
        self.m = len(X)
        self.cost_hist = self.gradientDescent()
        self.coeff_ = [i[0] for i in self.theta0] + [self.theta1]
        return self

    def computeCostFunction(self, y_pred):
        """Return cost value per specified cost function."""
        if self.cost_function == 'mse':
            return (1/self.m) * sum((self.y - y_pred)**2)
        elif self.cost_function == 'mae':
            return (1/self.m) * sum(abs((self.y - y_pred)))
        else:
            print('Error: cost_function is not valid.')
            sys.exit()

    def gradientCalculation(self, y_pred):
        """
        Calculate gradients for theta0 and theta1 provided the type of cost
        function.

        Parameters
        ----------
            y_pred : array of float/int
                the predicted label of current iteration.
        
        Returns
        -------
            None
        """

        if self.cost_function == 'mse':
            dtheta0 = (-1/self.m) * self.X.T.dot(self.y - y_pred)
            dtheta1 = (-1/self.m) * np.sum(self.y - y_pred)
        elif self.cost_function == 'mae':
            y_adj = self.y - y_pred 
            y_adj[y_adj==0] = 0
            y_adj[y_adj<0] = -1
            y_adj[y_adj>0] = 1
            dtheta0 = (-1/self.m) * np.sum(y_adj)
            dtheta1 = (-1/self.m) * np.sum(y_adj)
        self.theta0 -= self.learning_rate * dtheta0
        self.theta1 -= self.learning_rate * dtheta1

    def gradientDescent(self):
        """
        Create a loop of gradient descent method, appending cost_hist per iter.
        An early stopping criterion is also implemented using relative error of
        average cost function per 100 appended values. The value set is 0.005%.

        Parameters
        ----------
            None
        
        Returns
        -------
            cost_hist: list of float
                the historical cost function values
        """

        cost_hist, cost_threshold = [], []
        cost_mean = deque(maxlen=100)
        threshold = 100
        for i in range(self.max_iter):
            y_pred = self.predict(self.X)
            cost_hist.append(self.computeCostFunction(y_pred))
            cost_mean.append(self.computeCostFunction(y_pred))
            if (i+1 >= threshold) and self.early_stop:
                cost_threshold.append(np.mean(cost_mean))
                try:
                    cut_off = abs((cost_threshold[-1] - cost_threshold[-2]) \
                                * 100 / cost_threshold[-2])
                    if cut_off < 0.001: 
                        self.max_iter = len(cost_hist)
                        return cost_hist
                except:
                    continue
            self.gradientCalculation(y_pred)
        return cost_hist

    def predict(self, X):
        """Predict the label given feature X."""
        return X.dot(self.theta0) + self.theta1

    def score(self, y_val, y_pred):
        """Return the R-squared metric of the model."""
        y_pred = [i[0] for i in y_pred]
        y_bar = np.mean(y_val)
        denom = np.sum((y_bar - y_val)**2)
        numer = (denom - np.sum((y_pred - y_val)**2))
        R_squared = numer / denom
        return R_squared