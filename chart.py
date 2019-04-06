  """Charts module for regressor"""

import os

import numpy as np
from sklearn.learning_curve import learning_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve 


class LearningCurve(object):
    """scikit-learn Learning curve class"""

    def __init__(self, dirname):
        self.dirname = dirname
        self.regressor = None

    def set_regressor(self, regressor):
        """Set a regressor"""
        self.regressor = regressor

    def store(self, X, y, figure_id=1):
        """Save the learning curve"""

        plt.figure(figure_id)
        plt.xlabel("Training samples")
        plt.ylabel("Error")

        train_sizes, train_scores, test_scores = learning_curve(self.classifier, X, y[:, 0])

        train_error_mean = 1 - np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_error_mean = 1 - np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_error_mean + train_scores_std,
                         train_error_mean - train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_error_mean + test_scores_std,
                         test_error_mean - test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_error_mean, 'o-', color="r",
                 label="Training error")
        plt.plot(train_sizes, test_error_mean, 'o-', color="g",
                 label="Cross-validation error")
        plt.legend(loc="best")

        filepath = os.path.join(self.dirname, 'learning-curve.png')
        plt.savefig(filepath, format='png')

        if not os.path.isfile(filepath):
            return False

        return filepath
    
   #Determine training and test scores for varying parameter values.
  class ValidationCurve(object):
  
  """Class to generate a Validation curve chart through scikit-learn"""
   
    def __init__(self, dirname, figid):
        self.dirname = dirname
        plt.figure(figid)

    @staticmethod
    def add(fpr, tpr, label):
        """Add data to the chart"""
         plt.plot(fpr, tpr, label=label)
    
    def store(self):

        train_scores, test_scores = validation_curve(
            self.regressor, X, y)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Validation Curve")
        plt.xlabel(r"$\gamma$")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        lw = 2
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")

        filepath = os.path.join(self.dirname, 'validation-curve.png')
        plt.savefig(filepath, format='png')

        if not os.path.isfile(filepath):
             return False

        return filepath    


    
    
    
    
 
