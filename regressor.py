""" This class is to be added to the estimator to add the regression capabilities"""

from __future__ import division

import math
import logging
import time
import warnings
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

from ..model import tensor
from .. import chart

from sklearn.utils import shuffle
from sklearn.externals import joblib


OK = 0
GENERAL_ERROR = 1
NO_DATASET = 2
LOW_SCORE = 4
NOT_ENOUGH_DATA = 8

PERSIST_FILENAME = 'classifier.pkl'
EXPORT_MODEL_FILENAME = 'model.json'


class Regressor(Estimator):
    """Regressor"""

    def __init__(self, modelid, directory):

        super(Regressor, self).__init__(modelid, directory)

        self.meansquare = []
        self.r2= []
        self.variancescore=[]
 

        self.tensor_logdir = self.get_tensor_logdir()
        if os.path.isdir(self.tensor_logdir) is False:
            if os.makedirs(self.tensor_logdir) is False:
                raise OSError('Directory ' + self.tensor_logdir +
                              ' can not be created.')

    def get_regressor(self, X, y, initial_weights=False,
                       force_n_features=False):
        """Gets the regressor"""

        n_epoch = 50
        batch_size = 1000
        starter_learning_rate = 0.5

        if force_n_features is not False:
            n_features = force_n_features
        else:
            _, n_features = X.shape

       """This can be implemented once the regressor model is decided and implemented"""
        return regressor_model()
        
    def get_tensor_logdir(self):
        """Returns the directory to store tensorflow framework logs"""
        return os.path.join(self.logsdir, 'tensor')

    def store_regressor(self, trained_classifier):
        """Stores the regressor and saves a checkpoint of the tensors state"""

        # Store the graph state.
        saver = tf.train.Saver()
        sess = trained_regressor.get_session()

        path = os.path.join(self.persistencedir, 'model.ckpt')
        saver.save(sess, path)

        # Also save it to the logs dir to see the embeddings.
        path = os.path.join(self.get_tensor_logdir(), 'model.ckpt')
        saver.save(sess, path)

        # Save the class data.
        super(Regressor, self).store_regressor(trained_regressor)

    def export_regressor(self, exporttmpdir):
        if self.regressor_exists():
            regressor = self.load_regressor()
        else:
            return False

        export_vars = {}

        # Get all the variables in in initialise-vars scope.
        sess = regressor.get_session()
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='initialise-vars'):
            # Converting to list as numpy arrays can't be serialised.
            export_vars[var.op.name] = var.eval(sess).tolist()

        # Append the number of features.
        export_vars['n_features'] = regressor.get_n_features()

        vars_file_path = os.path.join(exporttmpdir, EXPORT_MODEL_FILENAME)
        with open(vars_file_path, 'w') as vars_file:
            json.dump(export_vars, vars_file)

        return exporttmpdir

    def import_regressor(self, importdir):

        model_vars_filepath = os.path.join(importdir,
                                           EXPORT_MODEL_FILENAME)

        with open(model_vars_filepath) as vars_file:
            import_vars = json.load(vars_file)

        n_features = import_vars['n_features']

        regressor = self.get_regressor(False, False,
                                         initial_weights=import_vars,
                                         force_n_features=n_features)

        self.store_regressor(regressor)

    def load_regressor(self, model_dir=False):
        """Loads a previously trained regressor and restores its state"""

        if model_dir is False:
            model_dir = self.persistencedir

        regressor = super(Regressor, self).load_regressor(model_dir)

        regressor.set_tensor_logdir(self.get_tensor_logdir())

        # Now restore the graph state.
        saver = tf.train.Saver()
        path = os.path.join(model_dir, 'model.ckpt')
        saver.restore(regressor.get_session(), path)
        return regressor

    def train(self, X_train, y_train, regressor=False):
        """Train the regressor with the provided training data"""

        if regressor is False:
            # Init the regressor.
            regressor = self.get_regressor(X_train, y_train)

        # Fit the training set. y should be an array-like.
        regressor.fit(X_train, y_train[:, -1])

        # Returns the trained regressor.
        return regressor

    def regressor_exists(self):
        """Checks if there is a previously stored regressor"""

        regressor_dir = os.path.join(self.persistencedir,
                                      PERSIST_FILENAME)
        return os.path.isfile(regressor_dir)

    def train_dataset(self, filepath):
        """Train the model with the provided dataset"""

        [self.X, self.y] = self.get_labelled_samples(filepath)

        if len(np.unique(self.y)) < 1:
            # We need samples for training.
            result = dict()
            result['status'] = NOT_ENOUGH_DATA
            result['info'] = []
            result['errors'] = 'Training data needs to include ' 
                
            return result

        # Load the loaded model if it exists.
        if self.regressor_exists():
            classifier = self.load_regressor()
        else:
            # Not previously trained.
            regressor = False

        trained_regressor = self.train(self.X, self.y, regressor)

        self.store_regressor(trained_regressor)

        result = dict()
        result['status'] = OK
        result['info'] = []
        return result

    def predict_dataset(self, filepath):
        """Predict values for the provided dataset"""

        [sampleids, x] = self.get_unlabelled_samples(filepath)

        if self.regressor_exists() is False:
            result = dict()
            result['status'] = NO_DATASET
            result['info'] = ['Provided model have not been trained yet']
            return result

        regressor = self.load_regressor()

        # Prediction
        y_pred = regressor.predict(x)
        

        result = dict()
        result['status'] = OK
        result['info'] = []
        # First column sampleids, second the prediction 
        result['predictions'] = np.vstack((sampleids,
                                           y_pred,
                                          ).T.tolist()

        return result

""" This is a rough idea of what would be done to evaluate the regressor. This can be improved further when 
    regression model is finalised and implemented"""

    def evaluate_dataset(self, filepath, n_test_runs=100,
                         trained_model_dir=False):
        """Evaluate the model using the provided dataset"""

        [self.X, self.y] = self.get_labelled_samples(filepath)

        if trained_model_dir is not False:
            # Load the trained model in the provided path and evaluate it.
            trained_model_dir = os.path.join(trained_model_dir, 'regressor')
            regressor = self.load_regressor(trained_model_dir)
            self.rate_prediction(regressor, self.X, self.y)

        else:
            # Evaluate the model by training the ML algorithm multiple times.

            for _ in range(0, n_test_runs):

                # Split samples into training set and test set (80% - 20%)
                X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                                    self.y,
                                                                    test_size=0.2)

                regressor = self.train(X_train, y_train)
                y_pred=regressor.predict(x_test)
                
                self.rate_prediction(regressor, X_test, y_test)

        # Return results.
        result = self.get_evaluation_results()

        # Add the run id to identify it in the caller.
        result['runid'] = int(self.get_runid())

        logging.info("meansquare score: %.2f%%", result['meansquare'] )
        logging.info("r2 Score: %.2f%%", result['score'] * 100)
   

        return result

    def rate_prediction(self, regressor, X_test, y_test):
        """Rate a trained regressor with test data"""

        #predict 
        y_pred = regressor.predict(X_test)

        # Transform it to an array.
        y_test = y_test.T[0]

        # Calculate mean square and r2
        meansquare = mean_squared_error(y_test,y_pred)
        r2 = r2_score(y_test,y_pred)
        variancescore = explained_variance_score(y_test, y_pred)
  
        self.meansquare.append(meansquare)
        self.r2.append(r2)
        self.variancescore.append(variancescore)
        


   
    def get_evaluation_results(self, accepted_deviation):
        """Returns the evaluation results after all iterations"""

       
        result = dict()
        result['meansquare'] = np.mean(self.meansquare)
        result['r2'] = np.mean(self.r2)
        result['variancescore'] = np.mean(self.variancescore]                                  
       
        result['dir'] = self.logsdir
        result['status'] = OK
        result['info'] = []

        # Will be implemented once the model of regression is chosen and implemented
       
       """
       # If deviation is too high we may need more records to report if
        # this model is reliable or not.
       
        """
        result['info'].append('Launch TensorBoard from command line by ' +
                              'typing: tensorboard --logdir=\'' +
                              self.get_tensor_logdir() + '\'')

return result
