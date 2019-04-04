"""Import module"""

from __future__ import print_function
import sys
import json
import time

from moodlemlbackend.processor import estimator


def import():
    """Imports a trained classifier or regressor."""

    modelid = sys.argv[1]
    directory = sys.argv[2]
    type = sys.argv[4]
    if type=="classifier" :
      binary_classifier = estimator.Binary(modelid, directory)
      binary_classifier.import_classifier(sys.argv[3])

      # An exception will be thrown before if it can be imported.
      sys.exit(0)
    
    else:
      regressor = estimator.Regressor(modelid, directory)
      regressor.import_classifier(sys.argv[3])

      # An exception will be thrown before if it can be imported.
      sys.exit(0)
      

import()
