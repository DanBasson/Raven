import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from static.constants import *
from src.feature_engineering import FeatureEngineering
from src.preprocessor import Preprocessor


class ClusterModel:

    def __init__(self, X, model_hyperparameters=None, elbow_max_clusters=10,
                 model=KMeans):
        if model_hyperparameters is None:
            model_hyperparameters = {}
        self.elbow_distortions = []
        self.X = X
        self.elbow_iter_range = range(1, elbow_max_clusters)
        self.model = model
        self.model_hyperparameters = model_hyperparameters

    def fit_model(self):
        self.fitted_model = self.model(n_clusters=self.model_hyperparameters['n_clusters'],
                                       random_state=self.model_hyperparameters['random_state'])
        self.fitted_model.fit(self.X)

    def elbow_method(self):
        for k in self.elbow_iter_range:
            model_ = self.model(n_clusters=k, random_state=1)
            model_.fit(self.X)
            self.elbow_distortions.append(model_.inertia_)
