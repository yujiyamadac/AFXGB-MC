import numpy as np

import xgboost as xgb
import random
import os

from skmultiflow.core.base import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import get_dimensions
from skmultiflow.drift_detection import ADWIN

xgb.set_config(verbosity=0)

class AdaptiveMulticlass(BaseSKMObject, ClassifierMixin):

    def __init__(self,
                 learning_rate=0.3,
                 max_depth=6,
                 max_window_size=1000,
                 min_window_size=None,
                 small_window_size=0,
                 max_buffer=5,
                 pre_train=2,
                 num_class=2,
                 detect_drift=True,
                 use_updater=True,
                 trees_per_train=1,
                 percent_update_trees=1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self._first_run = True
        self._booster = None
        self._temp_booster = None
        self._drift_detector = None
        self._X_buffer = np.array([])
        self._y_buffer = np.array([])

        self._max_buffer = max_buffer
        self._pre_train = pre_train
        self._X_small_buffer = np.array([])
        self._y_small_buffer = np.array([])
        self._samples_seen = 0
        self._model_idx = 0
        self._small_window_size = small_window_size
        self._count_buffer = 0
        self._main_model = "model"
        self._temp_model = "temp"

        self.num_class = num_class
        self.detect_drift = detect_drift
        self.use_updater = use_updater
        self.trees_per_train = trees_per_train
        self.percent_update_trees = percent_update_trees

        # calculo do inside_pre_train
        self._inside_pre_train = self._max_buffer - self._pre_train

        self._configure()

    def _configure(self):
        self._reset_window_size()
        self._init_margin = 0.0
        self._boosting_params = {
            "objective": "multi:softmax",
            "eta": self.learning_rate,
            "eval_metric": "mlogloss",
            "max_depth": self.max_depth,
            "num_class": self.num_class
        }
        self._boosting_params_update = self._boosting_params.copy()
        self._boosting_params_update["process_type"] = "update"
        self._boosting_params_update["updater"] = "refresh"
        if self.detect_drift:
            self._drift_detector = ADWIN()

    def reset(self):
        self._first_run = True
        self._configure()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        for i in range(X.shape[0]):
            self._partial_fit(np.array([X[i, :]]), np.array([y[i]]))
        return self

    def _change_small_window(self, npArrX, npArrY):
        if npArrX.shape[0] < self._small_window_size:
            sizeToRemove = 0
            nextSize = self._X_small_buffer.shape[0] + npArrX.shape[0]
            if nextSize > self._small_window_size:
                sizeToRemove = nextSize - self._small_window_size
            #deleta os dados velhos
            delete_idx = [i for i in range(sizeToRemove)]

            if len(delete_idx) > 0:
                self._X_small_buffer = np.delete(self._X_small_buffer, delete_idx, axis=0)
                self._y_small_buffer = np.delete(self._y_small_buffer, delete_idx, axis=0)
            
            self._X_small_buffer = np.concatenate((self._X_small_buffer, npArrX))
            self._y_small_buffer = np.concatenate((self._y_small_buffer, npArrY))
        else:
            self._X_small_buffer = npArrX[0:self._small_window_size]
            self._y_small_buffer = npArrY[0:self._small_window_size]


    def _partial_fit(self, X, y):
        if self._first_run:
            self._X_buffer = np.array([]).reshape(0, get_dimensions(X)[1])
            self._y_buffer = np.array([])
            self._X_small_buffer = np.array([]).reshape(0, get_dimensions(X)[1])
            self._y_small_buffer = np.array([])
            self._first_run = False
        self._X_buffer = np.concatenate((self._X_buffer, X))
        self._y_buffer = np.concatenate((self._y_buffer, y))

        while self._X_buffer.shape[0] >= self.window_size:
            self._count_buffer = self._count_buffer + 1
            # npArrX, npArrY = self._unlabeled_fit()
            npArrX = self._X_buffer
            npArrY = self._y_buffer
            if npArrX.shape[0] > 0:
                self._train_on_mini_batch(X=npArrX, y=npArrY)
                                    
            delete_idx = [i for i in range(self.window_size)]
            self._X_buffer = np.delete(self._X_buffer, delete_idx, axis=0)
            self._y_buffer = np.delete(self._y_buffer, delete_idx, axis=0)

            # Check window size and adjust it if necessary
            self._adjust_window_size()
        
        # Support for concept drift
        if self.detect_drift:
            correctly_classifies = self.predict(X) == y
            # Check for warning
            self._drift_detector.add_element(int(not correctly_classifies))
            # Check if there was a change
            if self._drift_detector.detected_change():
                # Reset window size
                self._reset_window_size()
                # if self.update_strategy == self._REPLACE_STRATEGY:
                self._model_idx = 0

    def _adjust_window_size(self):
        if self._dynamic_window_size < self.max_window_size:
            self._dynamic_window_size *= 2
            if self._dynamic_window_size > self.max_window_size:
                self.window_size = self.max_window_size
            else:
                self.window_size = self._dynamic_window_size

    def _reset_window_size(self):
        if self.min_window_size:
            self._dynamic_window_size = self.min_window_size
        else:
            self._dynamic_window_size = self.max_window_size
        self.window_size = self._dynamic_window_size

    def _train_on_mini_batch(self, X, y):
        # se contador é igual ou maior que a faixa de pre train, o temp é treinado
        if self._count_buffer >= self._inside_pre_train:
            temp_booster = self._train_booster(X, y, self._temp_model, self._temp_booster)
            self._temp_booster = temp_booster

        if self._count_buffer >= self._max_buffer:
            booster = self._temp_booster
            self._temp_booster = None
            self._count_buffer = 0
            self._temp_model, self._main_model = self._main_model, self._temp_model
            # reseta a janela quando troca de MAIN para TEMP
            self._reset_window_size()

        else:
            # modelo MAIN não precisa ser treinado caso esteja no momento da troca
            booster = self._train_booster(X, y, self._main_model, self._booster)

        # Update ensemble
        self._booster = booster

    def _train_booster(self, X: np.ndarray, y: np.ndarray, fileName, currentBooster):
        d_mini_batch_train = xgb.DMatrix(X, y.astype(int))

        if currentBooster:
            new_trees = 0
            booster = currentBooster
            if self.use_updater:
                num_boosted_rounds = currentBooster.num_boosted_rounds()
                booster = xgb.train(
                    params=self._boosting_params_update,
                    dtrain=d_mini_batch_train,
                    num_boost_round=int(num_boosted_rounds * self.percent_update_trees),
                    xgb_model=booster,
                )
                new_trees = num_boosted_rounds - booster.num_boosted_rounds()
            booster = xgb.train(params=self._boosting_params,
                                dtrain=d_mini_batch_train,
                                num_boost_round=self.trees_per_train + new_trees,
                                xgb_model=booster,)
            booster.save_model(fileName)
        else:
            booster = xgb.train(params=self._boosting_params,
                                dtrain=d_mini_batch_train,
                                num_boost_round=self.trees_per_train,
                                verbose_eval=False)
            booster.save_model(fileName)
        return booster

    def predict(self, X):
        if self._booster:
            predicted = self._booster.inplace_predict(X)
            return predicted
        # Ensemble is empty, return default values (0)
        return np.zeros(get_dimensions(X)[0])

    def predict_proba(self, X):
        """
        Not implemented for this method.
        """
        raise NotImplementedError(
            "predict_proba is not implemented for this method.")
