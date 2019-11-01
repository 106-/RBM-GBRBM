# -*- coding:utf-8 -*-

import numpy as np
import logging
from mltools import Parameter
from mltools import EpochCalc

def sigmoid(x):
    return np.piecewise(x, [x>0], [
        lambda x: 1 / (1+np.exp(-x)),
        lambda x: np.exp(x) / (1+np.exp(x))
    ])

class GBRBM_params(Parameter):
    def __init__(self, num_visible, num_hidden, initial_params=None):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.param_names = ["weight", "bias_v", "bias_h", "sigma"]
        
        if initial_params is None:
            uniform_range = np.sqrt( 6/(num_visible + num_hidden) )
            self.weight = np.random.uniform(-uniform_range, uniform_range, (num_visible, num_hidden))
            self.bias_v = np.random.rand(num_visible)
            self.bias_h = np.random.rand(num_hidden)
            self.sigma = np.random.randn(num_visible)
        elif isinstance(initial_params, dict):
            for p in self.param_names:
                setattr(self, p, initial_params[p])
        elif isinstance(initial_params, list):
            for p, i in zip(self.param_names, initial_params):
                setattr(self, p, i)
        else:
            raise TypeError("initial_params is unknown type :%s"%type(initial_params))

        params = {}
        for p in self.param_names:
            params[p] = getattr(self, p)

        super().__init__(params)

    def zeros(self):
        zero_params = {}
        for p in self.param_names:
            zero_params[p] = np.zeros(getattr(self, p).shape)
        return GBRBM_params(self.num_visible, self.num_hidden, initial_params=zero_params)

class GBRBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.params = GBRBM_params(num_visible, num_hidden)

    def _mean_visible(self, hidden_data):
        return sigmoid( self.params.bias_v + np.dot( hidden_data, self.params.weight.T ) )

    # P(h|v)
    def _prob_h_v(self, visible_data):
        return sigmoid( self.params.bias_h + np.dot( visible_data, self.params.weight ) )

    def _data_mean(self, visible_data, hidden_data):
        data_length = len(visible_data)
        visible_mean = np.average(visible_data, axis=0) 
        hidden_mean = np.average(hidden_data, axis=0) 
        weight_mean = np.dot(visible_data.T, hidden_data) / data_length
        sigma_mean = np.average( visible_data / (np.sqrt(2) * self.params.sigma ** 2), axis=0)
        return GBRBM_params(self.num_visible, self.num_hidden, initial_params=[weight_mean, visible_mean, hidden_mean, sigma_mean])

    def _contrastive_divergence(self, data_prob):
        data_length = len(data_prob)
        hidden_sampling = data_prob > np.random.rand(data_length, self.num_hidden)
        visible_sampling = self._mean_visible(hidden_sampling) + self.params.sigma * np.random.randn(data_length, self.num_visible)
        return self._data_mean(visible_sampling, data_prob)

    def _reconstruct(self, data, gauss=True):
        data_length = len(data)
        hidden_sampling = self._prob_h_v(data) > np.random.rand(data_length, self.num_hidden)
        if gauss: 
            visible_sampling = self._mean_visible(hidden_sampling) + self.params.sigma * np.random.randn(data_length, self.num_visible)
        else:
            visible_sampling = self._mean_visible(hidden_sampling)
        return visible_sampling

    def train(self, data, train_epoch, optimizer, minibatch_size=100, test_interval=1.0):
        ec = EpochCalc(train_epoch, len(data), minibatch_size)

        def per_test_epoch(update_time):
            logging.info("[ {} / {} ]( {} / {} )".format(ec.update_to_epoch(update_time, force_integer=False), ec.train_epoch, update_time, ec.train_update))

        per_test_epoch(0)
        for i in range(1, ec.train_update+1):
            data_prob = self._prob_h_v(data)
            data_exp = self._data_mean(data, data_prob)
            model_exp = self._contrastive_divergence(data_prob)
            diff = optimizer.update( model_exp - model_exp )
            self.params += diff
            if i % ec.epoch_to_update(test_interval) == 0:
                per_test_epoch(i)
    
    def recall(self, data, forgotten, gauss=True):
        data[forgotten] = 0
        if data.ndim == 1:
            data = data[np.newaxis, :]
        elif data.ndim > 2:
            raise ValueError("recall method accepts only 2-dimensional data.")
        recovered = self._reconstruct(data[np.newaxis, :], gauss)
        return recovered
