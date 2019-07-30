# -*- coding:utf-8 -*-

import numpy as np

def sigmoid(x):
    return np.piecewise(x, [x>0], [
        lambda x: 1 / (1+np.exp(-x)),
        lambda x: np.exp(x) / (1+np.exp(x))
    ])

class GBRBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        uniform_range = np.sqrt( 6/(num_visible + num_hidden) )
        self.weight = np.random.uniform(-uniform_range, uniform_range, (num_visible, num_hidden))
        self.bias_v = np.random.rand(num_visible)
        self.bias_h = np.random.rand(num_hidden)
        self.sigma = np.random.randn(num_visible)

    def _mean_visible(self, hidden_data):
        return sigmoid( self.bias_v + np.dot( hidden_data, self.weight.T ) )

    # P(h|v)
    def _prob_h_v(self, visible_data):
        return sigmoid( self.bias_h + np.dot( visible_data, self.weight ) )

    def _data_mean(self, visible_data, hidden_data):
        data_length = len(visible_data)
        visible_mean = np.average(visible_data, axis=0) 
        hidden_mean = np.average(hidden_data, axis=0) 
        weight_mean = np.dot(visible_data.T, hidden_data) / data_length
        sigma_mean = np.average( visible_data / (np.sqrt(2) * self.sigma ** 2), axis=0)
        return (visible_mean, hidden_mean, weight_mean, sigma_mean)

    def _contrastive_divergence(self, data_prob):
        data_length = len(data_prob)
        hidden_sampling = data_prob > np.random.rand(data_length, self.num_hidden)
        visible_sampling = self._mean_visible(hidden_sampling) + self.sigma * np.random.randn(data_length, self.num_visible)
        return self._data_mean(visible_sampling, data_prob)

    def _reconstruct(self, data):
        data_length = len(data)
        hidden_sampling = self._prob_h_v(data) > np.random.rand(data_length, self.num_hidden)
        # visible_sampling = self._mean_visible(hidden_sampling) + self.sigma * np.random.randn(data_length, self.num_visible)
        visible_sampling = self._mean_visible(hidden_sampling)
        return visible_sampling

    def train(self, data, learning_rate=0.01, learning_time=1000):
        for i in range(learning_time):
            data_prob = self._prob_h_v(data)
            data_visible, data_hidden, data_weight, data_sigma = self._data_mean(data, data_prob)
            model_visible, model_hidden, model_weight, model_sigma = self._contrastive_divergence(data_prob)

            grad_v = data_visible - model_visible 
            grad_h = data_hidden - model_hidden
            grad_w = data_weight - model_weight
            grad_sigma = data_sigma - model_sigma

            self.weight += grad_w * learning_rate
            self.bias_v += grad_v * learning_rate
            self.bias_h += grad_h * learning_rate
            self.sigma  += grad_sigma * learning_rate
    
    def recall(self, data, forgotten, return_broken_only=False):
        data[forgotten] = 0
        recovered = self._reconstruct(data[np.newaxis, :])
        if return_broken_only:
            return recovered[forgotten]
        else:
            return recovered
