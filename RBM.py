# -*- coding:utf-8 -*-

import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        uniform_range = np.sqrt( 6/(num_visible + num_hidden) )
        self.weight = np.random.uniform(-uniform_range, uniform_range, (num_visible, num_hidden))
        self.bias_v = np.random.rand(num_visible)
        self.bias_h = np.random.rand(num_hidden)

    # P(v|h)
    def _prob_v_h(self, hidden_data):
        return sigmoid( self.bias_v + np.dot( hidden_data, self.weight.T ) )

    # P(h|v)
    def _prob_h_v(self, visible_data):
        return sigmoid( self.bias_h + np.dot( visible_data, self.weight ) )

    def _data_mean(self, visible_data, hidden_data):
        data_length = len(visible_data)
        visible_mean = np.average(visible_data, axis=0) 
        hidden_mean = np.average(hidden_data, axis=0) 
        weight_mean = np.dot(visible_data.T, hidden_data) / data_length
        return (visible_mean, hidden_mean, weight_mean)

    def _contrastive_divergence(self, data_sig):
        data_length = len(data_sig)
        hidden_sampling = data_sig > np.random.rand(data_length, self.num_hidden)
        visible_sampling = self._prob_v_h(hidden_sampling) > np.random.rand(data_length, self.num_visible)
        return self._data_mean(visible_sampling, hidden_sampling)

    def _gibss(self, data_num, burn_in=1000):
        visible_data = np.random.rand(data_num, self.num_visible) > 0.5
        hidden_data = None
        for i in range(burn_in):
            hidden_data = self._prob_h_v(visible_data) > np.random.rand(data_num, self.num_hidden)
            visible_data = self._prob_v_h(hidden_data) > np.random.rand(data_num, self.num_visible)
        return self._data_mean(visible_data, hidden_data)

    def train(self, data, learning_rate=0.01, learning_time=1000):
        for i in range(learning_time):
            data_sig = self._prob_h_v(data)
            data_visible, data_hidden, data_weight = self._data_mean(data, data_sig)
            model_visible, model_hidden, model_weight = self._contrastive_divergence(data_sig)

            grad_v = data_visible - model_visible
            grad_h = data_hidden - model_hidden
            grad_w = data_weight - model_weight

            self.weight += grad_w * learning_rate
            self.bias_v += grad_v * learning_rate
            self.bias_h += grad_h * learning_rate
