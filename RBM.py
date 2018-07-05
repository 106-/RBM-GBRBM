#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import random

class RBM:

    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        # この0.5は適当. おそらくランダムで決める必要あり
        self.bias_visible = [0.5 for i in range(num_visible)]
        self.bias_hidden = [0.5 for i in range(num_hidden)]
        self.weight = [[0.5 for i in range(num_hidden)] for n in range(num_visible)]

    def _sigmoid(self, x):
        return 1/(1+math.exp(-x))
    
    def _calc_lambda(self, nodes, weights, bias):
        sum = 0
        for n,w in zip(nodes, weights):
            sum += n*w
        return sum + bias

    def _node_sampling(self, nodes, weights, bias):
        firing_prob = self._sigmoid(self._calc_lambda(nodes, weights, bias))
        uniform_rnd = random.uniform(0, 1)
        if firing_prob >= uniform_rnd:
            return 1
        else:
            return 0

    def _cd_gibbs_sampling(self, data):
        visible = data
        hidden = [None for i in range(self.num_hidden)]

        # sample hidden layer
        for i,h in enumerate(hidden):
            bias = self.bias_hidden[i]
            weights = list(map(lambda x: x[i], self.weight))
            hidden[i] = self._node_sampling(visible, weights, bias)

        # sample visible layer
        for i,v in enumerate(visible):
            bias = self.bias_visible[i]
            weights = self.weight[i]
            visible[i] = self._node_sampling(hidden, weights, bias)

        return [visible, hidden]

    def _calc_mean(self, layers):
        means = [None for i in range(len(layers[0]))]
        for i,m in enumerate(means):
            values = list(map(lambda x:x[i], layers))
            means[i] = float(sum(values)) / float(len(layers))
        return means

    def _calc_mean_weight(self, visibles, hiddens):
        weight_sum = [[0 for i in range(self.num_hidden)] for i in range(self.num_visible)]
        sample_num = float(len(visibles))
        for v_layer,h_layer in zip(visibles, hiddens):
            for i,v in enumerate(v_layer):
                for n,h in enumerate(h_layer):
                    weight_sum[i][n] += v*h

        weight_mean = [[None for i in range(self.num_hidden)] for i in range(self.num_visible)]
        for i,v in enumerate(weight_sum):
            for n,h in enumerate(v):
                weight_mean[i][n] = weight_sum[i][n] / sample_num
                
        return weight_mean

    def _differential_b(self, visible_data, visible_sample_mean):
        visible_data_sum = [0 for i in range(self.num_visible)]
        visible_data_num = len(visible_data)

        for i,v in enumerate(visible_data_sum):
            for data_v_layer in visible_data:
                visible_data_sum[i] += data_v_layer[i]
        
        diff_b = []
        for i in range(self.num_visible):
            diff_b.append(visible_data_sum[i] - visible_data_num * visible_sample_mean[i])
        
        return diff_b

    def _differential_c(self, visible_data, hidden_sample_mean):
        lambda_sum = [0 for j in range(self.num_hidden)]
        visible_data_num = len(visible_data[0])

        for j,h in enumerate(lambda_sum):
            for data_v_layer in visible_data:
                weights = list(map(lambda x: x[j], self.weight))
                lambda_sum[j] += self._sigmoid(self._calc_lambda(data_v_layer, weights, self.bias_hidden[j]))
        
        diff_c = []
        for j in range(self.num_hidden):
            diff_c.append(lambda_sum[j] - visible_data_num * hidden_sample_mean[j])
        
        return diff_c

    def _differential_w(self, visible_data, weight_sample_mean):
        lambda_sum = [[0 for j in range(self.num_hidden)] for i in range(self.num_visible)]
        visible_data_num = len(visible_data[0])

        for i,v in enumerate(lambda_sum):
            for j,h in enumerate(v):
                for data_v_layer in visible_data:
                    weights = list(map(lambda x: x[j], self.weight))
                    lambda_sum[i][j] += data_v_layer[i] * self._sigmoid(self._calc_lambda(data_v_layer, weights, self.bias_hidden[j]))
        
        diff_w = []
        for i in range(self.num_visible):
            diff_wi = []
            for j in range(self.num_hidden):
                diff_wi.append(lambda_sum[i][j] - visible_data_num * weight_sample_mean[i][j])
            diff_w.append(diff_wi)

        return diff_w
                
    def train(self, visible_data, train_times, learning_rate=0.01, sampling_num=1000):
        
        if not (self.num_visible == len(visible_data[0])):
            raise TypeError

        for train_time in range(train_times):
            samplings = []
            for i in range(sampling_num):
                samplings.append(self._cd_gibbs_sampling(visible_data[0]))

            visibles = list(map(lambda x: x[0], samplings))
            hiddens = list(map(lambda x: x[1], samplings))
            visible_sample_mean = self._calc_mean(visibles)
            hidden_sample_mean = self._calc_mean(hiddens)
            weight_sample_mean = self._calc_mean_weight(visibles, hiddens)

            diff_b = self._differential_b(visible_data, visible_sample_mean)
            diff_c = self._differential_c(visible_data, hidden_sample_mean)
            diff_w = self._differential_w(visible_data, weight_sample_mean)

            for i,b in enumerate(self.bias_visible):
                self.bias_visible[i] += learning_rate * diff_b[i]

            for j,c in enumerate(self.bias_hidden):
                self.bias_hidden[j] += learning_rate * diff_c[j]

            for i,b in enumerate(self.weight):
                for j,c in enumerate(b):
                    self.weight[i][j] += learning_rate * diff_w[i][j]

    def print_parameters(self):
        print("weights:")
        print(self.weight)
        print("visible layer bias:")
        print(self.bias_visible)
        print("hidden layer bias:")
        print(self.bias_hidden)

def main():
    datas = []
    for i in range(50):
        datas.append(random.choices([0,1], 728))
    print(datas)

    rbm = RBM(728, 500)
    rbm.train([[1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 0]], 1000)
    rbm.print_parameters()

if __name__=="__main__":
    main()