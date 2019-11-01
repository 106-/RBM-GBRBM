#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from GBRBM import GBRBM
from sklearn.datasets import load_iris
from mltools.optimizer import Adamax

def main():
    iris = load_iris()
    gbrbm = GBRBM(4, 3)
    adamax = Adamax()
    gbrbm.train(iris.data, adamax)

if __name__=='__main__':
    main()