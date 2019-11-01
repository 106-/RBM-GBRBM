#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from GBRBM import GBRBM
from sklearn.datasets import load_iris

def main():
    iris = load_iris()
    gbrbm = GBRBM(4, 3)
    gbrbm.train(iris.data)

if __name__=='__main__':
    main()