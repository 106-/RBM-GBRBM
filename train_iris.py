#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import logging
import argparse
from GBRBM import GBRBM
from sklearn.datasets import load_iris
from mltools.optimizer import Adamax

parser = argparse.ArgumentParser("DBM learning script.", add_help=False)
parser.add_argument("learning_epoch", action="store", type=int, help="numbers of epochs.")
parser.add_argument("-l", "--log_level", action="store", type=str, default="INFO", help="learning log output level.")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : [%(levelname)s] %(message)s', level=getattr(logging, args.log_level))
np.seterr(over="raise", invalid="raise")

def main():
    iris = load_iris()
    gbrbm = GBRBM(4, 5)
    adamax = Adamax()
    gbrbm.train(iris.data, args.learning_epoch, adamax, minibatch_size=150)

if __name__=='__main__':
    main()