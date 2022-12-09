from __future__ import print_function

import argparse
import logging
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
import os
import numpy as np
import json
import time
import pandas

from spam_model import train, save

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--log-interval', type=int, default=200)

    # an alternative way to load hyperparameters via SM_HPS environment variable.
    parser.add_argument('--sm-hps', type=json.loads, default=os.environ['SM_HPS'])

    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_VAL'])

    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))

    return parser.parse_args()


if __name__ =='__main__':
    args = parse_args()
    logger.info(f'Args: \n\n{args}')
    
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    num_gpus = int(os.environ["SM_NUM_GPUS"])

    net = train(
        args.current_host,
        args.hosts,
        num_cpus,
        num_gpus,
        args.train,
        args.val,
        args.model_dir,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.momentum,
        args.log_interval
    )

    save(net, args.model_dir)
