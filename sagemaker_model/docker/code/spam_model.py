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
import shutil

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    logger.info(f"Preparing model from {model_dir}")
    net = gluon.nn.SymbolBlock(
        outputs=mx.sym.load('{}/model.json'.format(model_dir)),
        inputs=mx.sym.var('data'))

    net.load_parameters(f'{model_dir}/model.params', ctx=mx.cpu())
    # net.load_params('{}/model.params'.format(model_dir), ctx=mx.cpu())

    logger.info("Model prepared!")
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    logger.info(f"Received request:\nData:{data}\nInputContentType:{input_content_type}\nOutputContentType:{output_content_type}")
    try:
        parsed = json.loads(data)
        nda = mx.nd.array(parsed)
        
        output = net(nda)
        sigmoid_output = output.sigmoid()
        prediction = mx.nd.abs(mx.nd.ceil(sigmoid_output - 0.5))
        
        output_obj = {}
        output_obj['predicted_label'] = prediction.asnumpy().tolist()
        output_obj['predicted_probability'] = sigmoid_output.asnumpy().tolist()

        response_body = json.dumps(output_obj)
        return response_body, output_content_type
    except Exception as ex:
        response_body = '{error: }' + str(ex)
        logger.info(f"Exception thrown: {response_body}")
        return response_body, output_content_type



# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

def save(net, model_dir, include_inference_code=True):
    y = net(mx.sym.var('data'))
    y.save('{}/model.json'.format(model_dir))
    net.collect_params().save('{}/model.params'.format(model_dir))
    
    if include_inference_code:
        src = os.path.dirname(os.path.abspath(__file__))
        dest =  f'{model_dir}/code'
        logger.info(f'src: {src}')
        logger.info(f'dest: {dest}')
        shutil.copytree(src, dest)

def get_train_data(data_path, batch_size):
    logger.info('Train data path: ' + data_path)
    df = pandas.read_csv('{}/train.gz'.format(data_path))
    features = df[df.columns[1:]].values.astype(dtype=np.float32)
    labels = df[df.columns[0]].values.reshape((-1, 1)).astype(dtype=np.float32)
    
    return gluon.data.DataLoader(gluon.data.ArrayDataset(features, labels), batch_size=batch_size, shuffle=True)

def get_val_data(data_path, batch_size):
    logger.info('Validation data path: ' + data_path)
    df = pandas.read_csv('{}/val.gz'.format(data_path))
    features = df[df.columns[1:]].values.astype(dtype=np.float32)
    labels = df[df.columns[0]].values.reshape((-1, 1)).astype(dtype=np.float32)
    
    return gluon.data.DataLoader(gluon.data.ArrayDataset(features, labels), batch_size=batch_size, shuffle=False)

def test(ctx, net, val_data):
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        
        output = net(data)
        sigmoid_output = output.sigmoid() 
        prediction = mx.nd.abs(mx.nd.ceil(sigmoid_output - 0.5))
        
        metric.update([label], [prediction])
    return metric.get()

def define_network():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(64, activation="relu"))
        net.add(nn.Dense(1))
    return net

def train(
    current_host,
    hosts,
    num_cpus,
    num_gpus,
    training_dir,
    val_dir,
    model_dir,
    batch_size,
    epochs,
    learning_rate,
    momentum,
    log_interval):
    # SageMaker passes num_cpus, num_gpus and other args we can use to tailor training to
    # the current container environment, but here we just use simple cpu context.
    ctx = mx.cpu()

    # retrieve the hyperparameters and apply some defaults in case they are not provided.
    train_data = get_train_data(training_dir, batch_size)
    val_data = get_val_data(val_dir, batch_size)

    # define the network
    net = define_network()

    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Normal(sigma=1.), ctx=ctx)
    
    # Trainer is for updating parameters with gradient.
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'

    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate},
                            kvstore=kvstore)
    
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    for epoch in range(epochs):
        
        # reset data iterator and metric at begining of epoch.
        metric.reset()
        btic = time.time()
        for i, (data, label) in enumerate(train_data):
            # Copy data to ctx if necessary
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            
            # Start recording computation graph with record() section.
            # Recorded graphs can then be differentiated with backward.
            with autograd.record():
                output = net(data)
                L = loss(output, label)
            L.backward()

            # take a gradient step with batch_size equal to data.shape[0]
            trainer.step(data.shape[0])

            # update metric at last.
            sigmoid_output = output.sigmoid() 
            prediction = mx.nd.abs(mx.nd.ceil(sigmoid_output - 0.5))
            metric.update([label], [prediction])

            if i % log_interval == 0 and i > 0:
                name, acc = metric.get()
                logger.info('[Epoch %d Batch %d] Training: %s=%f, %f samples/s' %
                      (epoch, i, name, acc, batch_size / (time.time() - btic)))

            btic = time.time()

        name, acc = metric.get()
        logger.info(f'[Epoch {epoch}] Training: {name}={acc}')

        name, val_acc = test(ctx, net, val_data)
        logger.info(f'[Epoch {epoch}] Validation: {name}={val_acc}')

    return net
