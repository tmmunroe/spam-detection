from __future__ import print_function

import argparse
import logging
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
from sagemaker_mxnet_container.training_utils import save
import os
import numpy as np
import json
import time
import pandas

def model_fn(model_dir):
    net = gluon.nn.SymbolBlock(
        outputs=mx.sym.load('{}/model.json'.format(model_dir)),
        inputs=mx.sym.var('data'))
    
    net.load_params('{}/model.params'.format(model_dir), ctx=mx.cpu())

    return net

def transform_fn(net, data, input_content_type, output_content_type):
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
        return response_body, output_content_type


def get_train_data(data_path, batch_size):
    print('Train data path: ' + data_path)
    df = pandas.read_csv('{}/train.gz'.format(data_path))
    features = df[df.columns[1:]].values.astype(dtype=np.float32)
    labels = df[df.columns[0]].values.reshape((-1, 1)).astype(dtype=np.float32)
    
    return gluon.data.DataLoader(gluon.data.ArrayDataset(features, labels), batch_size=batch_size, shuffle=True)

def get_val_data(data_path, batch_size):
    print('Validation data path: ' + data_path)
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

    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': learning_rate, 'momentum': momentum},
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
                print('[Epoch %d Batch %d] Training: %s=%f, %f samples/s' %
                      (epoch, i, name, acc, batch_size / (time.time() - btic)))

            btic = time.time()

        name, acc = metric.get()
        print('[Epoch %d] Training: %s=%f' % (epoch, name, acc))

        name, val_acc = test(ctx, net, val_data)
        print('[Epoch %d] Validation: %s=%f' % (epoch, name, val_acc))

    return net


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
