import argparse
import os
import json

import pandas as pd
import numpy as np

from sms_spam_classifier_utilities import one_hot_encode, vectorize_sequences, default_vocabulary_length

raw_data_path = "/opt/ml/processing/input/SMSSpamCollection"

base_output_dir = "/opt/ml/processing"
train_data_gz_path = "{}/train/train.gz".format(base_output_dir)
val_data_gz_path = "{}/val/val.gz".format(base_output_dir)


if __name__ =='__main__':
    df = pd.read_csv(raw_data_path, sep='\t', header=None)
    df[df.columns[0]] = df[df.columns[0]].map({'ham': 0, 'spam': 1})

    targets = df[df.columns[0]].values
    messages = df[df.columns[1]].values

    # one hot encoding for each SMS message
    # encoded_messages = transform(messages, default_vocabulary_length)
    
    one_hot_data = one_hot_encode(messages, default_vocabulary_length)
    encoded_messages = vectorize_sequences(one_hot_data, default_vocabulary_length)
    df2 = pd.DataFrame(encoded_messages)
    df2.insert(0, 'spam', targets)

    # Split into training and validation sets (80%/20% split)
    split_index = int(np.ceil(df.shape[0] * 0.8))
    train_set = df2[:split_index]
    val_set = df2[split_index:]

    train_set.to_csv(train_data_gz_path, header=False, index=False, compression='gzip')
    val_set.to_csv(val_data_gz_path, header=False, index=False, compression='gzip')
