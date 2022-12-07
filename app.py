#!/usr/bin/env python3
import os

import aws_cdk as cdk

from spam_detection.spam_detection_stack import SpamDetectionStack
from sagemaker_model_docker.sagemaker_training_stack import SageMakerWorkflowStack

app = cdk.App()
env = env=cdk.Environment(account='756059218166', region='us-east-1')
spam_stack = SpamDetectionStack(app, "SpamDetectionStack", env=env)
SageMakerWorkflowStack(app, "SageMakerWorkflowStack", env=env, bucket=spam_stack.spam_detection_bucket, object_prefix="sms-spam-mxnet-workflow")
app.synth()
