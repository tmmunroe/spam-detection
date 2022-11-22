#!/usr/bin/env python3
import os

import aws_cdk as cdk

from spam_detection.spam_detection_stack import SpamDetectionStack


app = cdk.App()
SpamDetectionStack(app, "SpamDetectionStack",
    env=cdk.Environment(account='756059218166', region='us-east-1'))

app.synth()
