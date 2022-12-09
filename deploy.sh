cp sagemaker_model/docker/code/sms_spam_classifier_utilities.py spam_detection/lambda/sms_spam_classifier_utils.py
cdk deploy --all --parameters SpamDetectionStack:ModelEndpoint=spam-mxnet-classifier-endpoint
