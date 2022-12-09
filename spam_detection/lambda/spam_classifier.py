import boto3
from botocore.exceptions import ClientError

import codecs
from collections import namedtuple
from email import policy
from email.parser import BytesParser, Parser
import email.message
import json
import os
from sms_spam_classifier_utils import (
    one_hot_encode,
    vectorize_sequences,
    default_vocabulary_length
)

# resource, environment variables, constances
sender = os.getenv("SENDER_EMAIL")
region = os.getenv("REGION")
model_endpoint = os.getenv("ENDPOINT")
vocabulary_length = default_vocabulary_length

charset = "utf-8"
sagemaker_encoding = 'utf-8'

ses_client = boto3.client("ses", region_name=region)
sagemaker_client = boto3.client("sagemaker-runtime")
s3_client = boto3.resource("s3")


# helper classes
Prediction = namedtuple("Prediction", ("label", "confidence"))
class SimpleEmailMessage:
    def __init__(self, emailMessage:email.message.EmailMessage):
        if not isinstance(emailMessage, email.message.EmailMessage):
            raise ValueError(f'Expected emailMessage to be of type email.message.EmailMessage, but received type {type(emailMessage)}')
        self._message = emailMessage
    
    def __str__(self):
        return '\n'.join([
            f'From: {self.From}',
            f'To: {self.To}',
            f'Subject: {self.Subject}',
            f'Date: {self.Date}',
            f'Body: {self.Body}',
        ])

    @property
    def From(self):
        return self._message['From']
    
    @property
    def To(self):
        return self._message['To']

    @property
    def Date(self):
        return self._message['Date']

    @property
    def Subject(self):
        return self._message['Subject']

    @property
    def Body(self, removeNewLines=True):
        return self._message.get_body(preferencelist=('plain',)).get_content()


def send_response(predicted_email:SimpleEmailMessage, prediction:Prediction):
    response_lines = "\n\n".join([
            f"We received your email sent at {predicted_email.Date} with the subject '{predicted_email.Subject}'.",
            f"Here is a 240 character sample of the email body: \n\n{predicted_email.Body[:240]}",
            f"The email was categorized as {prediction.label} with {prediction.confidence * 100:.2f}% confidence."
        ])

    try:
        response = ses_client.send_email(
            Source=sender,
            Destination={
                "ToAddresses": [predicted_email.From],
            },
            Message={
                "Body": {
                    "Text": {
                        "Charset": charset,
                        "Data": response_lines,
                    },
                },
                "Subject": {
                    "Charset": charset,
                    "Data": "Email Classification Result",
                },
            }
        )	
    except ClientError as e:
        print(e.response["Error"]["Message"])
    else:
        print("Email sent! Message ID:"),
        print(response["MessageId"])


def predict(target_email: SimpleEmailMessage) -> Prediction:
    predictions = []
    print('Submitting to endpoint: ', model_endpoint)
    print('Email: ', target_email)

    print(f'Predicting body: {target_email.Body}')
    
    one_hot_data = one_hot_encode([target_email.Body], vocabulary_length)
    encoded_message = vectorize_sequences(one_hot_data, vocabulary_length)
    packed_message = json.dumps(encoded_message.tolist()).encode(sagemaker_encoding)

    response = sagemaker_client.invoke_endpoint(
        EndpointName=model_endpoint,
        Body=packed_message,
        ContentType='text/csv'
    )
    print('Received response: ')
    print(response)

    body = json.load(codecs.getreader('utf-8')(response['Body']))
    print(body)

    for labels, probabilities in zip(body['predicted_label'], body['predicted_probability']):
        if len(labels) != 1 or len(probabilities) != 1:
            raise ValueError('Expected 1 label and 1 probability')
        label = int(labels[0])
        probability = probabilities[0]

        category = 'Spam' if label == 1 else 'Ham'
        confidence = probability if label == 1 else (1-probability)
        predictions.append(Prediction(category, confidence))
    
    if len(predictions) != 1:
        raise ValueError(f'Expected 1 prediction but got {predictions}')
        
    return predictions[0]


def extract_email(event) -> SimpleEmailMessage:
    records = event['Records']
    if len(records) > 1:
        raise ValueError(f'Can only handle a single record. Received {len(records)}')
    
    s3_record = records[0]['s3']
    bucket_name = s3_record['bucket']['name']
    object_key = s3_record['object']['key']

    print(f'Extracting s3 object {bucket_name}::{object_key}')
    s3_object = s3_client.Object(bucket_name, object_key)
    s3_response = s3_object.get()
    email_blob = s3_response['Body'].read()

    message = BytesParser(policy=policy.default).parsebytes(email_blob)
    print('Message: ', message)

    return SimpleEmailMessage(message)


def classify_email_event(event):
    target_email = extract_email(event)
    prediction = predict(target_email)
    send_response(target_email, prediction)


def lambda_handler(event, context):
    print("Event: ")
    print(event)
    classify_email_event(event)
