import boto3
from botocore.exceptions import ClientError

import codecs
from collections import namedtuple
import json
import os
from sms_spam_classifier_utils.sms_spam_classifier_utils import (
    one_hot_encode,
    vectorize_sequences
)

Prediction = namedtuple("Prediction", ("label", "confidence"))

sender = os.getenv("SENDER_EMAIL")
region = os.getenv("REGION")
model_endpoint = os.getenv("MODEL_ENDPOINT")
vocabulary_length = 9013

charset = "utf-8"

ses_client = boto3.client("ses", region_name=region)
sagemaker_client = boto3.client("sagemaker-runtime")

def send_response(email, prediction:Prediction):
    recipient = "tmm2169@columbia.edu"
    receive_date = None
    subject = None
    email_body = "abcdefghijklmnopqrstuvwxyz"
    body_sample = email_body[:240]

    response_lines = "\n\n".join([
            f"We received your email sent at {receive_date} with the subject {subject}.",
            f"Here is a 240 character sample of the email body: {body_sample}",
            f"The email was categorized as {prediction.label} with a {prediction.confidence}% confidence."
        ])

    try:
        response = ses_client.send_email(
            Source=sender,
            Destination={
                "ToAddresses": [recipient],
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


def predict(email):
    print('Submitting to endpoint: ', model_endpoint)

    one_hot_data = one_hot_encode([email], vocabulary_length)
    encoded_message = vectorize_sequences(one_hot_data, vocabulary_length)

    response = sagemaker_client.invoke_endpoint(
        EndpointName=model_endpoint,
        Body=encoded_message,
        ContentType='text/csv'
    )
    print('Received response: ')
    print(response)

    body = json.load(codecs.getreader('utf-8')(response['Body']))
    print('Response body: ')
    print(body)

    return Prediction("spam", 0.9)

def lambda_handler(event, context):
    print("Event: ")
    print(event)
    
    email = "something"

    prediction = predict(email)
    send_response(email, prediction)

    return {"status": "ok"}