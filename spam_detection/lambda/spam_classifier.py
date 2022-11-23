import boto3
from botocore.exceptions import ClientError
from collections import namedtuple
import os

Prediction = namedtuple("Prediction", ("label", "confidence"))

sender = os.getenv("SENDER_EMAIL")
region = os.getenv("REGION")
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
    # sagemaker_client.invoke_endpoint()
    return Prediction("spam", 0.9)


def lambda_handler(event, context):
    email = "something"

    prediction = predict(email)
    send_response(email, prediction)

    return {"status": "ok"}