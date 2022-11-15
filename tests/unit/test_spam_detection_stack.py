import aws_cdk as core
import aws_cdk.assertions as assertions

from spam_detection.spam_detection_stack import SpamDetectionStack

# example tests. To run these tests, uncomment this file along with the example
# resource in spam_detection/spam_detection_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = SpamDetectionStack(app, "spam-detection")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
