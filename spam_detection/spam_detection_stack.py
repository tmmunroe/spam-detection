from aws_cdk import (
    CfnParameter,
    Duration,
    Environment,
    RemovalPolicy,
    Stack,
    # aws_sqs as sqs,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_route53 as route53,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
    aws_sagemaker_alpha as sagemaker,
    aws_ses as ses,
    aws_ses_actions as ses_actions
)
from constructs import Construct

class SpamDetectionStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, env:Environment) -> None:
        super().__init__(scope, construct_id, env=env)

        domain_name = "project-tmm2169.com"
        hosted_zone_id = "Z03027081WT8OA0QAB8NK"

        email_address = f"spamDetector@{domain_name}"

        # TODO - sagemaker 
        model_name = "spam-detection-tmm2169"
        ml_model = sagemaker.Model.from_model_name(self, "Sagemaker", model_name)
        model_endpoint_param = CfnParameter(self, "ModelEndpoint",
            type="String", default="sms-spam-classifier-mxnet-2022-12-03-19-07-10-957",
            description="The SageMaker endpoint for a spam classifier.")

        # layer
        lambda_layer = lambda_.LayerVersion.from_layer_version_attributes(self, "LambdaLayer",
            layer_version_arn='arn:aws:lambda:us-east-1:336392948345:layer:AWSSDKPandas-Python39:1',
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_9])

        # lambda
        lambda_classifier = lambda_.Function(self, "SpamClassifier",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="spam_classifier.lambda_handler",
            code=lambda_.Code.from_asset("spam_detection/lambda/"),
            timeout=Duration.seconds(30),
            layers=[lambda_layer],
            environment={
                "SENDER_EMAIL": email_address,
                "REGION": env.region,
                "ENDPOINT": model_endpoint_param.value_as_string,
            })

        # s3 bucket
        sagemaker_prefix = "sms-spam-classifier/"
        emails_prefix = "emails/"

        self.spam_detection_bucket = s3.Bucket(self, "SpamDetectionBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            versioned=True)

        self.spam_detection_bucket.add_object_created_notification(
            s3n.LambdaDestination(lambda_classifier),
            s3.NotificationKeyFilter(prefix=emails_prefix)
        )
        
        # SES receipt rule + S3 action
        receipt_rules = ses.ReceiptRuleSet(self, "EmailReceiptRules")
        receipt_rule = receipt_rules.add_rule("Email", 
            recipients=[email_address])
        receipt_rule.add_action(
            ses_actions.S3(
                bucket=self.spam_detection_bucket,
                object_key_prefix=emails_prefix
            )
        )
        
        # hosted zone
        hosted_zone = route53.HostedZone.from_hosted_zone_attributes(self, "route53-hosted-zone", 
            hosted_zone_id=hosted_zone_id,
            zone_name=domain_name)
        
        mx_record = route53.MxRecord(self, "MXRecord", 
            zone=hosted_zone,
            delete_existing=True,
            values=[
                route53.MxRecordValue(
                    host_name="inbound-smtp.us-east-1.amazonaws.com",
                    priority=10
                )
            ]
        )

        # email identity
        email = ses.EmailIdentity(self, "EmailIdentity",
            identity=ses.Identity.public_hosted_zone(hosted_zone=hosted_zone)
        )

        # permissions - 

        # --ses can write emails to s3 bucket
        self.spam_detection_bucket.grant_put(iam.ServicePrincipal("ses.amazonaws.com"), 
            emails_prefix)

        # --s3 can invoke lambda
        lambda_classifier.grant_invoke(iam.ServicePrincipal("s3.amazonaws.com"))

        # --lambda can invoke sagemaker and send emails and get from s3
        self.spam_detection_bucket.grant_read(lambda_classifier)

        sagemaker_policy = iam.PolicyStatement()
        sagemaker_policy.add_actions("sagemaker:InvokeEndpoint", "sagemaker:InvokeEndpointAsync")
        sagemaker_policy.add_all_resources()
        lambda_classifier.add_to_role_policy(sagemaker_policy)

        ses_policy = iam.PolicyStatement()
        ses_policy.add_actions("ses:SendEmail", "ses:SendRawEmail")
        ses_policy.add_all_resources()
        lambda_classifier.add_to_role_policy(ses_policy)
