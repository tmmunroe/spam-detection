from aws_cdk import (
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
        model_name = ""

        # TODO - sagemaker 
        # ml_model = sagemaker.Model.from_model_name(self, "Sagemaker", model_name)
        
        # lambda
        lambda_classifier = lambda_.Function(self, "SpamClassifier",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="spam_classifier.lambda_handler",
            code=lambda_.Code.from_asset("spam_detection/lambda/"),
            timeout=Duration.seconds(30),
            environment={
                "SENDER_EMAIL": email_address,
                "REGION": env.region,
            })

        # s3 bucket
        emails_prefix = "emails/"

        spam_detection_bucket = s3.Bucket(self, "SpamDetectionBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            versioned=True)

        spam_detection_bucket.add_object_created_notification(
            s3n.LambdaDestination(lambda_classifier),
            s3.NotificationKeyFilter(prefix=emails_prefix)
        )
        
        # SES receipt rule + S3 action
        receipt_rules = ses.ReceiptRuleSet(self, "EmailReceiptRules")
        receipt_rule = receipt_rules.add_rule("Email", 
            recipients=[email_address])
        receipt_rule.add_action(
            ses_actions.S3(
                bucket=spam_detection_bucket,
                object_key_prefix=emails_prefix
            )
        )
        
        # hosted zone
        hosted_zone = route53.HostedZone.from_hosted_zone_attributes(self, "route53-hosted-zone", 
            hosted_zone_id=hosted_zone_id,
            zone_name=domain_name)

        # email identity
        email = ses.EmailIdentity(self, "EmailIdentity",
            identity=ses.Identity.public_hosted_zone(hosted_zone=hosted_zone)
        )

        # permissions - 
        # --ses can write emails to s3 bucket
        spam_detection_bucket.grant_put(iam.ServicePrincipal("ses.amazonaws.com"), 
            emails_prefix)

        # --s3 can invoke lambda
        lambda_classifier.grant_invoke(iam.ServicePrincipal("s3.amazonaws.com"))

        # --lambda can invoke sagemaker and send emails
        sagemaker_policy = iam.PolicyStatement()
        sagemaker_policy.add_actions("sagemaker:InvokeEndpoint", "sagemaker:InvokeEndpointAsync")
        sagemaker_policy.add_all_resources()
        lambda_classifier.add_to_role_policy(sagemaker_policy)

        ses_policy = iam.PolicyStatement()
        ses_policy.add_actions("ses:SendEmail", "ses:SendRawEmail")
        ses_policy.add_all_resources()
        lambda_classifier.add_to_role_policy(ses_policy)
