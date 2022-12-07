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

class SageMakerWorkflowStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, env:Environment, bucket:s3.Bucket, object_prefix:str) -> None:
        super().__init__(scope, construct_id, env=env)

        # TODO - sagemaker 
        # model_name = "spam-detection-tmm2169"
        # ml_model = sagemaker.Model.from_model_name(self, "Sagemaker", model_name)
        # model_endpoint = "sms-spam-classifier-mxnet-2022-12-03-19-07-10-957"
        # model_endpoint_param = CfnParameter(self, "ModelEndpoint",
        #     type="String", default=model_endpoint,
        #     description="The SageMaker endpoint for a spam classifier.")
        role_arn = "arn:aws:iam::756059218166:role/service-role/AmazonSageMaker-ExecutionRole-20221123T110444"

        # lambda
        lambda_preprocessing = lambda_.Function(self, "SagemakerPreprocessor",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="preprocess.lambda_handler",
            code=lambda_.Code.from_asset("sagemaker_model_docker/lambda/"),
            timeout=Duration.seconds(30),
            environment={
                "image": "756059218166.dkr.ecr.us-east-1.amazonaws.com/mxnet-container-tmandevi",
                "role_arn": role_arn,
                "s3_raw_data": f"s3://{bucket.bucket_name}/{object_prefix}/data/raw/SMSSpamCollection",
                "s3_train_data": f"s3://{bucket.bucket_name}/{object_prefix}/output/data/train",
                "s3_val_data":f"s3://{bucket.bucket_name}/{object_prefix}/output/data/val",
                "local_raw_data": "/opt/ml/processing/input",
                "local_train_data": "/opt/ml/processing/train",
                "local_val_data": "/opt/ml/processing/val"
            })

        bucket.grant_read_write(lambda_preprocessing)

        lambda_preprocessing.add_to_role_policy(iam.PolicyStatement(actions=['sagemaker:CreateProcessingJob',],
        	resources=["*"]))
        lambda_preprocessing.add_to_role_policy(iam.PolicyStatement(actions=['iam:PassRole',],
        	resources=[role_arn])) 