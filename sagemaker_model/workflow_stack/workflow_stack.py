from aws_cdk import (
    CfnParameter,
    Duration,
    Environment,
    RemovalPolicy,
    Stack,
    # aws_sqs as sqs,
    aws_events as events,
    aws_events_targets as targets,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_route53 as route53,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
    aws_sagemaker_alpha as sagemaker,
    aws_ses as ses,
    aws_ses_actions as ses_actions,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_logs as logs
)
from constructs import Construct

class SagemakerWorkflowStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, env:Environment, bucket:s3.Bucket, object_prefix:str) -> None:
        super().__init__(scope, construct_id, env=env)

        base_model_name = "spam-mxnet-classifier"
        base_endpoint_config_name = "spam-mxnet-classifier-endpoint-config"
        base_endpoint_name = "spam-mxnet-classifier-endpoint"

        preprocessing_instance_type = "ml.m5.large"
        training_instance_type = "ml.c5.2xlarge"
        inference_instance_type = "ml.m5.large"

        role_arn = "arn:aws:iam::756059218166:role/service-role/AmazonSageMaker-ExecutionRole-20221123T110444"
        image_uri = "756059218166.dkr.ecr.us-east-1.amazonaws.com/mxnet-container-tmandevi"
        inference_image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38-ubuntu20.04-sagemaker"

        s3_raw_data = f"s3://{bucket.bucket_name}/{object_prefix}/data/raw/SMSSpamCollection"
        s3_train_data = f"s3://{bucket.bucket_name}/{object_prefix}/output/data/train"
        s3_val_data = f"s3://{bucket.bucket_name}/{object_prefix}/output/data/val"
        s3_model_output = f"s3://{bucket.bucket_name}/{object_prefix}/output/model"
        s3_model_logs = f"s3://{bucket.bucket_name}/{object_prefix}/output/model-logs"

        # workflow lambda
        lambda_workflow = lambda_.Function(self, "SagemakerWorkflowSteps",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="workflow.lambda_handler",
            code=lambda_.Code.from_asset("sagemaker_model/workflow_stack/lambda-workflow/"),
            timeout=Duration.seconds(30),
            environment={
                "role_arn": role_arn,
                "preprocess_instance_type": preprocessing_instance_type,
                "preprocess_image": image_uri,
                "train_instance_type": training_instance_type,
                "train_image": image_uri,
                "deploy_instance_type": inference_instance_type,
                "deploy_image": inference_image_uri,
                "s3_raw_data": s3_raw_data,
                "s3_train_data": s3_train_data,
                "s3_val_data": s3_val_data,
                "s3_model_output": s3_model_output,
                "s3_model_logs": s3_model_logs,
                "local_raw_data": "/opt/ml/processing/input",
                "local_train_data": "/opt/ml/processing/train",
                "local_val_data": "/opt/ml/processing/val",
                "model_name": base_model_name,
                "endpoint_config_name": base_endpoint_config_name,
                "endpoint_name": base_endpoint_name
            })

        bucket.grant_read_write(lambda_workflow)

        lambda_workflow.add_to_role_policy(iam.PolicyStatement(
            actions=[
                'sagemaker:CreateProcessingJob',
                'sagemaker:CreateTrainingJob',
                'sagemaker:CreateModel',
                'sagemaker:CreateEndpointConfig',
                'sagemaker:CreateEndpoint',
                'sagemaker:UpdateEndpoint',
                'sagemaker:ListEndpoints',
                'sagemaker:DescribeProcessingJob',
                'sagemaker:DescribeTrainingJob',
                'sagemaker:DescribeEndpoint'
            ],
        	resources=["*"]))

        lambda_workflow.add_to_role_policy(iam.PolicyStatement(
            actions=['iam:PassRole'],
        	resources=[role_arn]))


        # step functions
        preprocess_step = tasks.LambdaInvoke(self, "PreprocessStep",
            lambda_function=lambda_workflow,
            payload=sfn.TaskInput.from_object({
                "Task": "preprocess"
            }),
            result_path="$.Preprocessing"
        )

        preprocess_wait_step = sfn.Wait(self, "PreprocessWait", 
            time=sfn.WaitTime.duration(Duration.seconds(90)))


        preprocess_status_step = tasks.LambdaInvoke(self, "PreprocessStatusStep",
            lambda_function=lambda_workflow,
            payload=sfn.TaskInput.from_object({
                "ProcessingJobName": sfn.JsonPath.string_at("$.Preprocessing.Payload.ProcessingJobName"),
                "Task": "preprocess-status"
            }),
            result_path="$.PreprocessingResult"
        )
        preprocessing_failed_condition = sfn.Condition.string_equals("$.PreprocessingResult.Payload.Status", "Failed")
        preprocessing_completed_condition = sfn.Condition.string_equals("$.PreprocessingResult.Payload.Status", "Completed")

        training_step = tasks.LambdaInvoke(self, "TrainingStep",
            lambda_function=lambda_workflow,
            payload=sfn.TaskInput.from_object({
                "Task": "train"
            }),
            result_path="$.Training"
        )        

        training_wait_step = sfn.Wait(self, "TrainingWait", 
            time=sfn.WaitTime.duration(Duration.seconds(60)))


        training_status_step = tasks.LambdaInvoke(self, "TrainingStatusStep",
            lambda_function=lambda_workflow,
            payload=sfn.TaskInput.from_object({
                "TrainingJobName": sfn.JsonPath.string_at("$.Training.Payload.TrainingJobName"),
                "Task": "train-status"
            }),
            result_path="$.TrainingResult"
        )
        training_failed_condition = sfn.Condition.string_equals("$.TrainingResult.Payload.Status", "Failed")
        training_completed_condition = sfn.Condition.string_equals("$.TrainingResult.Payload.Status", "Completed")

        deployment_step = tasks.LambdaInvoke(self, "DeploymentStep",
            lambda_function=lambda_workflow,
            payload=sfn.TaskInput.from_object({
                "S3ModelArtifacts":sfn.JsonPath.string_at("$.TrainingResult.Payload.S3ModelArtifacts"),
                "Task": "deploy"
            }),
            result_path="$.Deployment"
        )

        deployment_wait_step = sfn.Wait(self, "DeploymentWait",
            time=sfn.WaitTime.duration(Duration.seconds(30)))
        
        deployment_status_step = tasks.LambdaInvoke(self, "DeploymentStatusStep",
            lambda_function=lambda_workflow,
            payload=sfn.TaskInput.from_object({
                "EndpointName": sfn.JsonPath.string_at("$.Deployment.Payload.EndpointName"),
                "Task": "deploy-status"
            }),
            result_path="$.DeploymentResult"
        )
        deployment_failed_condition = sfn.Condition.string_equals("$.DeploymentResult.Payload.Status", "Failed")
        deployment_out_of_service_condition = sfn.Condition.string_equals("$.DeploymentResult.Payload.Status", "OutOfService")
        deployment_in_service_condition = sfn.Condition.string_equals("$.DeploymentResult.Payload.Status", "InService")


        job_failed = sfn.Fail(self, "FailedStatus")
        job_succeeded = sfn.Succeed(self, "SucceedStatus")

        # fully defined tasks with polling to check completeness
        deployment_task = deployment_step.next(deployment_wait_step).next(deployment_status_step).next(
            sfn.Choice(self, "Deployment Complete?"
                ).when(
                    deployment_failed_condition, job_failed
                ).when(
                    deployment_out_of_service_condition, job_failed
                ).when(
                    deployment_in_service_condition, job_succeeded
                ).otherwise(
                    deployment_wait_step
                )
        )
        
        training_task = training_step.next(training_wait_step).next(training_status_step).next(
            sfn.Choice(self, "Training Complete?"
                ).when(
                    training_failed_condition, job_failed
                ).when(
                    training_completed_condition, deployment_task
                ).otherwise(
                    training_wait_step
                )
        )

        preprocessing_task = preprocess_step.next(preprocess_wait_step).next(preprocess_status_step).next(
            sfn.Choice(
                    self, "Preprocessing Complete?"
                ).when(
                    preprocessing_failed_condition, job_failed
                ).when(
                    preprocessing_completed_condition, training_task
                ).otherwise(
                    preprocess_wait_step
                )
        )

        # full model development chain
        model_development_chain = sfn.Chain.start(preprocessing_task)

        # state machine
        log_group = logs.LogGroup(self, "ModelDevelopmentLogs")

        sm = sfn.StateMachine(self, "StateMachine",
            definition=model_development_chain,
            timeout=Duration.minutes(45),
            logs=sfn.LogOptions(
                destination=log_group,
                level=sfn.LogLevel.ALL
            )
        )


        # # event to kick of state machine
        # rule = events.Rule(self, "Rule", 
        #     schedule=events.Schedule.cron(hour='17:00')
        # )
        # role = iam.Role(self, "Role",
        #     assumed_by=iam.ServicePrincipal("events.amazonaws.com")
        # )

        # sm_target = targets.SfnStateMachine(sm,
        #     role=role,
        #     retry_attempts=1
        # )

        # # permissions for event role
        # sm.grant_execution(role)
        # rule.add_target(sm_target)



        # event to kick of state machine
        rule = events.Rule(self, "Rule", 
            schedule=events.Schedule.cron(hour='19:00', minute='10')
        )
        sm_target = targets.SfnStateMachine(sm, retry_attempts=0)
        rule.add_target(sm_target)


