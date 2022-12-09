import json
import boto3
import os
import datetime

#USER VARIABLES
    
client = boto3.client("sagemaker")


region = "us-east-1"
role_arn = os.environ["role_arn"]

# sagemaker environments
preprocess_instance_type = os.environ["preprocess_instance_type"]
preprocess_image = os.environ["preprocess_image"]

train_instance_type = os.environ["train_instance_type"]
train_image = os.environ["train_image"]

deploy_instance_type = os.environ["deploy_instance_type"]
deploy_image = os.environ["deploy_image"]

# s3 locations
s3_raw_data = os.environ["s3_raw_data"]
s3_train_data = os.environ["s3_train_data"]
s3_val_data = os.environ["s3_val_data"]
s3_model_output = os.environ["s3_model_output"]
s3_model_logs = os.environ["s3_model_logs"]

# output names processing job
train_output_name = "train"
val_output_name = "val"

# local paths for sagemaker environments
local_raw_data = os.environ["local_raw_data"]
local_train_data = os.environ["local_train_data"]
local_val_data = os.environ["local_val_data"]

# model and endpoint names
base_model_name = os.environ["model_name"]
base_endpoint_config_name = os.environ["endpoint_config_name"]
base_endpoint_name = os.environ["endpoint_name"]



# PREPROCESSING LAMBDAS
def run_preprocessing(event:dict, timenow:str):
    processing_job_name = f"PreProcessingSpam-{timenow}"
    response = client.create_processing_job(
        ProcessingJobName=processing_job_name,
        AppSpecification={
            "ImageUri": preprocess_image,
            "ContainerEntrypoint": [
                "python3",
            ],
            "ContainerArguments": [
                "/opt/ml/code/preprocess.py",
            ]
        },
        RoleArn=role_arn,
        ProcessingInputs=[
            {
                "InputName": "input-1",
                "S3Input": {
                    "S3Uri": s3_raw_data,
                    "LocalPath": local_raw_data,
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": train_output_name,
                    "S3Output": {
                        "S3Uri": s3_train_data,
                        "LocalPath": local_train_data,
                        "S3UploadMode": "EndOfJob"
                    },
                    "AppManaged": False
                },
                {
                    "OutputName": val_output_name,
                    "S3Output": {
                        "S3Uri": s3_val_data,
                        "LocalPath": local_val_data,
                        "S3UploadMode": "EndOfJob"
                    },
                    "AppManaged": False
                },
            ],
        },
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": preprocess_instance_type,
                "VolumeSizeInGB": 30
            }
        },
    )

    print(response)

    return {
        "ProcessingJobName": processing_job_name,
        "S3TrainPath": s3_train_data,
        "S3ValPath": s3_val_data
    }


def processing_job_status(event:dict):
    processing_job_name = event["ProcessingJobName"]
    response = client.describe_processing_job(
        ProcessingJobName=processing_job_name
    )

    def find_output_path(output_name) -> str:
        outputs = response["ProcessingOutputConfig"]["Outputs"]
        for output in outputs:
            if output["OutputName"] == output_name:
                return output["S3Output"]["S3Uri"]
        return ''

    return {
        "ProcessingJobName": processing_job_name,
        "Status": response["ProcessingJobStatus"],
        "S3TrainPath": find_output_path(train_output_name),
        "S3ValPath": find_output_path(val_output_name)
    }


# TRAINING LAMBDAS
def run_training(event:dict, timenow:str):
    training_job_name = f"TrainingSpam-{timenow}"
    train_data_uri = s3_train_data
    val_data_uri = s3_val_data
    model_output_uri = s3_model_output
    model_logs_uri = s3_model_logs

    response = client.create_training_job(
        TrainingJobName=training_job_name,
        RoleArn=role_arn,
        AlgorithmSpecification={ 
            "TrainingImage": train_image,
            "TrainingInputMode": "File",
        },
        DebugHookConfig={ 
            "CollectionConfigurations": [],
            "S3OutputPath": model_logs_uri
        },
        HyperParameters={ 
            "batch-size": "100",
            "epochs": "20",
            "learning-rate": "0.01",
            "sagemaker_container_log_level": "20",
            "sagemaker_region": region,
            "sagemaker_program": "train.py",
        },
        InputDataConfig=[ 
            { 
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": { 
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                        "S3Uri": train_data_uri
                    }
                }
            },
            { 
                "ChannelName": "val",
                "DataSource": {
                    "S3DataSource": { 
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                        "S3Uri": val_data_uri
                    }
                },
            }
        ],
        OutputDataConfig={ 
            "S3OutputPath": model_output_uri
        },
        ProfilerConfig={ 
            "S3OutputPath": model_logs_uri,
        },
        ProfilerRuleConfigurations=[
            {
                "RuleConfigurationName": "ProfilerReport-1670535161",
                "RuleEvaluatorImage": "503895931360.dkr.ecr.us-east-1.amazonaws.com/sagemaker-debugger-rules:latest",
                "RuleParameters": {
                    "rule_to_invoke": "ProfilerReport"
                }
            }
        ],
        ResourceConfig={ 
            "VolumeSizeInGB": 30,
            "InstanceCount": 1,
            "InstanceType": train_instance_type,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 86400
        }
    )

    print(response)
    return { 
        "TrainingJobName": training_job_name
    }


def training_job_status(event:dict):
    training_job_name = event["TrainingJobName"]
    response = client.describe_training_job(
        TrainingJobName=training_job_name
    )
    print(response)

    status = response["TrainingJobStatus"]
    model_artifacts = None

    if status == "Completed":
        artifacts = response.get("ModelArtifacts")
        if artifacts is None:
            raise ValueError("Status is completed, but no model artifacts were detected")
        
        model_artifacts = artifacts.get("S3ModelArtifacts")
        if model_artifacts is None:
            raise ValueError("Status is completed, but no S3 model artifacts were detected")

    return {
        "Status": status,
        "S3ModelArtifacts": model_artifacts
    }


# DEPLOYMENT LAMBDAS

def create_model(model_data_url:str, timenow:str):
    model_name = f"{base_model_name}-{timenow}"

    print(f"Creating model: {model_name}")
    response = client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": deploy_image,
            "ModelDataUrl": model_data_url,
            "Environment": {
                "SAGEMAKER_PROGRAM": "spam_model.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": region,
                "MMS_DEFAULT_RESPONSE_TIMEOUT": "500"
            }
        }
    )

    print(response)
    return model_name


def create_endpoint_config(model_name:str, timenow:str):
    endpoint_config_name = f"{base_endpoint_config_name}-{timenow}"

    print(f"Creating endpoint config: {endpoint_config_name}")
    response = client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "variant1",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": deploy_instance_type,
            },
        ]
    )

    print(response)
    return endpoint_config_name


def endpoint_exists():
    print(f"Checking endpoint existence: {base_endpoint_name}")
    response = client.list_endpoints(NameContains=base_endpoint_name)
    print(response)
    endpoint_names = { endpoint["EndpointName"] for endpoint in response["Endpoints"] }
    return base_endpoint_name in endpoint_names


def deploy_endpoint(endpoint_config_name:str):
    if endpoint_exists():
        print(f"Updating endpoint: {base_endpoint_name}")
        response = client.update_endpoint(
            EndpointName=base_endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    else:
        print(f"Creating endpoint: {base_endpoint_name}")
        response = client.create_endpoint(
            EndpointName=base_endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    print(response)
    return base_endpoint_name


def run_deployment(event:dict, timenow:str):
    model_data_url = event["S3ModelArtifacts"]
    try:
        model_name = create_model(model_data_url, timenow)
        endpoint_config_name = create_endpoint_config(model_name, timenow)
        endpoint_name = deploy_endpoint(endpoint_config_name)
        status = "Completed"
    except Exception as ex:
        print("Deployment failed with exception: {ex}")
        endpoint_name = "DeploymentFailed"
        status = "Failed"

    return {
        "ModelDataUrl": model_data_url,
        "EndpointName": endpoint_name,
        "Status": status
    }


def endpoint_status(event:dict):
    endpoint_name = event["EndpointName"]
    response = client.describe_endpoint(EndpointName=endpoint_name)
    print(response)
    
    return {
        "EndpointName": endpoint_name,
        "Status": response["EndpointStatus"]
    }


def lambda_handler(event, context):    
    print(event)
    task = event["Task"]
    timenow = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    try:
        if task == "preprocess":
            response = run_preprocessing(event, timenow)
        elif task == "preprocess-status":
            response = processing_job_status(event)
        elif task == "train":
            response = run_training(event, timenow)
        elif task == "train-status":
            response = training_job_status(event)
        elif task == "deploy":
            response = run_deployment(event, timenow)
        elif task == "deploy-status":
            response = endpoint_status(event)
        else:
            raise ValueError(f'Unrecognized task {task}')

    except Exception as e:
        print("something went wrong ...")
        print(e)
        response = {
            "Status": "Failed",
            "Exception": str(e)
        }
            
    return response
