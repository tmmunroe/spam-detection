import json
import boto3
import os
import datetime

#USER VARIABLES
    
client = boto3.client('sagemaker')

timenow = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
image = os.environ['image']
role_arn = os.environ['role_arn']

s3_raw_data = os.environ['s3_raw_data']
s3_train_data = os.environ['s3_train_data']
s3_val_data = os.environ['s3_val_data']

local_raw_data = os.environ['local_raw_data']
local_train_data = os.environ['local_train_data']
local_val_data = os.environ['local_val_data']


def run_preprocessing():
    response = client.create_processing_job(
        ProcessingJobName=f'PreProcessingSpam-{timenow}',
        AppSpecification={
            'ImageUri': image,
            'ContainerEntrypoint': [
                'python3',
            ],
            'ContainerArguments': [
                '/opt/ml/code/preprocess.py',
            ]
        },
        RoleArn=role_arn,
        ProcessingInputs=[
            {
                'InputName': 'input-1',
                'S3Input': {
                    'S3Uri': s3_raw_data,
                    'LocalPath': local_raw_data,
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated',
                }
            },
        ],
        ProcessingOutputConfig={
            'Outputs': [
                {
                    'OutputName': 'train',
                    'S3Output': {
                        'S3Uri': s3_train_data,
                        'LocalPath': local_train_data,
                        'S3UploadMode': 'EndOfJob'
                    },
                    'AppManaged': False
                },
                {
                    'OutputName': 'val',
                    'S3Output': {
                        'S3Uri': s3_val_data,
                        'LocalPath': local_val_data,
                        'S3UploadMode': 'EndOfJob'
                    },
                    'AppManaged': False
                },
            ],
        },
        ProcessingResources={
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': 'ml.m5.large',
                'VolumeSizeInGB': 30
            }
        },
    )

    print(response)


def lambda_handler(event, context):    
    try:
        run_preprocessing()
    except Exception as e:
        print("something went wrong ...")
        print(e)
    
    return {
        'statusCode': 200,
        'body': 'body'
    }