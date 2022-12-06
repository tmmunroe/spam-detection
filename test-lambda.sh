cdk synth
sam build -t ./cdk.out/SpamDetectionStack.template.json
sam local invoke SpamClassifier -e s3_event.json -t ./cdk.out/SpamDetectionStack.template.json -l test-lambda.log
