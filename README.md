Hello everyone,

Assignment 3 has been released and can be accessed on Courseworks under Files -> Assignments [link].

The due date for submission is 12/05 11:59 PM.


Rubric:

Correct classification for SPAM email - 20 - GOOD

Correct classification for HAM email - 20 - GOOD

receive_date in response - 4 - GOOD

subject in response - 4 - GOOD

body in response - 4 - GOOD

Confidence Score in response - 4 - GOOD

Uses stack parameter/env variable for prediction endpoint - 4 - GOOD

Functional stack created without error - 5 - GOOD

5 of the following 7 resources created using CloudFormation (7 points/resource): - GOOD

- IAM roles

- Lambda Function LF1

- Bucket S1

- Lambda InvokePermission for S1

- BucketPolicy (Object Put/Create trigger)

- ReceiptRule

- ReceiptRuleSet (no need to activate)

(Extra Credit) Lambda function that runs the retraining and code deployment - 10 - GOOD


Note: Please make sure that your CloudFormation resources interface with each other for this assignment. The CloudFormation template should mimic the "actual" configuration as much as possible. For example, if there is a bucket that invokes a Lambda function every time a "Put" event is triggered on it, please make sure that your CloudFormation template takes care of it. Only partial credit will be awarded for creating "dummy" or standalone resources without appropriate permissions, policies or interactions.

Good luck!

Best,
Instructional Staff.

cdk destroy -> 11:12pm -> 11:21pm (with issues- receipt rule set wasn't deactivated)
./deploy.sh -> 11:32pm -> 12:42pm


destroy stacks - how long?
deploy new stacks - how long?
test email functionality

destroy steps-
deactivate ses rule set
cdk destroy --all


deployment steps-
./deploy-model.sh
./deploy.sh

go to receipt rule sets, activate receipt rule
put raw data in s3