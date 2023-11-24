import boto3
import json

def lambda_handler(event, context):
    
    sm_runtime = boto3.client('sagemaker-runtime')

    
    endpoint_name = '<Enter your endpoint here>'

    input_data = event['input']  

    
    input_payload = {
        'inputs': input_data
    }

    
    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(input_payload),  
        ContentType='application/json',  
    )

    
    result = json.loads(response['Body'].read().decode())

    return {
        'statusCode': 200,
        'body': result
    }
