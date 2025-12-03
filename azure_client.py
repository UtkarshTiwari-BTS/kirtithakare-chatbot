from openai import AzureOpenAI

client = None

deployment_model = None


def init_azure_client(api_key, endpoint, deployment):
    """
    Creates AzureOpenAI client dynamically with user-provided values.
    """
    global deployment_model
    global client
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2025-01-01-preview"
    )
    deployment_model = deployment
