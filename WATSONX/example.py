from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

credentials = Credentials(
   url = "{watsonx_ai_url}",
   api_key = "{apikey}",
)
client = APIClient(credentials)

model = ModelInference(
      model_id="ibm/granite-13b-instruct-v2",
      api_client=client,
      project_id="{project_id}",
      params = {
            "max_new_tokens": 100
      }
   )
   
prompt = 'How far is Paris from Bangalore?'

print(model.generate(prompt))
print(model.generate_text(prompt))
