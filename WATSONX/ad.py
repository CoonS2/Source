from dotenv import load_dotenv
import os

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import gradio as gr

# .env 내용 가져오기
load_dotenv()

apikey = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_ai_url = os.getenv("WATSONX_URL")

credentials = Credentials(
    url = f"{watsonx_ai_url}",
    api_key = f"{apikey}",
)
client = APIClient(credentials)

model = ModelInference(
    model_id="ibm/granite-4-h-small",
    api_client=client,
    project_id=f"{project_id}",
    params = {
            "max_tokens": 1000
            }
)


def ad_text(name, brand_name, strength, tone, keyword, value):
    
    
    
    system_prompt = """
                1. SWOT 분석하여 광고 문구 작성해줘
                """
    
    
    user_prompt=f"""
            아래 내용을 참고해서 1~2줄짜리 광고 문구 5개 작성해줘.
            - 제품명 : {name}
            - 브랜드명 : {brand_name}
            - 제품특징 : {strenghth}
            - 톤앤매너 : {tone}
            - 브랜드 핵심 가치 : {value}
            - 필수 포함 키워드 : {keyword}
        """
    
    
    messages = [
                    # 시스템 프롬프트
                    {"role" : "system", "content" : system_prompt},
                    {"role" : "user", "content" : user_prompt}, 
    ]
    
    generated_response = model.chat(messages=messages)
    return print(generated_response['choices'][0]['message']['content'])
    
demo = gr.Interface(
    fn=ad_text,
    inputs=[gr.Text(label = "제품명"),
            gr.Text(label = "브랜드명"), 
            gr.Text(label = "제품 특징"), 
            gr.Text(label = "톤앤매너"), 
            gr.Text(label = "필수 포함 키워드"), 
            gr.Text(label = "브랜드 핵심 가치"), 
    ],   
    
    outputs=[gr.Markdown()],
    title="광고 문구 프로그램",
    description="텍스트 입력 시 ai가 광고 문구를 작생해 드립니다."
)

demo.launch()