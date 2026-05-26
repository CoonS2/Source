from dotenv import load_dotenv
import os

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import gradio as gr

import logging

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


def interview_text(genre):
    
    
    system_prompt = """
                인터뷰를 한다고 생각하고 대상의 주제를 가지고 일단 핵심 내용을 정리하고 인터뷰 질문을 생성해줘.
                """
    
    
    user_prompt=f"""
            아래 내용 참고해서
            1. 장르 특징 5줄 정리
            2. 인터뷰 질문 8개 작성해줘
            - genre : {genre}
        """
    
    print(genre)
    
    messages = [
                    # 시스템 프롬프트
                    {"role" : "system", "content" : system_prompt},
                    {"role" : "user", "content" : user_prompt}, 
    ]
    
    generated_response = model.chat(messages=messages)
    return generated_response['choices'][0]['message']['content']
    
demo = gr.Interface(
    fn=interview_text,
    inputs=[gr.Text(label = "genre"),

    ],   
    
    outputs=[gr.Markdown()],
    title="인터뷰 질문 생성 프로그램",
    description="질문 생성"
)

demo.launch()