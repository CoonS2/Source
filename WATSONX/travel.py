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


def recommend(local, money, style, day):
    
    system_prompt = """
                너는 여행 스캐줄러 AI
                한글로 답변해주고
                여행사의 여행 스케쥴 처럼 짜줘
                """
    
    
    user_prompt=f"""
            다음 내용을 참고해서 계획 짜줘
            - 지역 : {local}
            - 비용 :{money}
            - 여행 스타일 : {style}
            - 기간 :{day}
        """
    
    messages = [
                    # 시스템 프롬프트
                    {"role" : "system", "content" : system_prompt},
                    {"role" : "user", "content" : user_prompt}, 
    ]
    
    # generated_response = model.chat(messages=messages)
    # return generated_response['choices'][0]['message']['content']
    
    # chat_stream()
    generated_response = model.chat_stream(messages=messages)
    
    full_response = ""
    for chunk in generated_response:
        if chunk['choices'] : 
            full_response += chunk["choices"][0]["delta"].get("content", "")
            yield full_response
            
demo = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Text(label = "여행 지역"),
        gr.Slider(10, 300, label="예산(만원)"),
        gr.Dropdown(["모험", "휴양", "문화", "음식", "액티비티"], label="여행 스타일"),
        gr.Radio(["1day", "2~3day", "4~7day", "1week~"], label="traval range"),
    ],   
    
    outputs=[gr.Textbox(lines=5)],
    title="AI 여행 플래너",
    description="여행지역, 예산, 여행스타일, 여행 기간 등을 입력하면 AI가 맞춤형 여행일정을 추천해 드립니다."
)

demo.launch()