from dotenv import load_dotenv
import os

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import gradio as gr
import base64
import io
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
    model_id="meta-llama/llama-3-2-11b-vision-instruct",
    api_client=client,
    project_id=f"{project_id}",
    params = {
            "max_tokens": 3000
            }
)

def image_to_base64(image):
    """
    pillow 형식의 이미지를 가져와 원하는 포맷으로 저장
    base64 인코딩 형식으로 이미지 리턴
    """
    
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    
    
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64

def image_text(image, user_prompt):
    
    if image is None:
        return  "이미지를 입력해 주세요"
    
    # base64
    base64_image = image_to_base64(image)
    
    
    system_prompt = """
                당신은 이미지 분석 전문가입니다.
                사용자의 요청에 따라 이미지를 분석하세요.
                - 이미지 설명
                - 분위기 분석
                - 감정 분석 
                - 객체 설명
                - 캡션 생성
                - 여행 추천
                - 스타일 분석

                등을 수행하세요

                항상:
                - 한국어로 답변
                - 사용자의 요청 의도를 우선 반영
                - 읽기 쉽게 작성
                """
    
    if user_prompt is None:
            user_prompt = f"""
                이미지를 분석해줘.
            """
    
    messages = [
                    # 시스템 프롬프트
                    {"role" : "system", "content" : system_prompt},
                    {"role" : "user", "content" : [
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/png;base64,{base64_image}"
            }
          },
          {
            "type": "text",
            "text": user_prompt
          },
        ],
    }
    ]
    
    generated_response = model.chat(messages=messages)
    return generated_response['choices'][0]['message']['content']
    
demo = gr.Interface(
    fn=image_text,
    inputs=[gr.Image(type="pil"),
            gr.Text(label="유저 명령"),
    ],

    outputs=[gr.Markdown()],
    title="이미지 분석 프로그램",
    description="이미지 업로드 시 AI가 이미지 분석"
)

demo.launch()