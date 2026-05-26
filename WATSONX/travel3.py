from dotenv import load_dotenv
import os

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import gradio as gr
from PIL import Image
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
            "max_tokens": 1000
            }
)

# 💡 복복형 이미지 변환 함수 추가
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def recommend(message, history):
    
    print("history :", history)
    
    system_prompt = """
                너는 여행 스캐줄러 AI
                
                
                사용자가 업로드한 이미지의
                - 분위기
                - 감성
                - 색감
                - 스타일
                을 분석해서 여행지를 추천해줘
                
                반드시
                1. 이미지 분위기 분석
                2. 추천 여행지
                3. 추천 이유
                4. 추천 활동
                5. 한글로 답변
                """
    
    user_prompt=f"""
            다음 내용을 참고해서 계획 짜줘
            - 내용 : {message}
        """
    
    messages = [
                    # 시스템 프롬프트
                    {"role" : "system", "content" : system_prompt},
                    # {"role" : "user", "content" : user_prompt},
    ]
    
    for item in history:
        role = item["role"]
        content = item["content"]
        
        # assistant answer save
        texts = []
        
        if isinstance(content, list):
            for c in content:
                
                # 텍스트만 추출
                if c.get("type") == "text":
                    texts.append(c.get("text", ""))
                    
        elif isinstance(content, str):
            texts.append(content)
            
        messages.append({"role" : role, "content" : " ".join(texts)})
        
    # messages : text, files
    text = message.get('text', '')
    files = message.get('files', '')
    
    if files:
        image = Image.open(files[0])

        base64_image = image_to_base64(image)
        
        # 💡 2. 공중에 떠 있던 딕셔너리를 messages.append()로 정상 추가합니다.
        messages.append({
            "role" : "user", 
            "content" : [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                },
                { "type": "text", "text": user_prompt },
            ],
        })
        
    else:
        messages = [{"role" : "user", "content" : text}]
    
    # generated_response = model.chat(messages=messages)
    # return generated_response['choices'][0]['message']['content']
    
    # chat_stream()
    generated_response = model.chat_stream(messages=messages)
    
    full_response = ""
    for chunk in generated_response:
        if chunk['choices'] : 
            full_response += chunk["choices"][0]["delta"].get("content", "")
            yield full_response
            
demo = gr.ChatInterface(
    fn=recommend, 
    multimodal=True,
    title="AI 여행 플래너",
    description="여행지역, 예산, 여행스타일, 여행 기간 등을 입력하면 AI가 맞춤형 여행일정을 추천해 드립니다."
)

demo.launch()