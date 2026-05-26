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


def summarize_text(text):
    
    if not text.strip():
        return "텍스트를 입력해주세요."
    
    instructions = """
                당신은 텍스트를 한국어로 요악하는 전문가이빈다.
                - 당신의 임무는 아래 주어진 텍스트 문장을 한국어로 요약하는 것입니다.
                - 요약 시 다음 사항을 반드시 반영해야 합니다.
                - 중복된 내용은 생략하되, 반복되는 내용은 요약해서 더 강조합니다.
                - 사례 중심보다는 개념과 주장 중심으로 요약합니다.
                - 3줄 이내로 요약합니다.
                - 블릿 기호 형식으로 작성합니다.
                """
    
    
    messages = [
                    # 시스템 프롬프트
                    {"role" : "system", "content" : instructions},
                    {"role" : "user", "content" : text}, 
    ]
    
    generated_response = model.chat(messages=messages)
    return print(generated_response['choices'][0]['message']['content'])
    
demo = gr.Interface(
    fn=summarize_text,
    inputs=[gr.TextArea(lines=10, placeholder="요약할 내용의 텍스트 입력..", label="입력")],
    outputs=[gr.Markdown()],
    title="watsonx 기반의 요약 프로그램",
    description="텍스트 입력 시 ai 요약"
)

demo.launch()