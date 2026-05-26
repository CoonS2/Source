import gradio as gr
from transformers import pipeline
import re

english_classifier = pipeline("sentiment-analysis", top_k = None)
korean_classifier = pipeline("sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment", top_k = None)

def is_korean(text):
    korean = re.search(r"[가-힣]", text)
    return korean is not None
    

def predict_sentiment(text):
    
    # 한국말인지 확인하기
    if is_korean(text):
        language = "한국어 모델"
        results = korean_classifier(text)[0]
    else:
        language = "영어 모델"
        results = english_classifier(text)[0]
        
    # {'label' : 'POSITIVE', 'score' : 0.9192341028490124}
    # {'label' : 'LABEL_1', 'score' : 0.9192341028490124}
    
    # label = results[0]["label"]
    label_map = {"LABEL_0" : "부정 😡", "LABEL_1" : "긍정 😄", "NEGATIVE" : "부정 😡", "POSITIVE" : "긍정 😄"}
    
    # label = label_map.get(label, label)
    # score1 = result[0]["score"]
    # score2 = result[1]["score"]
    
    # return f"사용모델 : {language}\n 감정 : {label}\n 확률 : ({score1:.4f})\n OTHER : ({score2:.4f})"
    
    scores = {}
    
    for item in results:
        label = item["label"]
        scores[label_map.get(label, label)] = item["score"]
        
    return scores
    
    

demo = gr.Interface(
    fn = predict_sentiment,
    inputs=[gr.Text(lines=3, placeholder="문장을 입력하세요")],
    outputs=[gr.Label(num_top_classes=2)],
    title = "AI 감정분석 웹",
    description="HuggingFace Transformer 기반 감정 분석 프로그램",
)

demo.launch()