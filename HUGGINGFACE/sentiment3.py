import gradio as gr
from transformers import pipeline
import re

# 문장 여러개 ( 파일 )
# 리뷰 데이터 -> 분석

english_classifier = pipeline("sentiment-analysis", top_k = None)
korean_classifier = pipeline("sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment", top_k = None)

def is_korean(text):
    korean = re.search(r"[가-힣]", text)
    return korean is not None
    

def predict_sentiment(text):
    
    # 엔터를 기준으로 문장 분리
    sentences = text.splitlines()
    sentences = [s.strip() for s in sentences if s.strip()]
    
    results_text=[]
    
    # 한국말인지 확인하기
    if is_korean(text):
        language = "한국어 모델"
        results = korean_classifier(sentences)
    else:
        language = "영어 모델"
        results = english_classifier(sentences)
        
    # {'label' : 'POSITIVE', 'score' : 0.9192341028490124}
    # {'label' : 'LABEL_1', 'score' : 0.9192341028490124}
    
    # label = results[0]["label"]
    label_map = {"LABEL_0" : "부정 😡", "LABEL_1" : "긍정 😄", "NEGATIVE" : "부정 😡", "POSITIVE" : "긍정 😄"}
    
    # label = label_map.get(label, label)
    # score1 = result[0]["score"]
    # score2 = result[1]["score"]
    
    # return f"사용모델 : {language}\n 감정 : {label}\n 확률 : ({score1:.4f})\n OTHER : ({score2:.4f})"
    
    for sentence, result in zip(sentences, results):
        # 결과를 전체 다 받은 상태
        best = max(result, key=lambda x:x['score'])
        label = best["label"]
        label = label_map.get(label, label)
        score = best['score']
        # results_text += (f"문장 : {sentence}\n 감정 : {label}\n 확률 : {score:.4f}\n\n")
        results_text.append([sentence, label, score])
        
    return results_text
    
    

demo = gr.Interface(
    fn = predict_sentiment,
    inputs=[gr.Textbox(lines=3, placeholder="문장을 입력하세요")],
    # outputs=[gr.Textbox(label="분석결과", lines=10)],
    outputs=[gr.Dataframe(headers=["문장", "감정", "확률"])],
    title = "AI 감정분석 웹",
    description="HuggingFace Transformer 기반 감정 분석 프로그램",
)

demo.launch()