import gradio as gr
from transformers import pipeline
import re
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 문장 여러개 ( 파일 )
# 리뷰 데이터 -> 분석

english_classifier = pipeline("sentiment-analysis", top_k = None)
korean_classifier = pipeline("sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment", top_k = None)

def is_korean(text):
    korean = re.search(r"[가-힣]", text)
    return korean is not None
    

def predict_sentiment(file):
    
    # 1. file => pandas
    df = pd.read_csv(file)
    reviews = df['review'].to_list()
    
    results_text = []
    # [수정] 변수 초기화 추가
    positive_count, negative_count = 0, 0
    positive_scores, negative_scores = 0.0, 0.0
    best_positive_score, best_negative_score = 0.0, 0.0
    best_positive_review, best_negative_review = "", ""
    positive_length, negative_length = [], []
    
    label_map = {"LABEL_0" : "부정 😡", "LABEL_1" : "긍정 😄", "NEGATIVE" : "부정 😡", "POSITIVE" : "긍정 😄"}
    
    for sentence in reviews:
        # 한국말인지 확인하기
        if is_korean(sentence):
            result = korean_classifier(sentence)[0]
        else:
            result = english_classifier(sentence)[0]

        best = max(result, key=lambda x:x['score'])
        label = best["label"]
        label = label_map.get(label, label)
        score = best['score']

        # 가장 긍정적인 리뷰 (best_positive_review)와 확률(best_positive_score)
        # 가장 부정적인 리뷰 (best_negative_review)와 확률(best_negative_score)
        # 긍정 점수 평균 positive_scores
        # 부정 점수 평균 negative_scores
        
        review_length = len(sentence)
        
        if label =="긍정 😄":
            positive_count += 1
            positive_scores += score
            positive_length.append(review_length)
            
            if best_positive_score < score :
                best_positive_score = score
                best_positive_review = sentence
        else:
            negative_count += 1
            negative_scores += score
            negative_length.append(review_length)
            
            if best_negative_score < score :
                best_negative_score = score
                best_negative_review = sentence
            

        results_text.append([sentence, label, score])
        
    # 총 리뷰수 : 20개
    total = len(reviews)
    # 긍정 리뷰 : 4개
    # 부정 리뷰 : 16개
    
    # 긍정 비율 : 21.05%
    positive_ration = positive_count / total * 100
    # 부정 비율 : 78.95%
    negative_ration = negative_count / total * 100
    
    # 부정 긍정 점수 평균
    avg_positive = positive_scores / positive_count if positive_count > 0 else 0 
    avg_negative = negative_scores / negative_count if negative_count > 0 else 0
    
    # 리뷰 길이 평균
    avg_positive_length = sum(positive_length) / len(positive_length) if positive_length else 0
    avg_negative_length = sum(negative_length) / len(negative_length) if negative_length else 0
    
    stats = f"""
    총 리뷰 수 : {total} 개
    
    긍정 리뷰 😄 : {positive_count} 개
    부정 리뷰 😡 : {negative_count} 개
    
    긍정 비율 😄 : {positive_ration:.2f}%
    부정 비율 😡 : {negative_ration:.2f}%
    
    가장 긍정적인 리뷰 😄 : {best_positive_review} = {positive_length}
                       = (긍정 확률 {best_positive_score:.4f})
    가장 부정적인 리뷰 😡 : {best_negative_review} = {negative_length}
                       = (부정 확률 {best_negative_score:.4f})
    
    긍정 점수 평균 😄 : {avg_positive:.2f}
    부정 점수 평균 😡 : {avg_negative:.2f}
    
    평균 긍정 리뷰 길이 😄 = {avg_positive_length:.2f}
    평균 부정 리뷰 길이 😡 = {avg_negative_length:.2f}
    """
    
    
    #차트
    fig, ax = plt.subplots()
    
    ax.pie(
        [positive_count, negative_count], 
        labels=["긍정", "부정"], 
        autopct="%.1f%%", 
        startangle=90, 
        counterclock=False
        )
        
    return (
        gr.update(value=results_text, visible=True), 
        gr.update(value=stats, visible=True),
        gr.update(value=fig, visible=True),
    )

with gr.Blocks() as demo :
    gr.Markdown("HuggingFace Transformer 기반 감정 분석 프로그램")
    
    inp = gr.File()
    btn = gr.Button("감정분석")
    
    df = gr.Dataframe(headers=["문장", "감정", "확률"], visible=False)
    stats = gr.Textbox(label="분석 결과", visible=False)
    chart_box = gr.Plot(label="감정 비율", visible=False)
    btn.click(fn=predict_sentiment, inputs = inp, outputs = [df, stats, chart_box])


demo.launch()