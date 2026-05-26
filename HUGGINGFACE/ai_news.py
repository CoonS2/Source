import gradio as gr
from transformers import pipeline

# 요약
summarizer = pipeline("summarization")
# 감성분석
classifier = pipeline("sentiment-analysis")
# 개체명인식(NER)
ner = pipeline("ner", grouped_entities=True)
# 변역
translator = pipeline("translation_en_to_ko", model="facebook/m2m100_418M")

def analyze_news(article):
    
    if not article:
        return "", "", "", ""
    
    # 1. 뉴스 요약
    summary_result = summarizer(article)
    summary = summary_result[0]['summary_text']
    # 2. 감성분석(뉴스원문)
    sentiment_result = classifier(article)
    sentiment = (
        f"감성 : {sentiment_result[0]['label']}"
        f"score : {sentiment_result[0]['score']:.4f}"
        
    )
    # 3. 키워드 추철
    ner_result = ner(article)
    
    keywords = []
    
    for item in ner_result:
        word = item['word']
        
        if word not in keywords:
            keywords.append(word)
    # 리스트 -> 문자열
    keyword_text = ", ".join(keywords)
    # 4. 번역
    translation_result = translator(summary)
    translator_summary = translation_result[0]['translation_text']
    
    return summary, sentiment, keyword_text, translator_summary


with gr.Blocks(title="AI 뉴스 분석기") as demo:
    gr.Markdown("## AI 뉴스 분석기")
    with gr.Row():
        with gr.Column(scale=2):
            article_input = gr.Textbox(label = "영문 뉴스 기사 입력", lines=15, placeholder="영문 뉴스 기사를 입력하세요..",)
            analyze_btn = gr.Button("뉴스 분석 시작")

        with gr.Column(scale=2):
            summary_output = gr.Textbox(label = "뉴스 요약", lines = 5)
            sentiment_output = gr.Textbox(label = "감성 분석")
            keyword_output = gr.Textbox(label = "키워드 추출")
            translation_output = gr.Textbox(label = "한국어 번역 ", lines= 5)
            
    analyze_btn.click(fn = analyze_news, inputs=article_input, outputs=[summary_output, sentiment_output, keyword_output, translation_output])
        
demo.launch()