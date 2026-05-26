import gradio as gr
from transformers import pipeline
import edge_tts
import asyncio

asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

voice_txt, current_answer = "", ""
async def text_to_voice(text):
    
    voice = "ko-KR-InJoonNeural"
    
    communicate = edge_tts.Communicate(text, voice)
    await  communicate.save("answer.mp3")
    

def make_voice():
    global current_answer
    if not current_answer:
        return None
    
    asyncio.run(text_to_voice(current_answer))
    
    return "answer.mp3"


def change_txt(file):
    
    global voice_txt
    
    result = asr(file, return_timestamps=True)
    voice_txt = result['text']
    return voice_txt

def question_answer(question):
    """_summary_
    
    question 답변 생성 후 리턴
    음성내용 -> 텍스트 변환한거 보내주고, 질문도 보내주고
    
    Args:
        question (_type_): _description_
    """
    
    global voice_txt, current_answer
    
    if not voice_txt:
        return "음성을 텍스트로 변환한 후 질문하세요."
    
    prompt = f"""
    다음 음성 내용을 참고해서 질문에 답변하세요
    음성내용 :
    {voice_txt}
    
    질문 :
    {question}
    
    답변:
    """
    
    result = generator(prompt, max_new_tokens=50, return_full_text = False, do_sample= False, pad_token_id=generator.tokenizer.eos_token_id)
    current_answer = result[0]['generated_text'].strip()
    return current_answer

with gr.Blocks(title="AI 음성 챗봇") as demo:
    gr.Markdown("## AI 음성 비서")
    with gr.Row():
        with gr.Column(scale=1):
            file = gr.Audio(type="filepath")
            txt_btn = gr.Button("텍스트 변환")
        with gr.Column(scale=1):
            out = gr.Textbox(label = "텍스트 변환", lines=3)
            
    with gr.Row():
        with gr.Column(scale=1):
            question = gr.Textbox(label="질문하기")
            question_btn = gr.Button("질문하기")
        with gr.Column(scale=1):
            answer = gr.Textbox(label = "answer", placeholder="답변")
            voice_btn = gr.Button("답변 음성 전환")
    with gr.Row():
        audio_output = gr.Audio(label="AI 음성 답변", autoplay=True)
        
    txt_btn.click(fn=change_txt, inputs=file, outputs=out)
    question_btn.click(fn=question_answer, inputs = question, outputs = answer)
    voice_btn.click(fn=make_voice, outputs= audio_output)
        
demo.launch()