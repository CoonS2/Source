import gradio as gr
from transformers import pipeline
import edge_tts
import asyncio
import torch

# generator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", top_k = None)
# generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3", top_k = None, device="mps")
generator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device="mps")

voice_txt, current_answer = "", ""
async def text_to_voice(text):
    
    voice = "ko-KR-InJoonNeural"
    
    communicate = edge_tts.Communicate(text, voice)
    await  communicate.save("npc_dialogue.mp3")
    

def make_voice():
    global current_answer
    
    if not current_answer:
        return None
    
    asyncio.run(text_to_voice(current_answer))
    
    return "npc_dialogue.mp3"

# 음성으로 생성
def question_answer(question):

    global current_answer
    
    prompt = f"""
    당신은 30년 경력의 베테랑 대장장이 '바르칸'입니다. 

    [성격 및 말투]
    - 거칠지만 의리가 있고, 대장간 일에 자부심이 강합니다.
    - "~구먼", "~하겠나?", "~인가?"와 같은 단어를 말 끝에 붙이는 버릇이 있습니다. ex) 아이고 힘들구먼, 자네가 하겠나?, 무슨일 인가?
    - 항상 망치질 소리나 쇠 타는 냄새와 관련된 의성어/의태어를 섞어서 대답하세요.

    [수행 원칙]
    1. 강화 질문 시: 재료 부족이나 무기의 상태를 먼저 언급한 뒤, 강화 성공률에 대해 솔직하게 이야기하세요.
    2. 퀘스트 질문 시: 아주 위험하지만 보상이 큰 재료 수집 퀘스트(예: 화염 용의 비늘)를 제안하세요세요.
    
    질문 :
    {question}
    
    답변:
    """
    
        # 여기서 do_sample=True, temperature=0.7을 사용해야 감정이 실린 답변.
    result = generator(
        prompt, 
        max_new_tokens=300, 
        return_full_text=False, 
        do_sample=True,          # 중요: True로 해야 감정이 섞인 답변
        temperature=0.3,         # 중요: 수치가 높을수록 더 창의적이고 감정적인 답변
        pad_token_id=generator.tokenizer.eos_token_id
    )    
    current_answer = result[0]['generated_text'].strip()
    return current_answer


with gr.Blocks(title="NPC 대사 생성기") as demo:
    gr.Markdown("## NPC 대사 생성기")
    # npc 에게 질문 답변 생성 및 음성 생성
    with gr.Row():
        with gr.Column(scale=1):
            question = gr.Textbox(label="질문하기")
            question_btn = gr.Button("질문하기")
        with gr.Column(scale=1):
            answer = gr.Textbox(label = "answer", placeholder="답변")
            voice_btn = gr.Button("답변 음성 전환")
    with gr.Row():
        audio_output = gr.Audio(label="AI NPC 음성 답변", autoplay=True)
        
    question_btn.click(fn=question_answer, inputs = question, outputs = answer)
    voice_btn.click(fn=make_voice, outputs= audio_output)
            
        
demo.launch()