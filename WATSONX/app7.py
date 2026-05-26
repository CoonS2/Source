import gradio as gr  # 1. 'gr' 정의 (NameError 해결)

# 응원 함수
def cheer(name, level):
    return f"{name} 화이팅! " + "🥳" * int(level)

# 별점 함수
def review(name, grade):
    return f"{name} " + "⭐" * int(grade)

# BMI 함수
def bmi_calculator(height, weight):
    if not height or not weight: # 입력값이 없을 때 방어
        return "키와 몸무게를 모두 입력해주세요."
    
    bmi = weight / ((height / 100) ** 2)
    
    if bmi < 18.5: result = "저체중"
    elif bmi < 22.9: result = "정상체중"
    elif bmi < 24.9: result = "과체중"
    else: result = "비만"
    
    return f"결과: {result} (BMI: {bmi:.2f})"

# 인터페이스 구성
with gr.Blocks() as demo:
    gr.Markdown("# 통합 도구 모음")
    
    with gr.Tab("응원"):
        c_name = gr.Textbox(label="이름")
        c_level = gr.Slider(1, 5, step=1, label="응원강도")
        c_out = gr.Textbox(label="결과")
        gr.Button("보내기").click(fn=cheer, inputs=[c_name, c_level], outputs=c_out)
        
    with gr.Tab("별점"):
        r_name = gr.Textbox(label="음식명")
        r_grade = gr.Slider(1, 5, step=1, label="만족도")
        r_out = gr.Textbox(label="리뷰 결과")
        # 2. 여기서 inputs에 r_name과 r_grade 두 개를 정확히 넣어줘야 합니다!
        gr.Button("등록").click(fn=review, inputs=[r_name, r_grade], outputs=r_out)
        
    with gr.Tab("BMI"):
        h = gr.Number(label="키(cm)")
        w = gr.Number(label="몸무게(kg)")
        b_out = gr.Textbox(label="판정 결과")
        gr.Button("계산").click(fn=bmi_calculator, inputs=[h, w], outputs=b_out)

# 3. 마지막에 괄호()를 붙여서 실행
demo.launch()