# 11) 간단 "정답 체크" 루브릭(자체평가) — 채점 프롬프트
import ollama
import json
from slack import slack 

def ask(model, task, system="한국어로 간결하고 정확하게 답해줘.", **options):
    return ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": task}
        ],
        options=options or {}
    )['message']['content']


student_answer = ask('gemma3:4b', "RAG의 정의를 한 문장으로 설명해줘.", temperature=0)

rubric = """
채점 기준:
1) 정의의 정확성(0~4)
2) 간결성(0~3)
3) 핵심 용어 사용(0~3)
총점=10, JSON으로만 출력: {"score": <0-10>, "feedback": "<한줄 피드백>"}
학생 답변:
""" + student_answer

grade = ollama.chat(
    model='gemma3:4b',
    messages=[{"role": "user", "content": rubric}],
    format='json',
    options={"temperature": 0}
)
output_buffer = ""
# 학생 답변 (줄 바꿈 추가)
output_buffer += "학생 답변: " + student_answer + "\n"

# 채점 결과 제목 (줄 바꿈 추가)
output_buffer += "\n채점 결과:\n"

# JSON 형식으로 예쁘게 포맷팅된 문자열을 추가
formatted_grade = json.dumps(json.loads(grade['message']['content']), indent=2, ensure_ascii=False)
output_buffer += formatted_grade

# 3. 화면에 출력합니다. (기존 print 역할)
print(output_buffer)

# 4. 취합된 단일 문자열을 slack 함수에 전달합니다.
slack(output_buffer)
