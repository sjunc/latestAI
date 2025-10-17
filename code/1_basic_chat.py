# 1) "헬로 LLM" — 가장 기본 대화
import ollama
from slack import slack
resp = ollama.chat(
    model='qwen3:8b',
    messages=[
        {"role": "system", "content": "You are a concise Korean assistant."},
        {"role": "user", "content": "LangChain의 필요성을 한 문장으로 설명해줘."}
    ]
)
slack(resp['message']['content'])

