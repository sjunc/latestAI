import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
# 1단계: 모델이 호출할 로컬 함수 정의
def get_current_weather(location: str, unit: str = "celsius"):
    """지정된 위치의 현재 날씨 정보를 가져옵니다."""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "15", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "20", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "12", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})
# 2단계: 함수 호출 워크플로우 실행
def run_conversation():
    messages = [
        {"role": "system", "content": "당신은 유용한 날씨 도우미입니다."},
        {"role": "user", "content": "샌프란시스코의 현재 온도를 알려줘. 섭씨로 부탁해."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "지정된 위치의 현재 날씨 정보를 가져옵니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "도시명. 예: San Francisco, Tokyo, Paris"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "온도 단위"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    # 3단계: 1차 API 호출 (모델에게 함수 사용 요청)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = getattr(response_message, "tool_calls", None)
    # 4단계: 모델이 함수 호출을 요청했는지 확인하고 실행
    if tool_calls:
        messages.append(response_message)  # 모델의 응답(함수 호출 요청)을 대화 기록에 추가
        available_functions = {"get_current_weather": get_current_weather}
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            # 5단계: 함수의 실행 결과를 대화 기록에 추가
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        # 6단계: 2차 API 호출 (함수 결과를 모델에게 전달하여 최종 답변 생성)
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return second_response.choices[0].message.content
    else:
        return response_message.content
final_answer = run_conversation()
print(final_answer)


'''
$ python
Python 3.13.5 | packaged by Anaconda, Inc. | (main, Jun 12 2025, 16:37:03) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> from call import *
>>> chat_streaming()
한국의 수도는 서울입니다. 서울은 대한민국의 정치, 경제, 문화의 중심지로, 한반도의  북서부에 위치하고 있습니다. 인구 약 1,000만 명 이상이 거주하며, 세계에서 가장 densely populated (인구 밀도가 높은) 도시 중 하나입니다.

서울은 오랜 역사와 풍부한 문화유산을 가지고 있으며, 경복궁, 창덕궁, 명동, 동대문 디>>>  플라자 등 다양한
>>> exit
>>> from function import *
현재 샌프란시스코의 온도는 섭씨 20도입니다.
'''