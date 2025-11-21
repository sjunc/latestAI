### LLM 동작 원리  
  
1. 질문 이해  
토큰 단위로 문장을 쪼갠 후, 각 토큰을 '의미지도' 위의 고유한 좌표(숫자)로 변환. 이를 통해 단어의 의미와 관계를 이해  
  
3. 문맥 파악  
'어텐션'을 통해 문장 전체를 보고 단어 간의 관계를 파악  
  
5. 한 단어씩 답변 생성  
이해한 맥락을 바탕으로, 다음에 올 가장 확률 높은 단어 한 단어씩 예측해 문장 완성.

### LLM 상용과 무료  
  
|구분 | Open LLM(무료/ 오픈소스) | 상용 LLM (유료/ 폐쇄형) | 
|------|---|------|
|대표 모델|LLaMA 2/3 (Meta), Mistral, Falcon, BLOOM, Gemma, DeepSeek, Qwen 등|GPT-4 (OpenAI), Claude (Anthropic), Gemini (Google), Grok (xAI), HyperCLOVA X (Naver) 등|
|개발 주체|연구기관, 빅테크, 오픈 커뮤니티(Meta, HuggingFace 등)|대형 기업 (OpenAI, Google, Anthropic, Microsoft, Naver 등)| 
|공개 여부|모델 가중치, 아키텍처 대부분 공개| 모델 내부 구조/가중치 비공개|
|비용|무료 사용 가능 (연구, 개인/기업 자유롭게 활용)|API 사용 시 과금 (토큰 단위 요금)|
|성능 수준|GPT-3.5 ~ GPT-4에 근접,빠르게 발전 중|현재 최고 성능 (복잡한 추론, 멀티모달 지원 등)|
|확장성/자유도| 파인튜닝, 로컬 실행, 커스터마이징 자유로움 | API 중심, 커스터마이징 제한적|
|하드웨어 요구사항|대형 모델은 GPU 다수 필요, 소형 모델은 PC에서도 실행 가능 |클라우드 제공 (사용자 하드웨어 불필요)|
|한국어 지원|일부 모델은 제한적 (영어/중국어 최적화가 많음), 한국어 튜닝 필요|GPT-4, Gemini → 한국어지원 우수 /HyperCLOVA X → 한국어 특화|
|활용 분야|연구, 실험, 교육, 기업 내 폐쇄망 서비스|상용 서비스, 대규모 제품 적용, 최고 성능 필요한 분야|
|장점|무료, 투명성, 연구와 학습에 적합, 로컬에서 데이터 보안 보장|최고 성능, 안정성, 유지보수/업데이트 제공, 다양한 언어와 기능 지원|  
|단점 | 성능 격차(아직 GPT-4보다 낮음), 실행에 고성능자원 필요|비용 발생, 모델 내부 불투명, 특정 기업 의존|  
  
### LLM 성능 평가(Benchmark)   
대표적 예시: GPQA 질문(기본 질문 데이터셋)을 통해 정확도를 보고 성능 평가 가능   
1. 왜 벤치마크가 필요한가?    
LLM(대규모 언어 모델)은 매우 복잡한 블랙박스처럼 보입니다. (수십억~수천억 개의 파라미터, 방대한 데이터)    
모델이 실제로 얼마나 잘 이해하고, 추론하고, 대답하는지를 객관적으로 알기 어렵습니다.   
그래서 벤치마크(benchmark) 라는 표준 시험지가 필요합니다.    
- 모델을 동일한 조건에서 시험 봐서 성능 비교  
   
2. 벤치마크가 평가하는 것   
언어 이해: 문장 의미 파악, 독해 능력 (GLUE, SuperGLUE)   
지식/상식: 일반 지식 문제 풀이 (MMLU, TriviaQA)   
수학/논리 추론: 계산, 논리 퍼즐 (GSM8K, MathQA)   
코딩 능력: 코드 작성/디버깅 (HumanEval, MBPP)   
멀티모달: 이미지+텍스트 이해 (MMBench 등)  
즉, 벤치마크는 모델의 "과목별 성적표" 역할을 함.  
  
3. 왜 연구자와 개발자에게 중요한가?  
공정한 비교: GPT, Claude, LLaMA 같은 모델을 같은 시험지로 비교해야 성능 차이를 명확히 알 수 있음.    
연구 발전: 새로운 모델이 나왔을 때 “이전 모델보다 얼마나 나아졌는가?”를 증명하는 근거 제공.  
실제 적용성 판단: 예를 들어, 코딩 보조에 쓸 모델을 찾는다면 HumanEval 점수를 보고 선택 의학 분야라면 MedMCQA 같은 전문 벤치마크 점수 참고  
약점 발견: 어떤 모델이 언어 이해는 뛰어나지만 수학은 약한지 파악 가능 → 개선 방향 설정   

| 평가 항목 | 설명 | 측정 방법 |
|------|---|------|
|응답 속도(Latency) | 질의에 대한 평균 시간(ms) |API|
|처리량(Thoughput)| 초당 처리 가능한 토큰 |------|
|정확성 (Accuracy)| 정답률, 평가 데이터셋에서의 정밀도|BLEU, ROUGE, GPT-4 평가 결과 비교 |
|일관성(Consistency)| 동일한 입력에 대해 결과가 얼마나 일관적인지 |다회 테스트 결과 비교 |
|메모리 사용량 (Memory Usage)|모델이 실행되는 동안 차지하는 RAM 및 VRAM| 시스템 모니터링 (온프레미스인 경우)|
|스케일링 가능성| 부하가 증가할 때 성능이 어떻게 변화하는지 |동시 요청 증가 테스트|
| 다국어 지원 | 다국어(한국어) 지원 여부|다국어 응답 결과 테스트, 가이드 참조|  
   
4. TPS(Tokens per Second)     
모델이 초당 생성하거나 처리할 수 있는 토큰 수.    
언어 모델(LLM)이나 API가 얼마나 빠르게 텍스트를 처리할 수 있는지를 측정하는 성능 지표입니다.    
   
TPS =  총 처리된 토큰 수 / 걸린 시간(초)   
  
## OLLAMA  
로컬 환경에서 대규모 언어 모델(LLM)을 구동  
Ollama는 LLaMA 3, Mistral, Gemma와 같은 오픈 소스 대규모 언어 모델(LLM)을 개인 컴퓨터에서 손쉽게 구동하고 관리할 수 있도록 설계된 도구입니다. Docker가 컨테이너를 다루는 방식을 추상화 했듯, Ollama는 LLM을 다루는 복잡한 과정을 추상화하여 개발자가 모델 자체에 집중할 수 있도록 돕습니다.   
역사   
1. '오픈웨이트' 형태 모델 부족이 아닌 거대한 모델을 일반 PC에서 실행하는가? 로 바뀌다.    
2. 'llama.cpp' 효율적인 추론엔징 등장, 최소한의 자원으로 구동 가능하지만 진입장벽  
3. ollama의 등장: 복잡한 과정 없이 추상화된 고급언어를 사용하듯 LLM을 다룰 수 있게 됨.  

장점  
프라이버시 및 데이터 보안  
- 완전한 오프라인 환경  
- 민감한 데이터나 기업의 코드를 외부 서버로 전송하지 않고 LLM을 활용  
- 데이터 보호  
- 모든 정보가 로컬 머신 내에서만 처리되어 보안에 매우 유리  
비용 절감 효과
- 클라우드 LLM 서비스
- 사용량에 따라 지속적으로 비용이 발생
- Ollama 사용 시 하드웨어만 준비되어 있다면 추가 비용 없이 무제한으로 모델 사용 가능
지연 시간 감소
- 네트워크 왕복 시간(Round-trip-delay)
- 클라우드 API 데이터가 원격 서버로 이동하고 다시 돌아오는 데 걸리는 시간은 실시간 상호작용이 중요한 애플리케이션, 예를 들어 대화형 챗봇이나 코드 자동 완성 도구의 사용자 경험을 저하시킬 수 있다. (하드웨어 성능에 달려있음.)
오프라인 환경에서의 사용
- 인터넷 연결 없이도 LLM 구동 가능
- 비행기 내에서도 개발 작업 지속
- 원격지에서 연구 및 개발 수행
- 네트워크 제한 환경에서 AI 활용
- 언제 어디서나 AI의 힘을 활용할 수 있는 자유로움
모델 커스터마이징 및 제어
- Modelfile을 통해 모델을 자유롭게 수정하여 특정 작업에 최적화할 수 있습니다  
Modelfile : 나만의 LLM 만들기  
Ollama 사용자는 `Modelfile`을 통해 모델을 정의하고 커스터마이징할 수 있습니다. 이는 특정 목적에 최적화된 맞춤형 모델을 생성하는 강력한 기능입니다.  

OLLAMA-LLAMA.CPP
Ollama는 고성능 llama.cpp 엔진을 기반으로 한 사용자 친화적 래퍼(Wrapper)입니다.
자체 기술 개발 대신, 검증된 최고 성능의 오픈소스 엔진을 채택하여 속도와 호환성을 확보했습니다.

|CPU 우선 설계 (CPU-first Design) | GPU 가속 지원 (GPU Acceleration) |정수 양자화(Quantization) | 
|------|---|------|
|CPU에서 최대 효율을 내도록설계되었으며, ARM NEON(Apple Silicon) 및 AVX/AVX2(x86) 같은 특수 명령어 세트를 활용해 연산을 가속합니다|NVIDIA (CUDA), Apple(Metal), AMD (ROCm) 등 다양한 GPU 하드웨어 가속을완벽하게 지원하여 추론 속도를 극대화합니다.|모델의 가중치를 저정밀도 정수로 변환하여 메모리 사용량을 줄이고 추론 속도를 향상 시킵니다|  
   
명령어 처리 절차   
1. 클라이언트 요청
2. API 수신
3. 모델 관리
4. 모델 로딩
5. 추론 엔진 호출
6. 생성 루프
7. 스트리밍 응답(실시간)
8. 리소스 정리(자동)
  
가능한 이유  
GGUF 파일 형식과 양자화 기술  
### 양자화 (Quantization): 모델을 가볍게  
양자화는 LLM의 '무게', 즉 모델을 구성하는 수치(가중치)의 정밀도를 낮추어 파일 크기를 줄이고 계산 속도를 높이는 최적화 기술입니다.  
원리:  
기존 AI 모델은 매우 정밀한 소수(예: 32비트 부동소수점)를 사용하여 복잡한 정보를 저장합니다.  
양자화는 이 정밀한 소수를 더 낮은 정밀도의 정수(예: 4비트, 5비트)로 변환합니다.  
비유 : 고화질 원본 사진(32비트)을 화질 저하를 최소화하면서 용량이 작은 JPG 파일(4비트)로 압축하는 것과 같습니다. 약간의 정보 손실이 있을 수 있지만, 크기가 대폭 줄어들고 로딩 속도가 빨라집니다.  
ex) FP32 값: 0.123456789 -> INT8 로 양자화: 0.12 (숫자의 비트수를 줄임)  
  
필요한 VRAM 크기  
파라미터 개수 × 파라미터당 바이트 수 = 가중치 메모리  
Ex) 70억(7B) 파라미터 모델  
FP16 가중치를 사용한다면  
7,000,000,000 params × 2 bytes = 14,000,000,000 bytes ≈ 14 GB(10^9 기준)  
4비트로 양자화된 버전(Q4)  
7,000,000,000 params × 0.5 bytes = 3.5 GB(10^9 기준)  
메모리 사용량 감소: 모델이 차지하는 RAM과 VRAM의 크기를 획기적으로 줄여줍니다.  
추론 속도 향상: 단순화된 숫자는 CPU와 GPU에서 더 빠르게 계산될 수 있습니다.  
접근성 증대: 이 기술 덕분에 수십 기가바이트에 달하는 대규모 모델도 일반 노트북이나 데스크톱에서 실행할 수 있게 됩니다.  
### GGUF  
로컬 모델의 보편적 형식   
LLM의 빠른 로딩과 효율적인 추론을 위해 특별히 설계된 바이너리 파일 형식   
단일 파일(Single File): 모델의 구조, 양자화된 가중치, 토크나이저(문장을 숫자로 변환하는 규칙) 등 모델 실행에 필요한 모든 것을 하나의 파일에 담고 있습니다.  
이식성(Portability): GGUF 파일 하나만 있으면 어떤 환경에서든 동일한 모델을 쉽게 실행할 수 있습니다. 복잡한 설정이나 여러 파일을 다운로드할 필요가 없습니다.  
확장성(Extensibility): 새로운 모델 아키텍처나 기술을 쉽게 추가할 수 있도록 유연하게 설계되었습니다.  
#### 모델 읽는 법
gemma3:27b-it-qat  
모델명 젬마 3: 270억 파라미터를 가진 Instruction-Tuned Quantization-Aware Training  
it  
• Instruction-Tuned의 약자  
→ 원래의 사전학습(pretrained) 모델을, 사람이 쓰는 지시문(Instruction) 형식에 잘 맞게 추가 학습시킨 버전    
예: 질문-답변, 요약, 번역 같은 지시형 프롬프트에 잘 응답하도록 조정됨  
qat  
• Quantization-Aware Training의 약자  
→ 단순히 학습이 끝난 모델을 양자화(post-training quantization)하는 게 아니라, 훈련 중에 양자화를 고려하여 학습시킨 방식  
장점:  
모델을 8비트, 4비트 등으로 줄여도 정확도가 크게 떨어지지 않음   
더 효율적인 실행 가능 (메모리 절약 + 속도 향상)   

## 3주차 OpenAI API 
  
OpenAI API는 개발자가 복잡한 머신러닝 인프라 없이도 자신의 서비스에 손쉽게 통합할 수 있도록 지원하는 강력한 도구  
간편함: 몇 줄의 코드로 AI 기능 추가  
강력함: 최첨단 모델을 즉시 활용  
유연성: 챗봇, 콘텐츠 생성, 코드 분석 등 무한한 가능성  

### API 사용을 위해
API 키 (Key)  
서비스 접근을 위한 비밀번호. 보안이 가장 중요  
모델 (Model)   
목적에 맞는 적합한 모델 선택  
토큰 (Token)   
API 사용량과 비용을 고려  
  
GPT(텍스트, 비전), DALL-E 3(이미지), Whisper(음성-텍스트), gpt-oss-*(오픈 모델)  

tokenizer(모델마다 다름)에 따라 비용 및 성능에 영향을 줌  

API 키 보안  
키 공유 X  
프론트엔드 코드(JS)에 노출 X  
Git에 커밋 X  
**환경 변수 사용(.env 파일)**  
자체 백엔드 서버 통해 api 호출   
키 관리 서비스(KMS) 사용 고려  
  
- git bash    
- >>> python  
- >>> from call( py 파일 명) import *  
- >>> chat_streaming()  
  
|매개변수 | 설명 | 역할 및 특징| 
|------|---|------|
|model 필수 | 사용할 모델 ID 지정 (예: "gpt-4o")|AI의 '두뇌'를 선택하는 가장 기본적인 설정|
|message 필수| 대화의 맥락을 형성하는 메시지 객체 배열 | 'system', 'user', 'assistant' 역할로 대화 흐름 제어|
|temperature| 답변의 창의성/ 무작위성 조절(0.0 ~ 2.0)| 낮을수록 일관적, 높을수록 창의적인 답변 생성|
|max_tokens| 생성될 답변의 최대 길이(토큰) 제한| 비용 관리 및 응답 길이 제어|
|top_p| 핵 샘플링(Nucleus Sampling) 방식| 'temperature' 와 함께 무작위성 제어 (하나만 조정 권장) |
|stream|'True' 설정 시, 응답을 실시간 스트림 형태로 전송 |사용자 경험 (UX) 향상 |

json 객체 반환
|키(key) | 타입 (Type) | 설명 | 
|------|---|------|
|id, object, created, model|문자열, 정수|요청과 응답에 대한 고유 식별자 및 메타데이터|
|choices |배열(Array)|모델이 생성한 응답(들)을 담는 가장 중요한 부분입니다. 실제 텍스트는 **choices[0].message.content** 경로로 접근합니다.|
|usage|객체 (Object) |요청에 사용된 토큰 수를 알려줍니다. - 'promt_tokens': 입력 토큰 수 -'completion_tokens': 출력(응답) 토큰 수 - 'total_tokens': 총 사용 토큰 수|

- 'usage' 객체는 비용 청구 및 사용량 추적의 핵심 기준
  
messages 배열   
system: ai 모델의 전반적인 역할 설정  
user: 사용자의 질문
assistant: 이전 AI의 답변(대화 맥락 유지)  
대화의 종류  
1: 단일 턴(Single-Turn) 한 번의 질문과 한 번의 답변으로 끝나는, '기억'이 없는 독립적인 상호작용   
2: 다중 턴(Multi-Turn) 여러 번의 질문과 답변이 오가며, 이전 대화의 맥락을 계속 이어가는 '기억'이 있는 대화  
멀티 턴의 트레이드오프 (장단점)  
토큰 증가로 비용증가, 컨텍스트 창 제한 문제(한번에 기억할 수 있는 양)  

### 함수 호출(Tool use) 
언어 모델이 외부 함수나 API를 호출하여, 실시간 데이터 조회나 외부 서비스 제어등 실제 세계와 상호작용 가능한 추론엔진으로 변화시킴  
순서  
1. 함수 정의와 함께 호출  
2. 모델의 함수 호출 요청  
3. 함수 실행 및 결과 전달   

날씨 찾기 일 때 예제  
```python
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
                            "enum": ["celsius", "fahrenheit"],  # 화씨 혹은 섭씨 사용할지 
                            "description": "온도 단위"
                        }
                    },
                    "required": ["location"] # 꼭 필요한 정보 "required" 
                }
            }
        }
    ]
    # 3단계: 1차 API 호출 (모델에게 함수 사용 요청)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",   # tool_choice = "auto" ai에게 tool 선택 자유를 줌  
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
```
두번의 API 호출이 필요  
1. 1단계: 모델에 함수 정보와 함께 요청
2. 2단계: 애플리케이션에서 함수 실행
3. 3단계: 실행 결과를 모델에 다시 전달 

### AI 대화 관리 기법

레벨 1. 슬라이딩
윈도우가장 오래된 메세지를 삭제해 항상 최근 n개의 기억하는 가장 간단한 기법
- 장점: 구현이 매우 간단하고 비용 예측 및 제어가 쉬움.  
- 단점: 대화 초기 중요 내용을 잊음  
레벨 2. 요약(summarization)  
오래된 메시지를 ai를 이용해 요약본으로 압축해 핵심 맥락을 보존하는 방식
- 장점: 긴 대화의 맥락 핵심을 보존할 수 있음.  
- 단점: 요약을 위한 추가 API 호출 비용 및 지연 발생, 요약의 품질이 전체 대화 질에 큰 영향  
레벨 3. 검색 증강 생성 (RAG)  
외부 데이터베이스에서 현재 질문과 가장 관련성 높은 정보만 검색하여 ai의 컨텍스트에 주입하는 기법
- 장점: 컨텍스트 창 한계를 넘는 거의 무한한 기억력, 매우 높은 토큰 사용 효율성, 환각 감소 및 출처 제시  
- 단점: 구현이 가장 복잡하며 추가 기술 스텍이 필요  
속도 제한 탐색 및 오류 처리  
### OPEN LLM 
ollama API 개념 및 실습 가이드  
발전시키면 좋을 기술  
RAM: 모델 크기에 따라 용량 결정  
1. 7b 모델: 최소 8gb 권장  
2. 13b 모델: 최소 16gb 권장  
3. 33b+ 모델: 최소 32gb 이상 권장
저장 공간: 여러 모델 저장을 위해 최소 10GB ~ 50GB의 여유 공간으로 시작하는 것을 권장
GPU 가속: 최적의 성능을 위해 NVIDIA (CUDA) 또는 AMD 계열(자동 관리 리소스 관리 필요 X)  

ollama --version  
ollama run <모델명>  
로컬에 없으면 모델을 자동으로 다운로드함  

기존의 파인튜닝(하드웨어/시간/학습에서의 어려움 존재)  
/save 프롬프트를 이용해서 조금씩 건들고 있음  

파일 입출력  
파이프라이닝  
  
ollama serve(linux환경에선 설정해줘야 백그라운드에서 실행)  
openllm은 
ollama show 명령어를 통해 파라미터 수, 아키텍쳐, 라이센스 등의 정보를 확인 가능  
기본 주소 localhost:11434  
응답(스트림)  
json 형태 객체 순차 반환  
배치 사이즈(batch size)  
모델이 한 번에 처리하는 데이터 샘플의 개수  
배치 크기 1: 문제 하나 풀고, 바로 정답 확인  
배치 크기 10: '평균적인 경향' 을 파악하여 지식을 보강  
  
Large Batch Size: 안정적인 학습 경로, 한번에 많은 데이터를 보기 때문에 노이즈가 적고, 학습 방향이 비교적 정확하고 안정적  
small Batch size: 적은 데이터로 자주 업데이트, 노이즈가 많아 학습 방향이 흔들릴 수 있음.  
큰배치: 안정적, 무난함 -> 빠르지만 덜 창의적  
작은 배치: 불안정, 다양한 길 -> 느리지만 더 일반화될 가능성  
ollama의 라이브러리: api 형태로 저장되어 있음  
임베딩 api 지원  

### LLM 프롬프트 엔지니어링  
3가지 핵심 기법  
1.제로샷 프롬프트 (Zero-shot)  
어떤 사전 예시도 제공하지 않고, 모델이 가진 지식을 믿고 바로 작업을 지시하는 가장 간단한 방식  
2. 원샷 프롬프트 (One-shot)
3. 퓨샷 프롬프트 (Few-shot)  
두 개 이상의 여러 예시를 제공하여 작업의 패턴을 충분히 학습시키는 가장 정교한 방식  
고급 기법: 사고의 사슬(CoT: Chain-of-Thought)  
모델에게 "정답만 말하지 말고, 중간 사고 과정을 적어가며 단계별로 풀라" 고 지시하는 프롬프트 기법  
모델명 뒤에 _cot가 붙기도 함  

핵심 요소  
프롬프트 형식  
컨텍스트 및 예시  
미세 조정 및 적응  
멀티턴 대화  

In-Context Learning (ICL)  
별도의 추가 학습(Fine-tuning) 없이, 프롬프트 내에 주어진 정보와 예시만으로 새로운 작업을 수행하는 능력   
제로-샷 CoT  
CoT를 직접 길게 입력하기 보다 "step by step", "단계별로 생각해줘" 라는 말이 더 효과가 좋았었단 결과   
Auto CoT  
샘플 데이터 질문 클러스터링  
단계별로 생각해보라는 말을 추가  
  
자기 일관성(Self-Consistency)  
하나의 질문에 대해 여러 다른 생각의 경로를 만들게 한 뒤, 그중 가장 많이 나온 답을 최종 정답으로 채택하는 프롬프트 기법  
같은 질문 반복 -> 다른 경로를 통해 나온 결과 추합 -> 가장 많은 정답 선택  
장점: 높은 정확도 및 안정성 향상  
단점: 비용 및 시간 증가  
  
검색 증강 생성(RAG: Retrieval-Augmented Generation)  
LLM이 외분의 최신 또는 비공개 데이터베이스를 참고해 더 정확하고 신뢰성 높은 답변 생성을 만드는 기술   
  
적대적 프롬프팅(Adversarial Prompting) 공격  
규칙을 이해하는 게 아닌 '패턴'을 따른다는 취약점 이용  
안전장치가 인식하는 유해 패턴을 교묘하게 비껴가는 새로운 패턴을 생성  
  
탈옥(Jailbreaking)  
가장 널리 알려진 기법, 모델에게 가상의 역할이나 시나리오 부여하여 안전 규칙 무시  
프롬프트 주입(Prompt Injection)   
사용자가 제공한 텍스트(예:문서) 속 악의적 명령어를 숨겨두는 공격, 숨겨진 명령어 원래 지시보다 우선 실행  
  
고급 공격 기법 및 방어 전략   
어휘 위장: 유해단어를 교묘하게 다른 어휘로 바꾸는 기법.    
다중 턴 확전(Multi-turn Escalation): 여러 번의 대화를 통해 점진적으로 유해한 요청으로 나아가는 방식.  
함의 연쇄: 직접적인 요청 대신 암시를 통해 유도하는 기법.  
적대적 훈련: 다양한 공격 프롬프트 예시를 모델에 학습  
프롬프트 변환: 입력 프롬프트의 의미는 유지하되 구조를 변경하여 악의적 패턴을 무력화  
  

### LLM 평가 처리 방식 (lm-eval)  
1. 엄격한 일치(strict-match): 모델이 생성한 답변이 정답과 완벽하게 동일해야 정답으로 처리하는 방식. 공백, 대소문자, 특수문자까지 모두 일치 사용처: 객관식 문제  
2. 유연한 추출(flexible-extract): 모델이 생성한 전체 답변 텍스트 안에서 정답에 해당하는 내용이 포함되어 있는지 확인하는 방식, 보통 정규화 후 비교

## 5주차 Slack 연결 

### Linux 원격 연결 SSH.  
echo $ 와 함께 환경변수 확인 할 때 씀   
man --help 정보 알려줌.   
history: 이전 명령어 확인  
ls 현재 파일 목록  
pwd 현재 작업 디렉토리 절대경로    
cd 디렉토리 이동  
/ 로 시작하면 절대경로   
### cd 
~ 홈 디렉토리   
.. 부모 디렉토리.  
. 현재 디렉토리  
/ 루트 디렉토리   
### ls
-l : 긴 포맷(long)으로 권한, 소유자, 크기 등 상세 정보를 표시합니다.  

-a : 숨김 파일(.file)을 포함한 모든(all) 파일을 표시합니다.  

-h : 파일 크기를 사람이 읽기 쉬운(human-readable) 단위(K, M, G)로 표시합니다.  

-t : 수정된 시간(time) 순서대로 최신 파일부터 정렬합니다.  
mkdir 디렉토리 생성,  touch 빈 파일 생성  
cp [원본] [대상] 복사  
mv 이동, 이름 변경  
rm 삭제  
cat (Concatenate)  
파일의 전체 내용을 터미널 화면에 빠르게 출력하는 명령어  

파일 편집 Vi 와 nano. 
vi. 
vi는 '명령 모드'와 '입력 모드'가 분리.   
명령 모드.   
기본 상태. 커서 이동, 삭제, 복사, 붙여넣기 등 '명령' 수행.   
입력 모드.   
텍스트 입력 및 수정. i, a, o 키로 진입. 
명령행 모드.  
파일 저장, 종료, 검색 등 고급 명령. : 키로 진입.   
패키지 관리지: dnf.   
centOS의 후계자 레드햇 계열 dnf.  
rocky linux.    
우분투에선 apt 사용    
  
## langchain & n8n  
#### 둥장배경   
외부 데이터 접근 어려움  
복잡한 워크플로우 설계    
문맥 유지 힘듦   
상업용 사용은 엄격히 막음   
#### 철학  
1. 데이터 인식 (Data-aware)   
LLM을 최신의, 또는 비공개 외부 데이터 소스에 연결하여 더 유용하고 신뢰성 높은 답변을 생성합니다.   
  
2. 행위 주도 (Agentic)   
LLM이 스스로 추론하고, 행동을 계획하며, 도구를 사용하여 복잡한 작업을 자율적으로 해결하는 '에이전트'가 되도록 합니다.    
 
### n8n  
노드(Node) 기반 워크플로우 자동화 도구입니다.  

다양한 웹 서비스와 애플리케이션을 코딩 없이 시각적으로 연결하여, 반복적인 작업을 자동화하고 복잡한 프로세스를 구축할 수 있게 도와줍니다.  

상업적 재판매나 n8n의 클라우드 버전과 직접 경쟁하는 서비스를 만드는 것에는 제한이 있는 Fair-Code 라이선스  

n8n 워크플로우 템플릿[https://n8n.io/workflows/]  


## RAG

langchain  
다양한 소스들 지원 웹페이지, pdf, csv, 데이터베이스  
데이터 변환 및 정제 기능  
외부 모듈 연동 langchain_community에 위치함  

directoryLoader를 통해 다양한 형식의 파일을 폴더로 가지고 올 수 있음  
csv는 한줄이 하나의 문서  
이렇게 형식마다 인식하는 게 다름  
Text Splitter: 긴 문서 처리의 핵심  

tokenizer -> embedding -> 임베딩 모델별로 데이터마다 결과가 다르기 때문에 적당한 모델을 선택 -> 벡터 저장소 저장 (Faiss, Chroma, Elasticsearch, Pinecone) -> 검색도구  



----
