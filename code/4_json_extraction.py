# 4) 구조화 출력(JSON)로 정보추출
import ollama
import json
from slack import slack 

text = """
주문 3건:
1) 상품: 무선마우스, 수량 2, 가격 25,000원
2) 상품: 기계식 키보드, 수량 1, 가격 89,000원
3) 상품: USB-C 케이블, 수량 3, 가격 9,900원
총 배송지는 서울시 강남구 테헤란로 1
"""

prompt = f"""
아래 텍스트에서 주문 항목을 JSON으로 추출해.
스키마:
{{
  "orders":[{{"item":str,"qty":int,"price_krw":int}}],
  "shipping_address": str,
  "total_price_krw": int
}}
텍스트:
{text}
반드시 JSON만 출력.
"""

resp = ollama.chat(
    model='gemma3:4b',
    messages=[{"role": "user", "content": prompt}],
    format='json',  # JSON 모드
    options={"temperature": 0}
)

data = json.loads(resp['message']['content'])
print(json.dumps(data, indent=2, ensure_ascii=False))


# ============================================================================
# 미션: 영화 리뷰 분석 - 복잡한 JSON 구조 추출 연습
# ============================================================================
"""
[미션]
영화 리뷰 텍스트에서 다음 정보를 JSON으로 추출하는 코드를 작성하세요:
- 영화 제목 (title), 감독 (director), 장르 (genre)
- 평점 (rating) - 5점 만점의 숫자
- 장점 리스트 (pros) - 문자열 배열
- 단점 리스트 (cons) - 문자열 배열
- 추천 여부 (recommended) - true/false

아래 예시를 참고하여 다른 영화 리뷰로도 테스트해보세요!
"""

print("\n" + "="*80)
print("🎬 [미션] 영화 리뷰 분석")
print("="*80)

# 예시 영화 리뷰 텍스트
review_text = """
영화 '어쩔수가없다'(감독 박찬욱)가 제69회 런던 국제영화제의 레드카펫과 공식 상영을 성료했다.


'어쩔수가없다'는 '다 이루었다'고 느낄 만큼 삶이 만족스러웠던 회사원 만수(이병헌)가 덜컥 해고된 후, 아내와 두 자식을 지키기 위해, 어렵게 장만한 집을 지켜내기 위해, 재취업을 향한 자신만의 전쟁을 준비하며 벌어지는 이야기를 담는다. 제82회 베니스국제영화제를 시작으로 50회 토론토국제영화제, 제63회 뉴욕 영화제까지 세계 유수 영화제를 휩쓸고 있는 '어쩔수가없다'가 이번엔 69회 런던 국제영화제를 끌어당겼다.


런던 국제영화제에 박찬욱 감독과 이병헌이 참석한 가운데, 지난 15일(현지 시각) 진행된 레드카펫과 공식 상영이 성황리에 마무리됐다. 상영 전 레드카펫에 오른 박찬욱 감독과 이병헌은, 여유 있는 포즈와 훈훈한 미소로 폭발적인 취재 세례에 화답했다. 이어 전 세계 영화 팬들의 셀카와 사인 요청에 응답하며 현장을 뜨거운 에너지로 채웠다.


상영 이후에는 시대를 관통하는 이야기와 박찬욱 감독의 독창적인 연출, 캐릭터와 혼연일체 된 배우들의 열연에 관객들의 환호와 박수갈채가 쏟아졌다. 특히 해외 유수 영화제에서의 공식 상영과 더불어 각국에서 개봉을 이어가는 가운데, 로튼 토마토(Rotten Tomatoes) 리뷰가 60건이 누적된 현재까지도 신선도 100%를 유지하며 외신과 해외 비평가들 사이 열띤 호평이 이어지고 있다.

본문 이미지 - 런던 국제영화제
런던 국제영화제
본문 이미지 - 런던 국제영화제
런던 국제영화제
살롱닷컴(Salon.com)은 "박찬욱의 시각적 우아함은 화면 속에 묘사된 인간의 추락과 상반되어 보이지만, 바로 그 간극이 손에 땀을 쥐게 하는 긴장감을 만들어낸다", 디지털 저널(Digital Journal)은 "가족 드라마와 블랙코미디가 절묘하게 균형 잡힌 시나리오", 댓 해시태그 쇼(That Hashtag Show)는 "긴장과 유머 사이를 능숙하게 오가며, 박찬욱 감독의 작품 중 가장 유머러스하면서도 뛰어난 영화로 손꼽힐 만한 완성도를 보여준다", 스크린아나키(ScreenAnarchy)는 "'어쩔수가없다'는 눈부신 색채감의 미술 디자인, 생동감 넘치는 촬영, 정교하게 구성된 편집을 통해 그 자체로 황홀한 시각적 미학을 완성한다" 등 해외 언론은 긴장감 넘치는 전개 속 아이러니한 유머와 디테일한 설정에 극찬을 아끼지 않았다.


한편 '어쩔수가없다'는 전국 극장에서 상영 중이다.
"""

# JSON 추출 프롬프트
review_prompt = f"""
아래 영화 리뷰 텍스트에서 정보를 추출하여 JSON으로 출력해줘.

스키마:
{{
  "title": str,           // 영화 제목
  "director": str,        // 감독
  "genre": str,           // 장르
  "rating": float,        // 평점 (0.0 ~ 5.0)
  "pros": [str],          // 장점 리스트
  "cons": [str],          // 단점 리스트
  "recommended": bool     // 추천 여부
}}

리뷰 텍스트:
{review_text}

반드시 JSON만 출력하고, 텍스트에서 유추 가능한 모든 정보를 포함해줘.
"""

# Ollama 호출
review_resp = ollama.chat(
    model='gemma3:4b',
    messages=[{"role": "user", "content": review_prompt}],
    format='json',
    options={"temperature": 0}
)

# 결과 파싱 및 출력
review_data = json.loads(review_resp['message']['content'])
print("\n📊 추출된 영화 정보:")
print(json.dumps(review_data, indent=2, ensure_ascii=False))

# 결과를 보기 좋게 출력
slack("\n" + "-"*80)
slack(f"🎬 영화: {review_data.get('title', 'N/A')}")
slack(f"🎥 감독: {review_data.get('director', 'N/A')}")
slack(f"🎭 장르: {review_data.get('genre', 'N/A')}")
slack(f"⭐ 평점: {review_data.get('rating', 'N/A')}/5.0")
slack(f"👍 추천: {'예' if review_data.get('recommended', False) else '아니오'}")

slack(f"\n✅ 장점:")
for i, pro in enumerate(review_data.get('pros', []), 1):
    slack(f"  {i}. {pro}")

slack(f"\n❌ 단점:")
for i, con in enumerate(review_data.get('cons', []), 1):
    slack(f"  {i}. {con}")

slack("="*80)


# ============================================================================
# 추가 연습: 다른 영화 리뷰로 테스트해보세요!
# ============================================================================
"""
💡 연습 과제:
1. 위 코드를 참고하여 좋아하는 영화의 가상 리뷰를 작성하고 JSON 추출을 테스트해보세요.
2. 여러 개의 리뷰를 처리하는 반복문을 만들어보세요.
3. 추출된 JSON 데이터를 파일로 저장해보세요 (json.dump() 사용).
4. 평점이 4.0 이상이고 추천하는 영화만 필터링해보세요.

예시 리뷰 텍스트 (테스트용):
- "봉준호 감독의 '기생충'은 블랙 코미디 스릴러로, 계급 갈등을 예리하게 그려냈다..."
- "제임스 카메론의 '아바타'는 판타지 액션 영화로, 3D 기술이 혁신적이다..."
"""

