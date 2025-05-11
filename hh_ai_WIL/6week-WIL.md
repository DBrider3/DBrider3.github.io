---
layout: default
title: 6주차 WIL
---

## 6주차 WIL: 패션 추천 멀티모달 챗봇 만들기

### 이번 주에 집중한 이유와 목표

지난 5주 동안 LLM을 **“터미널·주피터 노트북 장난감”** 수준에서만 굴려 봤다. 물론 쉘에서 모델을 돌려 보는 것도 재미는 있지만, *사람한테 보여 줄 수 있는 형태*로 만들지 않으면 언제까지나 개인 공부다. 그래서 6주차 미션은 한 줄로 정의했다.

> 전신 사진과 옷 사진 몇 장만 던지면, 착용자의 체형·취향·상황(TPO)을 고려해 코디를 추천하는 챗봇을 **눈으로 볼 수 있게** 만들어 보자!

세부 목표 네 가지

1. **멀티모달 입력** – 이미지 여러 장 + 텍스트를 한 번에 받는다.
2. **대화 문맥 유지** – `st.session_state`로 새로고침해도 컨텍스트 유지.
3. **로컬 테스트 완전 통과** – Tesla T4 환경에서 지연 없이 동작.
4. **다음 주 배포 준비** – 코드 구조·보안키·리사이즈 파이프라인 정리.

멀티모달 프롬프트 자체보다 **이미지 사이즈·메모리 관리**가 훨씬 큰 난관이었다. "모델 성능 ≪ I/O·UX"를 다시 체감.

---

### 구현 과정 자세히 뜯어보기

#### 2‑1. Streamlit UI 설계 – 사용 흐름은 짧게, 피드백은 즉시

* **단계별 업로더**: 전신 → 옷 → <kbd>대화 시작하기</kbd> 버튼.
* **세션 스테이트**: 이미지 바이트와 대화 히스토리를 `st.session_state`에 저장.
* **스트리밍 응답**: `ChatOpenAI(..., streaming=True)` + `for chunk in model.stream([...])` 로 0.5 초 간격 실시간 출력.

#### 2‑2. 멀티모달 프롬프트 – 이미지와 텍스트를 한 그릇에 담기

```python
content = [
    {"type": "text", "text": f"{prompt_tpl}\n\n{history}\n\nUser: {question}"},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{body_img}"}},
] + [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cloth}"}}
    for cloth in clothes_imgs
]
```

* GPT‑4o‑mini Vision이 **base64 Data URI**를 그대로 읽는 점 활용.
* 최근 10개 메시지만 남기는 **슬라이딩 윈도우**로 토큰 폭발 방지.

#### 2‑3. 추천 로직 – ‘왜 이 옷?’을 설명해 주기

1. **체형·톤 분석** – 전신 사진만 보고 착용자 타입 3줄 요약.
2. **아이템 매칭** – 옷 사진의 색·실루엣·소재를 분류해 TPO별 조합.
3. **대안 제시** – 비슷한 계열·다른 색상 아이템 2‑3개 추가 추천.

#### 2‑4. 로컬 테스트 시나리오

| 입력                        | 기대 결과                  | 실제 결과                             |
| ------------------------- | ---------------------- | --------------------------------- |
| 전신 + 옷 2장 + “출근용 데일리룩 추천” | 톤·실루엣 근거 + 상·하·악세서리 세트 | ✅ 정확히 설명 + “비 오는 날엔 이런 재질 추천” 보너스 |

---

### 아직 해결 못 한 문제 & 다음 주 계획

#### 3‑1. 이미지 메모리 최적화(예정)

* `Pillow`로 **512 px 썸네일** 생성 후 인코딩 → 용량 90 %↓ 목표.
* JPG → WebP 변환 및 **URL 링크 전달**도 실험.

#### 3‑2. 클라우드 배포(계획)

* **Nginx Reverse Proxy + Let’s Encrypt SSL** → `https://fashion‑bot.my‑domain.com`
* GitHub Actions → EC2 무중단 배포.

#### 3‑3. RAG & 이미지 생성

* 룩북·쇼핑몰 크롤링 → 임베딩 → GPT + RetrievalQA.
* DALL·E로 **스타일 시뮬레이션 이미지** 생성 → 사용자 피팅.

---

### 배운 점 & 느낀 점

1. **Streamlit만으로 빠르게 MVP**를 만들 수 있었다. 코드 <300줄.
2. **이미지 리사이즈·압축 파이프라인**을 안 짜면 메모리 병목이 금방 온다.
3. **응답 속도 = 품질**. 모델 정확도보다 스트리밍이 먼저 체감된다.

> “AI가 전부 해결해 줄 거야”라는 막연한 기대와 달리, 진짜 서비스 품질은 **인프라·UX·데이터 파이프라인**이 70 % 이상을 차지한다는 걸 깨달았다.

---

### 참고 코드 스니펫

```python
@st.cache_resource
def init_model():
    return ChatOpenAI(model="gpt-4o-mini", streaming=True)

model = init_model()

with st.chat_message("assistant"):
    response_placeholder = st.empty()
    full_response = ""
    for chunk in model.stream([HumanMessage(content=content)]):
        full_response += chunk.content or ""
        response_placeholder.markdown(full_response)

st.session_state.messages.append({"role": "assistant", "content": full_response})
```

* `@st.cache_resource` 로 모델 객체를 한 번만 초기화해 **메모리 절약**.
* 스트리밍 루프에서 `chunk.content` **Null 체크** 필수.

---

### 끝으로

친구에게 직접 보여 주자마자 첫 피드백을 받았다. "사진 업로더가 한 번에 못 올려서 귀찮다"는 의견 → **다중 드래그 & 드롭** 지원 예정. 6주차는 "모델 완성"보다 "보여 줄 수 있는 상태"를 만드는 공정이 얼마나 많은지 체감한 시간이었다. 다음 주엔 실제 배포까지 완주해, "챗봇이 아니라 작은 서비스"로 내놓는 게 목표다.

---

#항해99 #항해 플러스 AI 후기 #개발자 커뮤니티 #LLM
