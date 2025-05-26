---
layout: default
title: 8주차 WIL
---

## 8주차 WIL: FlashAttention·DeepSpeed·Quant & LoRA

### 0\. 들어가며

벌써 항해 플러스 AI 과정 8주차. 진짜 시간 순삭이다.

이번 주는 “모델을 더 빠르게, 더 가볍게, 더 싸게” 만드는 네 가지 키워드를 집중적으로 파봤다.

1.  **FlashAttention** – 긴 입력도 쌩으로 돌려서 속도 3배 뽑기
2.  **Data Parallelism & DeepSpeed** – GPU 여러 대를 네트워크로 엮어 학습 속도 뻥튀기
3.  **Quantization** – 파라미터를 16bit → 8bit로 눌러서 메모리 절약
4.  **PEFT(LoRA)** – 모델 전체 대신 저차 ΔW만 학습해서 비용↓ 성능↑

덕분에 “GPT‑급 모델을 노트북 한 대로 fine‑tuning 할 수 있을까?” 같은 로망이 좀 더 현실로 다가온 느낌이다.

---

### 1\. FlashAttention 

Transformer Self‑Attention은 Q·K·V 행렬 계산 때문에 **O(n²)** 메모리/연산량이 터짐. FlashAttention은

-   Q/K/V를 **block**(예: 64×64) 단위로 잘라서
-   GPU **SRAM**에 올린 뒤, 블록별 softmax를 스트리밍 방식으로 계산
-   결과를 다시 HBM으로 쓰는 방식으로

HBM 왕복을 최소화해. 실제로 4K 토큰 → 16K 토큰으로 늘려도 **VRAM 사용량이 거의 평평**했고, wandb 로그 보니까 **throughput 3.2×** 찍혔다.

```
model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=True,
)
```

한 줄이면 끝. 덕분에 긴 문장 요약 태스크에서 latency가 1/3로 줄어서 진짜 체감됨.

---

### 2\. Data Parallelism & DeepSpeed 

-   **DDP(Distributed Data Parallel)** : 파라미터 복사 → 데이터 나눠서 각 GPU가 gradient 계산
-   **DeepSpeed Stage 3** : 파라미터·gradient·optimizer state까지 3‑way 샤딩으로 VRAM 반 토막

```
deepspeed --num_gpus 2 train.py \
   --deepspeed ds_config_stage3.json
```

Stage 3 offload 옵션 켜면 optimizer state가 CPU RAM으로 넘어가는데, NVMe offload까지 물리면 장시간 학습도 무난. 단점은 **통신 병목**인데, `fp16/bp16` 모드와 gradient accumulation으로 어느 정도 해결.

---

### 3\. Quantization 🔧

FP16 → INT8 양자화는 두 가지 맛으로 실험:

1.  **Post‑Training Quantization(PTQ)** : 학습 끝난 모델을 직접 양자화. `bitsandbytes` `bnb.nn.Linear8bitLt` 사용.
2.  **Quant‑Aware Training(QAT)** : 학습 중에 fake quant op 삽입해서 분포 적응.

```
from transformers import BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)
```

실습 과제는 “채팅봇 엔진을 모바일에 올려라”였는데, 양자화 덕분에 APK 크기 90 → 54 MB로 줄여서 통과했다.

---

### 4\. PEFT – LoRA 🪄

모델 전체를 미세조정하려면 파라미터·VRAM·시간이 다 터진다. LoRA는 weight ΔW를 두 저차 행렬 A( n×r )·B( r×m )로 분해해서 **r(n+m)**개만 학습한다는 아이디어.

-   `r=8`, `α=32`, dropout 0.1 설정
-   target modules: 모든 `nn.Linear` (단, `lm_head` 제외)

```
from peft import LoraConfig, get_peft_model

peft_cfg = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32,
                      lora_dropout=0.1, target_modules=target_modules)
model = get_peft_model(base_model, peft_cfg)
```

결과:

-   학습 VRAM 15.3 GB → 4.2 GB
-   학습 속도 1.8× 상승
-   BLEU 점수는 full fine‑tune 대비 −0.4 만 손해. 충분히 trade‑off 수준.

---

### 5\. 실습 & 프로젝트 기록 

| 실습 태스크 | 핵심 지표 | Before | After |
| --- | --- | --- | --- |
| 긴 문서 요약(FlashAttn) | latency (tok/s) | 32 | **99** |
| 13B 모델 학습(DS Stage3) | max batch | 1 | **4** |
| 모바일 챗봇 추론(PTQ) | 모델 크기 | 6.2 GB | **3.7 GB** |
| 삼행시 챗봇 PEFT | 배포 파일 | 13 GB ckpt | **20 MB adapter** |

---

### 6\. 느낀 점 🙏

이번 주는 “스케일링”에 대한 감각을 제대로 익힌 주차였어. “모델은 클수록 좋다”라는 말이 실제론 **리소스·시간·돈**과의 싸움이라는 걸 몸소 체험했고, 그 싸움을 이기는 무기가 바로 FlashAttention, DeepSpeed, Quantization, PEFT 같은 최적화 스킬이라는 걸 깨달음.

특히 FlashAttention 한 줄 옵션이 이렇게 성능을 바꿔놓을 줄 몰랐다. 앞으로 실서비스나 해커톤에서도 **“먼저 켜보고 시작”** 해야 할 기본 옵션으로 자리 잡을 듯.

또, LoRA adapter만 따로 배포해도 충분한 성능이 나온다는 걸 체험해서, 추후 회사 프로젝트에서도 **대형‑언어모델 SaaS → LoRA fork → 자체 서비스** 패턴을 적극적으로 고려할 계획이다.

---

### 7\. 마무리

프로젝트 발표도 했는데, 발표자료를 전날 급하게 만드느라 3시간 밖에 못잤는데 생각보다 발표를 잘해서 뿌듯했다.

많이 성장하는 8주간의 여졍이였다. 앞으로 이러한 이해를 바탕으로 더 크게 성장할 거라 믿는다.


#항해99 #항해플러스AI후기 #개발자커뮤니티 #LLM

