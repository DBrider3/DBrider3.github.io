---
layout: default
title: 7주차 WIL
---

## 7주차 WIL: GPT Fine-tuning 시 Validation Loss 측정하기

### 1. GPT Fine-tuning에 validation 데이터를 사용하는 이유

기존에 fine-tuning 작업을 진행할 때는 보통 train 데이터만 사용해왔다. 그러나 단순히 train loss만 낮아지는 것만으로는 모델의 성능을 완전히 신뢰하기 어렵다. 실제로 모델을 서비스에 적용하려면, 모델이 과적합(overfitting)되었는지 아닌지, 그리고 성능이 안정적으로 개선되고 있는지를 정확하게 측정해야 한다.

Validation 데이터는 학습 과정에서 독립적으로 평가를 수행할 수 있게 해준다. 이를 통해 학습 중간에 모델이 학습 데이터에만 지나치게 맞춰지고 있지는 않은지를 체크할 수 있다. 이번 주차의 가장 큰 목표는 validation loss를 함께 측정하여, 훈련 성능뿐만 아니라 일반화 성능까지 고려한 fine-tuning 프로세스를 구축하는 것이었다. 이 과정을 통해 단순히 loss를 줄이는 것이 아니라, 실제 환경에서도 잘 작동하는 모델을 만드는 데 필요한 기반을 마련할 수 있었다.

### 2. 데이터 준비와 전처리 과정

이번 fine-tuning 실습에서는 HuggingFace의 wikitext 데이터셋을 사용했다. 이 데이터셋은 공개적으로 활용 가능한 고품질 텍스트 데이터로 구성되어 있으며, 언어 모델 학습에 적합하다. 데이터셋을 불러오고 나서는 train과 validation 세트로 나누어 사용하였다.

HuggingFace의 load_dataset API를 이용해 간편하게 데이터를 가져온 후, 전처리 작업을 수행하였다. 전처리는 크게 두 단계로 구성되었다.

첫 번째 단계는 토큰화(tokenization)다. 텍스트를 모델이 이해할 수 있는 token 단위로 변환하는 과정으로, AutoTokenizer를 이용하여 구현했다. GPT 모델에 맞는 pre-trained tokenizer를 로드하고, 각 문장을 token id로 변환했다.

두 번째는 block 단위로 시퀀스를 구성하는 작업이다. GPT 모델은 길이 제한이 있는 시퀀스를 기준으로 학습하기 때문에, 학습 데이터를 일정 길이(block size = 1024)로 나누는 작업이 필요했다. 이 과정을 통해 모델에 적합한 input 형식을 구성하고, 다음 token 예측을 위한 학습 구조를 만들 수 있었다.

### 3. Trainer 설정과 학습 진행

모델 학습은 HuggingFace의 Trainer 클래스를 이용해서 진행했다. 이 클래스는 학습, 평가, 로깅, 체크포인트 저장 등 다양한 기능을 제공해주어 매우 편리하게 사용할 수 있었다. 이번 실습에서는 기존 코드에 eval_dataset 항목을 추가하여 validation 데이터를 포함시켰고, evaluation_strategy를 "steps"로 설정해 일정 step마다 평가가 진행되도록 설정했다.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)
```

평가 주기는 eval_steps = 500으로 설정하여, 500 스텝마다 eval loss가 기록되도록 하였다. 실시간으로 loss 변화를 확인하기 위해 wandb(Weights & Biases)를 연동하여 학습 로그를 시각화하였다. wandb의 로그 그래프를 통해 train loss와 eval loss의 변화를 동시에 확인할 수 있어, 모델이 학습에 잘 적응하고 있는지를 직관적으로 파악할 수 있었다.

또한 save_steps, logging_steps 등의 설정도 병행하여 모델 저장과 로깅을 일정 주기로 자동화함으로써, 실험의 재현성과 안정성을 높였다. 이 과정에서 wandb 프로젝트 이름을 지정하고 run 이름도 구체화하여 여러 실험 간 비교도 가능하도록 하였다.

### 4. 학습 결과 분석과 시각화

학습이 완료된 후 wandb 로그를 통해 학습 결과를 분석했다. wandb는 매우 강력한 시각화 도구로, train loss와 validation loss의 추이를 선 그래프로 표현해주며, 이를 통해 과적합 여부를 바로 확인할 수 있다.

이번 실습에서의 그래프를 보면, 초반에는 train loss와 eval loss가 모두 감소했으나, 특정 스텝 이후로는 train loss는 계속 줄어드는 반면 eval loss는 정체되거나 약간 증가하는 모습을 보였다. 이는 모델이 training data에 과적합되는 초기 징후일 수 있으며, 이 시점에서 학습을 멈추거나 학습률을 조정하는 것이 바람직하다는 것을 알 수 있었다.

wandb에서 확인한 이번 실습의 주요 로그는 다음과 같다:

- [train/loss wandb 링크](https://api.wandb.ai/links/chodavid-t-ime/eya5szss)

- [eval/loss wandb 링크](https://api.wandb.ai/links/chodavid-t-ime/1wayin1t)

이를 통해 학습이 잘 진행되었는지를 점검하고, 추후 하이퍼파라미터 튜닝이나 모델 구조 변경 시 비교 기준으로 삼을 수 있게 되었다.

### 5. 느낀 점과 배운 점

이번 주의 실습은 단순한 모델 학습이 아닌, "좋은 모델"이란 무엇인가에 대한 고민을 다시금 하게 해준 경험이었다. 특히 train 데이터만으로 판단할 때는 좋아 보이던 모델이, validation loss로 보았을 때는 그렇지 않을 수 있다는 점이 인상 깊었다. 실제 서비스를 생각하면 이러한 객관적 평가지표는 필수라는 생각이 들었다.

또한 wandb 같은 도구의 활용이 얼마나 유용한지 체감할 수 있었다. 단순히 로그만 기록하는 것이 아니라, 학습 흐름을 실시간으로 확인하고, 학습 중간에 조기 중단(Early Stopping)이나 튜닝을 고려할 수 있는 근거를 마련해준다는 점에서 매우 큰 의미가 있었다.

기술적인 면에서는 HuggingFace의 Trainer 클래스를 통한 학습 파이프라인의 단순화가 정말 인상 깊었다. 기존에는 여러 기능을 수작업으로 구현해야 했다면, 이제는 설정 한두 줄만으로 학습, 평가, 로깅까지 가능한 구조를 갖출 수 있었다. 이는 향후 모델 실험과 프로토타이핑을 반복하는 데 있어 매우 큰 시간 절약이 될 것으로 보인다.

### 마무리 및 앞으로의 계획

이제 본격적인 팀 프로젝트가 시작된다. 이번 주까지의 실습을 바탕으로, 나만의 데이터셋과 요구사항에 맞는 GPT 모델을 설계하고 fine-tuning할 예정이다. validation loss와 같은 정량 지표 외에도, 사용자 경험을 고려한 qualitative한 평가도 병행하여 모델의 실용성과 정확성을 균형 있게 고려하려고 한다.

이번 주의 학습은 AI 모델을 "서비스화"하는 데 있어 어떤 요소들이 필요한지에 대해 명확한 방향을 잡을 수 있는 계기였다. validation은 단순한 평가 단계가 아닌, 전반적인 모델 개발 전략의 핵심이라는 점을 몸소 체험했고, 이 교훈은 앞으로의 프로젝트와 커리어에도 큰 자산이 될 것이다.

#항해99 #항해 플러스 AI 후기 #개발자 커뮤니티 #LLM
