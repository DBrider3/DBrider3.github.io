---
layout: default
title: 2주차 WIL
---

## 2주차 WIL: 자연어 처리와 딥러닝 구조의 본질을 배운 시간

### 1\. 이번 주 학습 주제 요약

2주차 학습에서는 딥러닝 모델이 자연어 처리(NLP, Natural Language Processing) 분야에서 어떻게 활용되는지를 중심으로 다루었다.  
자연어는 이미지나 숫자 데이터와 달리 순차성과 문맥성을 가지며, 이러한 특성을 반영할 수 있는 모델 구조에 대한 이해가 핵심이었다.  
이번 주차에는 RNN, Attention, Transformer 같은 구조들을 중심으로 개념을 익히고, 이를 활용한 간단한 문장 예측 실습도 함께 진행했다.

---

### 2\. 자연어 데이터의 특성과 입력 전처리

자연어는 단어의 순서와 맥락에 따라 의미가 크게 달라지기 때문에, 기존의 MLP나 CNN과 같은 모델로는 제대로 처리하기 어렵다.  
이를 해결하기 위한 자연어 처리용 딥러닝 모델에서는 다음과 같은 입력 전처리 과정을 거친다.

-   **Tokenization**: 문장을 단어(또는 subword) 단위로 분리하여 토큰 시퀀스로 만든다.
-   **Embedding**: 토큰을 고정된 차원의 벡터로 변환해 신경망에서 처리할 수 있도록 한다.
-   **Padding**: 시퀀스의 길이를 통일하기 위해 짧은 문장에는 padding token을 추가한다.

---

### 3\. RNN과 Sequence-to-Sequence 구조

자연어 시퀀스를 처리하기 위한 대표적인 모델은 **RNN (Recurrent Neural Network)**이다.  
RNN은 문장을 순차적으로 입력받아 이전 정보를 hidden state로 기억하며 다음 입력을 처리한다.  
그러나 긴 문장에서 앞쪽 정보를 잊는 문제(long-term dependency)가 발생하기 때문에 이를 보완한 **LSTM**, **GRU** 등의 구조가 발전해왔다.

또한 번역과 같이 입력과 출력이 모두 시퀀스인 문제에서는 **Encoder-Decoder 구조의 Sequence-to-Sequence 모델**이 사용된다.  
하지만 이 구조 역시 입력 전체를 하나의 벡터로 압축하는 과정에서 정보 손실이 발생할 수 있어, 이를 보완하기 위해 **Attention 메커니즘**이 도입되었다.

---

### 4\. Transformer – Attention Is All You Need

Transformer는 RNN을 대체할 수 있는 구조로, **병렬 연산이 가능하고** **문장 전체의 관계를 더 효과적으로 모델링**할 수 있다.  
Transformer의 핵심은 다음과 같다:

-   **Positional Encoding**: 순서 정보를 반영하기 위해 각 단어 위치에 대해 sin/cos 기반 값을 embedding에 더한다.
-   **Self-Attention**: 문장의 각 단어가 다른 모든 단어와의 관계를 계산하여 문맥을 반영한 벡터를 생성한다.
-   **Feed-Forward Layer**: Attention 결과를 비선형 함수를 통해 더욱 표현력 있는 벡터로 변환한다.

이러한 구조는 이후 등장한 BERT, GPT 등 모든 현대 NLP 모델의 기반이 되었다.

---

### 5\. 실습 요약 – Self-Attention 기반 단어 예측 모델 구현

이번 실습에서는 IMDB 데이터셋을 기반으로 **문장에서 다음 단어를 예측하는 Transformer 기반 모델**을 구현했다.

-   **전처리**: 문장을 토큰화하고, 마지막 3개 토큰 중 하나를 정답(label)으로 지정. 나머지는 입력 시퀀스로 사용.
-   **Self-Attention**: 직접 구현하여 Query, Key, Value 계산 및 softmax attention 연산을 실습.
-   **Masking**: padding token이 학습에 영향을 주지 않도록 mask를 활용해 attention score에서 제외.
-   **Positional Encoding**: 입력 embedding에 위치 정보를 더해 순서를 반영.
-   **전체 구조**: Embedding → Positional Encoding → Transformer Layer Stack → Classification (x\[:, 0\] 사용)

---

#### Self-Attention 구현

```python
class SelfAttention(nn.Module):
  def __init__(self, input_dim, d_model):
    super().__init__()
    self.wq = nn.Linear(input_dim, d_model)
    self.wk = nn.Linear(input_dim, d_model)
    self.wv = nn.Linear(input_dim, d_model)
    self.dense = nn.Linear(d_model, d_model)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x, mask):
    q, k, v = self.wq(x), self.wk(x), self.wv(x)
    score = torch.matmul(q, k.transpose(-1, -2)) / sqrt(self.dense.out_features)
    if mask is not None:
      score += (mask * -1e9)
    score = self.softmax(score)
    result = torch.matmul(score, v)
    return self.dense(result)
```

#### Positional Encoding

```python
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, None], np.arange(d_model)[None, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return torch.FloatTensor(angle_rads[None, ...])
```

---

### 7\. 회고 – 개념을 알고 쓰는 사람이 되기 위해

처음 이 과정을 시작했을 때, 나는 AI를 **활용하는 입장**이었지, 내부 구조나 원리를 깊이 이해하고 있지는 않았다.  
Transformer, Attention 같은 용어들은 모두 처음 듣는 낯선 개념들이었고, 처음엔 이해도 잘 되지 않았다.  
그러나 주차가 거듭될수록 점점 **이해가 되고**, **재미도 느껴지고 있다.**

물론 과제는 쉽지 않았고, 바쁜 일상 속에서 충분한 시간을 할애하기 어려웠던 것도 사실이다. 하지만 하루하루 꾸준히 학습해 나가는 과정은 결국 흔들리지 않는 기반을 쌓는 길이라는 걸 믿고 있다.

무엇보다 이 과정에서 가장 크게 느낀 점은 다음과 같다.  
링크드인에서 읽었던 피드인데 “**이 기술이 왜 필요하고 어떻게 만들어졌는지** 이해하고 사용하는 것”과 “그냥 우와~ 좋다니까 써보자!” 하고 접근하는 것 사이에는 분명한 차이가 있다는 것이다.

이 차이를 아는 개발자가 되기 위해, 그리고 진짜로 AI를 내 도구로 만들기 위해, 앞으로도 개념과 원리를 단단히 다지는 공부를 계속 이어나가고 싶다.




#LLM #개발자 커뮤니티 #향해 플러스 AI 후기 #항해99 #AI #딥러닝 #자연어처리 #Transformer
