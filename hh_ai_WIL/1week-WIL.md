# 1주차 WIL: 1주차 WIL: 선형 회귀부터 MLP, 과적합 해결까지


**1\. 이번 주 학습 주제 요약**

이번 주는 딥러닝의 시작점이라 할 수 있는 **선형 회귀(Linear Regression)** 개념부터 출발하여, 이를 다층 구조로 확장한 **MLP(Multi-Layer Perceptron)** 모델까지 실제로 구현하며 실습을 진행했다. 특히 예측 함수, 손실 함수, gradient 계산 방식과 같은 수학적 직관도 함께 학습하면서, 단순한 구현을 넘어선 이해를 키울 수 있었다.

또한 **과적합(overfitting)**을 방지하기 위한 기법들인 **Dropout**, **Weight Decay**, **BatchNorm**, **Validation/Early Stopping** 등의 핵심 개념도 함께 다뤘고, 마지막에는 MNIST 손글씨 분류를 통해 실제 학습 흐름을 구현했다.

---

**2\. 핵심 개념 정리**

**🔹 선형 회귀란?**

가장 기본적인 예측 모델로, 입력 x에 대해 y = wx + b 형태로 결과를 예측한다. 여기서의 목표는 실제값 y와 예측값의 오차를 줄이는 w, b를 찾는 것.

• 예측 함수: pred(w, b, x)

• 손실 함수: loss(w, b, x, y)는 평균 제곱 오차(MSE)를 사용

• 기울기 계산: 오차 × 입력값 → 각 파라미터가 손실에 얼마나 영향을 주는지 반영

**🔹 MLP (Multi-Layer Perceptron)**

선형 회귀만으로는 XOR 같은 문제 해결이 불가능하다. 그래서 **은닉층**을 하나 이상 추가하고, **비선형 함수(ReLU 등)**를 통해 복잡한 결정 경계를 만들 수 있도록 한 것이 MLP이다.

구성:

• 입력층 → 은닉층 1~n → 출력층

• 각 층 사이에는 활성화 함수 (ReLU 등) 사용

• torch.nn.Linear, torch.nn.ReLU 등을 활용

**🔹 활성화 함수와 비선형성**

비선형성이 없으면 여러 개의 선형 함수를 이어붙여도 결국 선형이다. 따라서 ReLU와 같은 비선형 함수를 사이에 넣어야만 더 복잡한 함수를 표현할 수 있다.

• **ReLU**: 0보다 크면 그대로, 작으면 0으로 만드는 함수

• **Leaky ReLU**: 음수도 아주 작게 통과시킴 (기울기 소실 방지)

• **GELU**: Transformer에서 많이 쓰는 확률 기반 활성화 함수

---

**3\. 과적합과 정규화 기법**

**Overfitting이란?**

• 학습 데이터에는 잘 맞지만, 새로운 데이터에서는 성능이 떨어지는 현상

• 원인: 너무 많은 파라미터, 너무 적은 데이터, 반복 학습 등

**해결 방법**

• **Validation 데이터 사용**: 모델이 잘 학습되고 있는지 점검

• **Early Stopping**: validation 성능이 나빠지면 학습 조기 종료

• **Weight Decay (L2 Regularization)**: 가중치를 작게 유지하여 복잡한 모델 방지

• **Dropout**: 학습 시 일부 뉴런을 랜덤하게 꺼서 특정 뉴런 의존도 낮춤

• **BatchNorm**: 각 layer의 출력을 정규화해서 학습 안정화

---

**4\. Optimizer와 학습 방식**

**Adam Optimizer**

• **Momentum**: 이전 gradient 방향을 반영해 가속

• **Adaptive Learning Rate**: 파라미터마다 learning rate를 조절

• gradient가 크면 learning rate 줄이고, 작으면 크게 해서 효율적인 학습 가능

---

**5\. 실습: MNIST 손글씨 분류**

• 28x28 크기의 흑백 이미지 (0~9 숫자)

• torchvision.datasets.MNIST로 데이터 로드

• transforms.ToTensor()로 정규화

• 모델 구성:

```python
class Model(nn.Module):
  def __init__(self, input_dim, n_dim):
    super().__init__()
    self.layer1 = nn.Linear(input_dim, n_dim)
    self.layer2 = nn.Linear(n_dim, n_dim)
    self.layer3 = nn.Linear(n_dim, 10)  # 클래스 10개
    self.act = nn.ReLU()

  def forward(self, x):
    x = torch.flatten(x, start_dim=1)
    x = self.act(self.layer1(x))
    x = self.act(self.layer2(x))
    x = self.layer3(x)
    return x
```

• 손실 함수: nn.CrossEntropyLoss()

• optimizer: Adam(model.parameters(), lr=0.001)

• 학습 epoch마다 정확도 측정:

```python
def accuracy(model, dataloader):
    cnt, acc = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            preds = model(inputs)
            preds = torch.argmax(preds, dim=-1)
            acc += (labels == preds).sum().item()
            cnt += labels.size(0)
    model.train()
    return acc / cnt
```

• 정확도 시각화도 matplotlib로 구현

```python
def plot_acc(train_accs, test_accs, label1='train', label2='test'):
    x = np.arange(len(train_accs))

    plt.plot(x, train_accs, label=label1)
    plt.plot(x, test_accs, label=label2)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy')

    plt.legend()
    plt.grid(True)
    plt.show()
```

---

**6\. 어려웠던 점 & 배운 점**

• CrossEntropyLoss 사용 시 예측값은 softmax 전의 logit 상태여야 하고, 정답은 정수 인덱스여야 한다는 걸 몰라 에러가 났었다.

• Dropout을 왜 학습 때는 켜고 테스트 때는 끄는지 헷갈렸지만, 여러 모델을 실험하고 평균을 취하는 방식이라는 개념이 명확해졌다.

• 학습과정에서 .zero\_grad() → forward → loss.backward() → optimizer.step() 순서가 반복됨을 체득함.

---

**7\. 느낀 점**

• 단순히 구현하는 걸 넘어서, 수학적인 이유와 개념적 원리를 함께 학습할 수 있어 재미있고 유익했다.

• PyTorch의 구성 요소들이 점점 익숙해지고, 각 개념이 어떤 문제를 해결하려고 만들어졌는지를 알게 되니 흥미가 더해진다.

• 앞으로 더 깊은 주제와 실습이 기다리고 있지만, 이번 주의 기반이 큰 도움이 될 것 같다.


#항해99 #항해 플러스 AI 후기 #개발자 커뮤니티 #LLM
