---
title:  "[Math] Conjugate Gradients"
date:   2020-03-01 12:04:36 +0900
categories: [Math]
---

이 포스팅에서는 $$Ax = b$$의 해를 찾아내는 과정을, 최적화 문제로 환원하여 해결하는 **conjugate gradients method**를 살펴봅니다. 
Justin Solomon 교수님의 [CS 205a: Mathematical Methods for Robotics, Vision, and Graphics][cs205a] 강의를 기본으로 하되 흐름에 맞게 내용들을 추가하고 재배치했음을 밝힙니다. 부족한점에 대한 지적과 질문은 자유롭게 댓글로 남겨주세요.

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

## Introduction

우리가 쉽게 마주하는 연립일차방정식은 중학교때 배우는 풀이법을 활용하여 해결할 수 있지만, 방정식의 개수가 많은 경우에는 행렬을 활용하면 답을 찾아내곤 합니다. 
연립일차방정식은 행렬로 이루어진 선형 방정식 $$Ax = b $$의 형태로 나타낼 수 있고, Gaussain elimination을 활용한다면 $$n$$개의 방정식에 대하여 언제나 $$O(n^3)$$의 시간안에 해를 구할 수 있습니다. 

하지만, 만약 저희가 풀고자 하는 문제의 $$n$$의 개수가 너무나 크다면 $$O(n^3)$$의 계산 시간을 기다리는 것은 사실상 불가능 할 것입니다. 
그렇다면 해를 구할 수 없으니 이제 손을 놓고 바라만 봐야 하는 걸까요?

Conjugate gradient method는 이러한 상황에서 적용하기 좋은 알고리즘입니다. 주어진 시간 동안 정확한 해를 구하는 것이 불가능하다면, 최대한 해에 가까운 값을 추정하는 것입니다.
선형방정식 $$Ax=b$$의 해를 구하는 문제를 최적화 문제로 환원하여 해를 반복적으로 근사하되, $$O(n^3)$$시간 안에는 정확한 해를 찾아내는 것입니다.
그렇다면, conjugate gradient method가 정확히 무엇이고, 어떻게 구현할 수 있는지 한번 알아보도록 하겠습니다.

## Gradient Descent

선형 방정식 $$Ax = b $$를 최적화 문제로 바꾸기에 앞서, conjugate gradient를 적용하기 위한 조건들에 대해 먼저 살펴보겠습니다.

- $$ A \in \mathbb{R}^{n \times n}$$ square matrix
- $$ A $$ is positive definite matrix

이런 조건들을 만족하는 matrix로는 identity matrix, invertible matrix 등이 있습니다. 
저는 강화학습 알고리즘중 하나인 Trust region policy optimization(TRPO)에서 효율적인 계산을 위해 conjugate gradient method를 공부하게 되었습니다.
주어진 positive definite matrix $$F$$에 대해서 $$F^{-1}b$$의 값을 구해야 하는데, $$F^{-1}$$을 구하는데 시간이 너무나 오래걸리는 것이 문제였습니다.
따라서, $$F^{-1}$$을 직접 구하는 것이 아니라, $$Fx = b$$의 해를 근사하는 방법으로 $$F^{-1}b$$의 값을 추정하여 사용하였습니다.

### Linear equation to optimization

$$A\vec{x} = \vec{b}$$의 해를 조금만 자세히 살펴보면 아래의 방정식 $$f(\vec{x})$$의 최솟값의 해와 동일하다는 것을 쉽게 확인할 수 있습니다.

$$f(\vec{x}) = \cfrac{1}{2}\vec{x}^TA\vec{x} - \vec{b}^T\vec{x} + c$$

$$f(\vec{x})$$의 최솟값은 c값에 상관없이, 미분 값이 0인 지점이 되기 때문에

$$\nabla f(\vec{x}) = A\vec{x} - \vec{b},$$

에서 $$\nabla f(\vec{x}) = 0$$의 해는 $$A\vec{x} = \vec{b}$$의 해와 동일하게 됩니다.

이제 $$A\vec{x} = \vec{b}$$의 해를 구하는 문제가 $$f(\vec{x})$$의 최솟값을 구하는 문제로 바뀔 수 있는 것을 알았으니, 
$$\nabla f(\vec{x}) = 0$$을 직접적으로 찾는 것이 아니라, gradient descent 방법을 통해서 최적화를 진행할 수 있습니다.
Gradient descent 알고리즘을 간단하게 돌아보면 아래와 같습니다.

1. update 방향을 찾는다. $$\vec{d_k} = -\nabla f(\vec{x_{k-1}}) = \vec{b} - A\vec{x_{k-1}}$$
2. 추정값을 update한다. $$\vec{x_k} = \vec{x_{k-1}} + \alpha_k \vec{d_k} $$

### Determining step size

일반적으로 $$f$$를 $$\vec{d_k}$$방향에서 최소화 시킬 수 있는 이동의 크기 $$\alpha_k$$를 찾는 것은 쉬운일이 아니기 때문에, line search등의 방법을 통해 $$\alpha_k$$의 값을 찾곤 합니다.
하지만, 위와 같은 형태의 이차함수 $$f(\vec{x}) = \frac{1}{2}\vec{x}^TA\vec{x} - \vec{b}^T\vec{x} + c$$에선 $$\vec{d_k}$$방향으로 $$f$$를 최소화시키는 $$\alpha_k$$의 값을 직접 어렵지않게 계산할 수 있습니다.

Gradient descent가 이루어진 $$f$$의 값은 $$g(\alpha) = f(\vec{x} + \alpha \vec{d})$$이기 때문에 $$g(\alpha)$$를 최소화하는 $$\alpha$$의 값을 찾으면 됩니다.

<div>
$$
\begin{align*}
  g(\alpha) &= f(\vec{x} + \alpha \vec{d})\\[10pt]
 &= \frac{1}{2}(\vec{x} + \alpha \vec{d})^TA(\vec{x} + \alpha \vec{d}) - \vec{b}^T(\vec{x} + \alpha \vec{d}) + c\\[8pt]
 &= \frac{1}{2}(\vec{x}^TA\vec{x} + 2\alpha\vec{x}^TA\vec{d} + \alpha^{2}\vec{d}^TA\vec{d}) - \vec{b}^T\vec{x} + \vec{b}^T \alpha \vec{d} + c\\[8pt]
 &= \frac{1}{2}\alpha^{2}\vec{d}^TA\vec{d} + \alpha(\vec{x}^TA\vec{d} + \vec{b}^T \vec{d}) + const. \\[6pt]
\end{align*}
$$
<br></div>

이제 $$g(\alpha)$$를 $$\alpha$$에 대해서 최소화 하고 싶다면, $$\cfrac{dg}{d\alpha}(\alpha) = 0 $$이 되는 $$\alpha$$를 찾으면 됩니다.

$$ \frac{dg}{d\alpha}(\alpha) = \alpha\vec{d}^TA\vec{d} + \vec{d}^T(A\vec{x} - \vec{b}) = 0$$

$$ \alpha = \frac{\vec{d}^T(\vec{b} - A\vec{x})}{\vec{d}^TA\vec{d}} $$

그럼 이렇게 찾은 $$\alpha$$의 값을 저희의 gradient descent 알고리즘에 적용하면 아래와 같습니다.

1. update 방향을 찾는다. $$\vec{d_k} = \vec{b} - A\vec{x_{k-1}}$$
2. $$\vec{d_k}$$방향으로 $$f$$를 최소화 시키는 step size, $$ \alpha_k = \cfrac{\vec{d}^T_k (\vec{b} - A\vec{x_{k-1}})}{\vec{d}^T_k A\vec{d_k}}$$를 구한다.
3. 추정 값 $$\vec{x_{k-1}}$$를 update한다. $$ \vec{x_k} = \vec{x_{k-1}} + \alpha_k \vec{d}_k $$

주어진 gradient descent 알고리즘의 수렴성과 수렴시간에 대한 자세한 내용은 cs 205a 강의의 note를 참조해주세요.

## Conjugate Gradient

비록, 위의 gradient descent는 local gradient $$\nabla f(\vec{x})$$의 방향으로 $$f$$를 최소화를 시켜주지만, 최솟값 $$\vec{x} = A^{-1}b$$으로 수렴하는데는 상당히 오랜 시간이 걸립니다.
$$n \times n$$ matrix $$A$$에 대하여 matrix-vector의 곱연산은 $$O(n^2)$$의 시간이 걸립니다. 
위의 gradient descent 알고리즘 역시 한번의 iteration마다 $$O(n^2)$$의 시간이 걸리게 됩니다. 따라서, iteration이 $$n$$번 이상 진행되어야만 값이 수렴한다면 $$O(n^3)$$의 시간안에 정확한 해를 찾을 수 있는 Gaussian elimination을 적용하는 것이 오히려 나은 방법일 것입니다.

따라서, 저희가 원하는 알고리즘은 n번 안에 정확한 해를 찾을 수 있는 gradient descent 알고리즘, conjugate gradient descent 입니다.

### Quadratic optimization to Measuring distance

Conjugate gradient 알고리즘은 저희의 이차함수 $$f(\vec{x})$$를 조금은 다른 시각에서 바라보며 발전되었습니다.
우선, 저희가 구하고자 하는 해 $$\vec{x}^*$$를 알고있다고 가정하겠습니다. 그렇다면, 자연스럽게 $$A\vec{x}^* = \vec{b}$$를 만족하게 됩니다.
이제 식을 $$\vec{x}^*$$를 이용하여 다시 전개해 보겠습니다.

<div>
$$
\begin{align*}
  f(\vec{x}) &= \cfrac{1}{2}\vec{x}^TA\vec{x} - \vec{b}^T\vec{x} + c \\[6pt]
 &= \cfrac{1}{2}(\vec{x}^TA\vec{x} - \vec{x}^T A \vec{x}^* - \vec{x}^{*T} A \vec{x} + \vec{x}^{*T}A\vec{x}^*) +  \cfrac{1}{2}(\vec{x}^T A \vec{x}^* + \vec{x}^{*T} A \vec{x} - \vec{x}^{*T}A\vec{x}^*) - \vec{b}^T\vec{x} + c \\[6pt]
 &= \cfrac{1}{2}(\vec{x} - \vec{x}^{*})^T A (\vec{x} - \vec{x}^*) + \cfrac{1}{2}(\vec{x}^T A \vec{x}^* + (A \vec{x}^{*})^T \vec{x} - \vec{x}^{*T}A\vec{x}^*) - \vec{b}^T\vec{x} + c\\[6pt]
 &= \cfrac{1}{2}(\vec{x} - \vec{x}^{*})^T A (\vec{x} - \vec{x}^*) + \vec{x}^T \vec{b} - \cfrac{1}{2}\vec{x}^{*T} \vec{b} - \vec{b}^T\vec{x} + c, \text{ since } A\vec{x}^* = \vec{b} \\[6pt]
 &= \cfrac{1}{2}(\vec{x} - \vec{x}^{*})^T A (\vec{x} - \vec{x}^*) - \cfrac{1}{2}\vec{x}^{*T} \vec{b} + c \\[6pt]
 &= \cfrac{1}{2}(\vec{x} - \vec{x}^{*})^T A (\vec{x} - \vec{x}^*) + const, \text{ since } \vec{x}^* \text{ is constant in terms of } \vec{x} \\[6pt]
\end{align*}
$$
<br></div>

저희는 앞에서 주어진 matrix $$A$$가 symmetric, postive definite matrix라고 가정한 바가 있습니다. 따라서, $$A = LL^T$$의 형태로 Cholesky분해를 할 수 있게 되고, 위의 수식을 아래의 형태로 전개할 수 있게 됩니다.

$$
\begin{align*}
  f(\vec{x}) &= \cfrac{1}{2}(\vec{x} - \vec{x}^{*})^T A (\vec{x} - \vec{x}^*) + const \\[6pt]
 &= \cfrac{1}{2}\vert \vert L^T(\vec{x} - \vec{x}^{*}) \vert \vert^2_2 + const \\[6pt]
\end{align*}
$$

여기서 lower-triangular matrix $$L^T$$는 역행렬이 존재하기 때문에, 이제 $$f(\vec{x})$$의 의미는 $$\vec{x}$$와 $$\vec{x}^*$$의 거리를 측정하는 함수로 생각할 수도 있습니다.
$$\vec{y} = L^T \vec{x}$$로 $$\vec{y}^* = L^T \vec{x}^*$$로 정의한다면, $$\bar{f}(\vec{y}) = \vert \vert \vec{y} - \vec{y}^* \vert \vert^2_2 $$의 함수와 동일한 최솟값을 가지게 됩니다.

처음부터 다시 돌아보면, 저희의 기존의 목적 $$A\vec{x} = \vec{b}$$의 해를 구하는 문제는 $$\bar{f}(\vec{y}) = \vert \vert \vec{y} - \vec{y}^* \vert \vert^2_2 $$의 거리를 최소화 하는 문제와 동일해지게 됩니다.

### Conjugate-vectors

이제 $$\bar{f}(\vec{y})$$의 최솟점을 어떻게 효율적으로 찾을 수 있을지 생각해보도록 하겠습니다.
$$Q \in \mathbb{R}^{n \times n}$$ 가 orthogonal vector $$\{\vec{w_1}, \vec{w_2}, ... \vec{w_n}\}$$로 이루어진 orthogonal matrix라고 가정하겠습니다. 
$$Q$$는 orthogonal하기 때문에, $$\bar{f}$$의 좌표계를 $$Q$$를 통해 변환하더라도 해는 기존과 동일하게 됩니다.
따라서, $$\bar{f}(\vec{y}) = \vert \vert \vec{y} - \vec{y}^* \vert \vert^2_2 = \vert \vert Q^T\vec{y} - Q^T\vec{y}^* \vert \vert^2_2$$로 바꾸어 쓸 수 있습니다.
또한, 변환된 좌표계에서 $$\vec{w}$$는 standard basis를 이루게 됩니다.

이제, 함수의 최솟값을 n번의 line search를 통해 찾는 방법은 너무나 자명합니다. 
$$\vec{w_1}$$부터 $$\vec{w_n}$$까지 line search를 반복하면 각각의 방향들은 모두 수직이기 때문에, 반드시 n번의 search 안에 최솟값 $$y^*$$을 찾을 수 있게 됩니다.

<center><img src = "/assets/images/math/cg/Qty.png" width = "350"></center><br>

$$\bar{f}(\vec{y})$$에서 다시 기존의 목표 $$ f(\vec{x}) = \vert \vert L^T(\vec{x} - \vec{x}^{*}) \vert \vert^2_2$$를 바라보도록 하겠습니다.
$$f$$에서 $$\bar{f}$$로 바뀐 것은 $$Q^T$$와 유사하게 $$L^T$$를 통해 좌표계를 변환한것 뿐이였습니다. 
따라서, $$\bar{f}$$에서 $$w$$의 방향으로 line search를 진행하는 것은, $$f$$에서 $$(L^T)^{-1}w$$를 따라 line search를 진행하는 것과 동일하게 됩니다.

다시 말하자면, $$v_1 = (L^T)^{-1}w_1$$, $$v_2 = (L^T)^{-1}w_2$$로 정의하고 $$f$$에서 line search를 진행한다면 반드시 n번의 search 안에 $$x^*$$를 찾을 수 있게 되는 것입니다.

<center><img src = "/assets/images/math/cg/x.png" width = "350"></center><br>

임의의 서로 다른 $$i, j \leq n$$에 대하여 $$\{\vec{w_1}, \vec{w_2}, ... \vec{w_n}\}$$는 orthongal set이기 때문에, $$\vec{w_i} \cdot \vec{w_j} = 0$$의 관계를 만족하게 됩니다.
이를 통해 $$\vec{v_i}$$와 $$\vec{v_j}$$에 관계를 유도해보면 아래와 같습니다.

$$0 = \vec{w_i} \cdot \vec{w_j} = (L^T \vec{v_i})^T (L^T \vec{v_j}) = \vec{v_i}^T (LL^T) \vec{v_j} = \vec{v_i}^T A \vec{v_j} $$

위와 같이 임의의 matrix $$A$$에 대하여 두 vector $$\vec{v}, \vec{w}$$가 $$ \vec{v_i}^{T} A \vec{v_j} = 0 $$의 관계를 만족할 때 $$\vec{v}$$와 $$ \vec{w}$$는 **$$A$$-conjugate vectors**라고 말합니다.

다시 저희의 기존의 논의로 돌아오면 $$\{\vec{v_1}, \vec{v_2}, ... \vec{v_n}\}$$가 $$A$$-conjugate vector set이라면, 저희의 목표인 $$f$$는 $$v_1$$부터 $$v_n$$까지 순차적으로 line search를 함으로서 최솟값을 n번의 search 안에 반드시 찾을 수 있게 됩니다.

### Generating $$A$$-conjugate directions

이제 $$A$$-conjugate vectors, $$\{\vec{v_1}, \vec{v_2}, ... \vec{v_n}\}$$만 찾을 수 있다면, n번안에 최적화가 완료되는 conjugate gradient 알고리즘을 만들 수 있습니다.
본격적으로 conjugate vector들을 찾기에 앞서 계산상의 편의를 위해 추정값과 실제 값의 오차를 residual, $$r$$이라고 표기하도록 하겠습니다.

$$\vec{r_k} = -\nabla f(\vec{x}_{k-1}) = \vec{b} - A\vec{x}_{k-1}$$

A-conjugate vector set을 기존의 gradient descent에 적용한다면 아래와 같습니다.

1. Search direction $$v_k$$를 찾는다. 
2. Line search를 통해 $$\alpha_k$$를 찾는다.$$ \alpha_k = \cfrac{\vec{v}^T_k \vec{r}_{k-1}}{\vec{v}^T_k T A\vec{v_k}}$$를 구한다.
3. 추정 값을 update한다. $$ \vec{x_k} = \vec{x_{k-1}} + \alpha_k \vec{v}_k $$
4. Residual을 update한다. $$\vec{r_k} = \vec{b} - A\vec{x}_{k-1}$$

이제 정말로 A-conjugate vectors를 찾으면 끝이 나게됩니다. A-conjugate vector들은 A-orthogonal하기 때문에, Gram-Schmidt 직교화등을 활용한다면 쉽게  $$\{\vec{v_1}, ... \vec{v_n}\}$$를 찾을 수 있습니다.
이전과 똑같이 gradient descent를 진행하되, 이전에 진행했던 search direction에 대해서는 Gram-Shmidt를 적용하는 것입니다.

$$\vec{v}_1 = \vec{r}_1$$

$$\vec{v}_k = \vec{r}_{k-1} - \sum_{i \lt k} \cfrac{\vec{r}_{k-1} A \vec{v}_i }{\vec{v}_i A \vec{v}_i }\vec{v_i} $$

그런데 문제가 있습니다. 매 iteration 마다 Gram-Schmidt를 진행하는 것은 시간이 오래걸릴 뿐더러 $$v_1, ... v_{k-1}$$ 값을 모두 저장하고 있어야 하기 때문에, $$k$$가 커짐에 따라 메모리의 사용량이 크게 증가하게 됩니다.

하지만, 놀랍게도 모든 $$l \lt k$$에 대하여 $$\vec{r}_{k} A \vec{v}_ㅣ$$은 0이되기 때문에 바로 이전의 direction에 대해서만 Gram-Shmidt를 진행하여 주면 됩니다. 
이에 대한 자세한 유도 과정은 cs205의 note를 역시 참조해주세요.

### Conjugate Gradients Algorithm

지금까지 이야기된 내용들을 종합한 conjugate gradient algorithm은 아래와 같습니다.

1. 임의의 추정치 $$\vec{x}_0$$을 초기화
2. 잔차를 초기화 $$ \vec{r}_0 = \vec{b} - A\vec{x}_{0} $$
3. 학습 방향을 초기화 $$ \vec{v}_1 = \vec{r}_0 $$
4. Iterate
  - Line search: $$ \alpha_k = \cfrac{\vec{v}^T_k \vec{r}_{k-1}}{\vec{v}^T_k A\vec{v_k}}$$
  - Update estimate: $$ \vec{x_k} = \vec{x_{k-1}} + \alpha_k \vec{v}_k $$
  - Update residual: $$\vec{r_k} = \vec{b} - A\vec{x}_{k-1}$$
  - Direction update: $$\vec{v}_k = \vec{r}_{k-1} - \cfrac{\vec{r}_{k-1} A \vec{v}_{k-1} }{\vec{v}_{k-1} A \vec{v}_{k-1} }\vec{v}_{k-1}$$

## Implementation

이제 conjugate gradient 알고리즘이 실제로 잘 동작하는지 구현과 함께 확인해보도록 하겠습니다.

### Create positive-definite matrix

{% highlight python %}
n = 100
P = np.random.normal(size=[n, n])
A = np.dot(P.T, P)
b = np.random.rand(n)
{% endhighlight %}

### Gradient Descent

{% highlight python %}
x = np.zeros(n)
for i in range(n):
  r = b - np.dot(A,x)
  alpha = np.dot(r,r) / np.dot(r,np.dot(A,r))
  x = x + alpha*r
{% endhighlight %}

### Conjugate Gradient Descent

{% highlight python %}
x = np.zeros(n)
r = b - np.dot(A,x)
v = np.copy(r)

for i in range(n):
  Av = np.dot(A,v) + damping * v
  alpha = np.dot(v.T,r) / np.dot(v.T, Av)
  x = x + alpha*v
  r = r - alpha*Av
  v = r - (np.dot(np.dot(r.T,A),v) / np.dot(Av,v))*v
{% endhighlight %}

### Results

아래의 그림은 n의 크기를 바꿔가며 gradient descent와 conjugate gradient descent의 수렴속도를 비교한 결과입니다.

<center><img src = "/assets/images/math/cg/r_n10.png" width = "320">
        <img src = "/assets/images/math/cg/r_n100.png" width = "320">
        <img src = "/assets/images/math/cg/r_n1000.png" width = "320"></center><br>

비록 conjugate gradient descent는 작은 timestep에서는 gradient descent보다 noisy한 update를 보이지만, 일정 timestep이 지난 후에는 훨씬 가파르게 수렴하는 것을 확인할 수 있었습니다.

## Thoughts

하지만, 실제로 conjugate gradient를 구현하고 학습시키는데 있어서 두 가지 의문점이 생겼습니다.

### Damping factor

실제 수식과 구현상의 수식을 자세히 살펴보면, 구현에 있어서 damping factor라는 파라미터가 추가된 것을 확인하실 수 있습니다. 
$$ A \cdot v$$는 $$ \alpha$$ 값의 계산에 있어 분모에 들어가기 때문에, 값이 너무 작거나 부정확한 경우 numerical stability를 크게 해칠 수 있습니다. 
따라서, 이를 해결하기 위해 update 방향을 momentum과 같이 유지시켜주어 stability를 가져오는 것이 damping factor 입니다.

실험 결과 damping factor는 아래와 같이 convergence에 있어서 중요한 파라미터라는 것을 확인할 수 있었습니다.

<center><img src = "/assets/images/math/cg/100_damping.png" width = "320">
        <img src = "/assets/images/math/cg/1000_damp.png" width = "320"></center><br>

첫번째 의문점은 damping factor가 클 수록 수렴 속도가 빠르게 나왔었는데 왜 대부분의 다른 코드들은 damping factor를 $$0.1$$ 혹은 $$0.01$$로 고정해두고 사용하는가? 입니다.


### Fisher-Information matrix

제가 conjugate gradient descent를 적용하고자 하는 TRPO에서는 fisher-information matrix, $$F$$에 대하여 $$F^{-1}b$$의 값을 빠르게 구하기 위해 $$Fx = b$$의 해를 근사합니다.
하지만, TRPO 논문에선 파라미터 $$n$$의 크기가 1000보다 큼에도 불구하고 10번의 gradient descent만을 수행하여 해를 근사합니다. 
위의 실험 결과를 살펴보면 $$n$$의 크기가 1000일 때, 200번 이상 반복해야 값이 수렴했는데 과연 저렇게 부정확한 갓ㅂ을 사용하여도 되는지 의문이 생겼습니다.

그래서, fisher-information matrix를 직접 만들어서 conjugate gradient의 수렴 속도를 확인해보았습니다.

{% highlight python %}
# Data
X0 = np.random.randn(100, n) - 1
X1 = np.random.randn(100, n) + 1
X = np.vstack([X0, X1])
t = np.vstack([np.zeros([100, 1]), np.ones([100, 1])])

# Model
W = np.random.randn(n, 1) * 0.01

def sigm(x):
    return 1/(1+np.exp(-x))

def NLL(y, t):
    return -np.mean(t*np.log(y) + (1-t)*np.log(1-y))

alpha = 0.1

# Forward
z = X @ W
y = sigm(z)
loss = NLL(y, t_train)

# Backward
m = y.shape[0]
dy = (y-t_train)/(m * (y - y*y))
dz = sigm(z)*(1-sigm(z))
dW = X_train.T @ (dz * dy)

grad_loglik_z = (t_train-y)/(y - y*y) * dz
grad_loglik_W = grad_loglik_z * X_train
F = np.cov(grad_loglik_W.T)
{% endhighlight %}

<center><img src = "/assets/images/math/cg/fim_1000.png" width = "320">
        <img src = "/assets/images/math/cg/fim_10000.png" width = "320"></center><br>

놀랍게도, fisher-information matrix에 대해서는 conjugate gradient를 사용하면 10번안에 가파르게 수렴하는 것을 확인할 수 있었습니다.
따라서, TRPO에서 conjugate gradient를 사용해 10번만 update를 취해도 근사값에 큰 문제가 없다는 사실은 알았습니다. 하지만, fisher information matrix의 어떠한 성질이 이렇게 가파른 수렴을 가져올까? 라는 두번째 의문점은 해결할 수 없었습니다.

혹시나 이유를 알고 계시거나 의견이 있으시다면 댓글 부탁드립니다. 감사합니다.









[cs205a]: https://graphics.stanford.edu/courses/cs205a-13-fall/index.html