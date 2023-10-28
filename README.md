## 코딩테스트 ESSENTAIL
주요 핵심 알고리즘을 문제를 기반으로 정리합니다. 

정말 실전에 쓰이고 필요한 핵심 내용들만 최대한 간추려서 작성합니다.

시간이 지나도 빠르게 본실력으로 돌아올수 있도록 오답노트를 함께 작성합니다.

## 주요 알고리즘 유형
<details>
<summary> BFS </summary>
  
### 핵심개념
---

### 시간복잡도
---

### 핵심코드
---

### 문제1
---

### 문제2
---

</details>

<details>
<summary> DFS </summary>
  dsa
</details>

<details>
<summary> 백트래킹 </summary>
  dsa
</details>

<details>
<summary> 구현: 시뮬레이션과 완전탐색 </summary>
  https://www.youtube.com/watch?v=2zjoKjt97vQ&list=PLRx0vPvlEmdAghTr5mXQxGpHjWqSz0dgC&index=2
</details>

<details>
<summary> 이진탐색 </summary>
  dsa
</details>

<details>
<summary> DP </summary>
  dsa
</details>

<details>
<summary> 그리디 </summary>
  dsa
</details>

<details>
<summary> MST (union-find / 크루스칼 / 프림) </summary>
  union-find / 크루
</details>

<details>
<summary> 최단거리(다익스트라, 플로이드 와샬) </summary>
ㄴㅇㅁ
</details>

## 기타 알고리즘 유형
<details>
<summary> 위상정렬 </summary>
ㄴㅇㅁ
</details>

<details>
<summary> 우선순위큐 </summary>
ㄴㅇㅁ
</details>

<details>
<summary> 투포인터 </summary>
ㄴㅇㅁ
</details>

<details>
<summary> heap </summary>

### 핵심개념
---
```
비어 있는 최소 힙에 7, 6, 5, 4, 3, 2, 1 을 순서대로 삽입
```
https://github.com/briiidgehong/cote-essential/assets/73451727/6920245e-512e-4f6e-884a-fd5523ca57cf



### 시간복잡도
---

### 핵심코드
---

### 문제1
---

### 문제2
---

</details>

<details>
<summary> hash / stack / queue / 정렬 </summary>
ㄴㅇㅁ
</details>

<details>
<summary> 소수판별 </summary>
ㄴㅇㅁ
</details>

<details>
<summary> 에라토스테네스의 체 </summary>
ㄴㅇㅁ
</details>

<details>
<summary> 구간합 </summary>
ㄴㅇㅁ
</details>

<details>
<summary> 진수 </summary>
ㄴㅇㅁ
</details>

<details>
<summary> bisect left right </summary>
ㄴㅇㅁ
</details>

<details>
<summary> 최소공배수 / 최대공약수 </summary>
<img width="559" alt="스크린샷 2023-10-28 오전 11 47 58" src="https://github.com/briiidgehong/cote-essential/assets/73451727/e3404f13-3e76-4d7e-893b-138ac6c69aee">
```
# import math
# n,m = map(int, input().split())

# gcd = math.gcd(n,m)

# lcm = int((n*m) / gcd)
# print(lcm)

n,m = map(int, input().split())

gcd = 0
for idx in range(1, min(n,m)+1):
    if n % idx == 0 and m % idx == 0:
        gcd = idx

lcm = int((n*m)/gcd)
print(lcm)
```

<img width="494" alt="스크린샷 2023-10-28 오전 11 49 27" src="https://github.com/briiidgehong/cote-essential/assets/73451727/4535c7a7-9513-4448-9561-a2f1ba0e59d9">

```
import math

# 최대공약수 gcd / 최소공배수 lcm

# lcm(a,b,c,d) = lcm(lcm(lcm(a,b),c),d)

n = int(input())

list_a = list(map(int, input().split()))

if len(list_a) == 1:
    print(list_a[0])
else:
    lcm = 1
    for idx in range(len(list_a)-1):
        if idx == 0:
            gcd = math.gcd(list_a[idx],list_a[idx+1])
            lcm = int((list_a[idx]*list_a[idx+1])/gcd)
        else:
            gcd = math.gcd(lcm,list_a[idx+1])
            lcm = int((lcm*list_a[idx+1])/gcd)

    print(lcm)
```


</details>

## 오답노트






