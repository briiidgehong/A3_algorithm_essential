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
<summary> 구현 / 시뮬레이션 / 완전탐색 </summary>
  https://www.youtube.com/watch?v=2zjoKjt97vQ&list=PLRx0vPvlEmdAghTr5mXQxGpHjWqSz0dgC&index=2

### 문제 - 날짜구현
---
<img width="677" alt="스크린샷 2023-10-28 오전 11 54 09" src="https://github.com/briiidgehong/cote-essential/assets/73451727/392b392b-094b-4450-adba-b8dec710b8ce">
<img width="679" alt="스크린샷 2023-10-28 오전 11 54 22" src="https://github.com/briiidgehong/cote-essential/assets/73451727/6a1a6286-881d-4bf4-bc0a-d3cd0a7ddda1">

```
import datetime
Y, M, D = map(int, input().split())


def solution(y,m,d):
    try:
        y = str(y)
        if len(y) < 4:
            for each in range(4-len(y)):
                y = "0" + y
            
        datetime.datetime.strptime(y+"-"+str(m)+"-"+str(d), "%Y-%m-%d")
    except Exception as e:
        return -1
    else:
        if M in [3,4,5]:
            return "Spring"
        elif M in [6,7,8]:
            return "Summer"
        elif M in [9,10,11]:
            return "Fall"
        else:
            return "Winter"

print(solution(Y,M,D))
```

---
<img width="747" alt="스크린샷 2023-10-28 오후 12 00 29" src="https://github.com/briiidgehong/cote-essential/assets/73451727/2bc87420-dedb-4a18-9117-23a14ad11188">
```
import datetime
# 2024-m1-d1 ~ 2024-m2-d2 까지 A요일이 등장하는 횟수 단, 2024년 m1월 d1일이 월요일 이었다면 !

month_date_list = list(input().split()) # m1,d1 , m2,d2
str_day = input()
for idx in range(len(month_date_list)):
    if len(month_date_list[idx]) < 2:
        month_date_list[idx] = "0" + month_date_list[idx]

start_date = datetime.datetime.strptime(f"2024-{month_date_list[0]}-{month_date_list[1]}", '%Y-%m-%d')
end_date = datetime.datetime.strptime(f"2024-{month_date_list[2]}-{month_date_list[3]}", '%Y-%m-%d')

weekday_list = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

count = 0 
week_idx = 0
while start_date <= end_date:
    if weekday_list[week_idx] == str_day:
        count += 1
    week_idx += 1
    week_idx %= 7
    start_date += datetime.timedelta(days=1)
    
print(count)
```

### 문제 - 시뮬레이션 - 최장연속부분수열
---
<img width="749" alt="스크린샷 2023-10-28 오전 11 57 33" src="https://github.com/briiidgehong/cote-essential/assets/73451727/247811fa-21ad-406c-9a95-288070a9be83">

```
n, t = map(int, input().split())

input_list = list(map(int, input().split()))

result_list = []
sub_list = []
for each in input_list:
    if each > t:
        sub_list.append(each)
    else:
        if len(sub_list) > 0:
            result_list.append(sub_list)
            sub_list = []
if len(sub_list) > 0:
    result_list.append(sub_list)

result_count = 0
for each in result_list:
    if len(each) > result_count:
        result_count = len(each)

print(result_count)
```

### 문제 - 시뮬레이션 - 계속 중첩되는 사각형
<img width="751" alt="스크린샷 2023-10-28 오전 11 59 02" src="https://github.com/briiidgehong/cote-essential/assets/73451727/8b61e355-e88f-471e-b298-a397a9cfe44e">

```
n = int(input())
input_list = []
for _ in range(n):
    input_list.append(list(map(int, input().split())))

sq = list([0]*200 for _ in range(200))


for idx, each in enumerate(input_list):
    if idx % 2 == 0:
        color = "red"
    else:
        color = "blue"
    for idx_x in range(each[0]+100, each[2]+100):
        for idx_y in range(each[1]+100, each[3]+100):
            sq[idx_y][idx_x] = color

count = 0 
for idx_y in range(len(sq)):
    for idx_x in range(len(sq[idx_y])):
        if sq[idx_y][idx_x] == "blue":
            count += 1

print(count)
```

  
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






