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

---

<img width="747" alt="스크린샷 2023-10-28 오후 12 04 44" src="https://github.com/briiidgehong/cote-essential/assets/73451727/e264da06-1967-4001-9f04-88e37d436662">
<img width="749" alt="스크린샷 2023-10-28 오후 12 04 54" src="https://github.com/briiidgehong/cote-essential/assets/73451727/7ac39ef7-62ba-424b-87cf-9b40b8fde5ef">

```
import traceback
# "x L" / "x R"왼쪽으로 뒤집으면 흰색으로 바뀌고, 오른쪽으로 뒤집으면 검은색
# 현재 타일 포함 총 x개의 타일을 움직임
n = int(input())

rec_list = []
for _ in range(n):
    rec_list.append(list(input().split()))

# -> 4 R
# <- 5 L
# -> 7 R
# <- 4 L
temp_list = [None] * 100 * n * 2

start_id = 100 * n

for each in rec_list:
    if each[1] == "L":
        move = -1
        for sub_idx, sub_each in enumerate(range(int(each[0]))):
            temp_list[start_id] = "white"
            if sub_idx == int(each[0])-1:
                continue
            else:
                start_id += move
    else: # R
        move = 1
        for sub_idx, sub_each in enumerate(range(int(each[0]))):
            temp_list[start_id] = "black"
            if sub_idx == int(each[0])-1:
                continue
            else:
                start_id += move
            

white_count = temp_list.count("white")
black_count = temp_list.count("black")

print(white_count, black_count)
```

### 문제 - 시뮬레이션 - dx dy technique
<img width="750" alt="스크린샷 2023-10-28 오후 12 09 45" src="https://github.com/briiidgehong/cote-essential/assets/73451727/5dd59a58-9643-4afb-8b40-dc37a4d8daed">
<img width="743" alt="스크린샷 2023-10-28 오후 12 09 51" src="https://github.com/briiidgehong/cote-essential/assets/73451727/18f3233a-a156-4ea7-9600-d11cc51c6749">

```
# N∗N크기의 정사각형 모양의 격자 정보가 주어졌을 때, 
# 가운데 위치에서 북쪽을 향한 상태로 움직이는 것을 시작하려 합니다. 

# T개의 명령에 따라 움직이며, 명령어는 L,R,F로 주어집니다. 
# 명령 L은 왼쪽으로 90도 방향 전환을, 명령 R은 오른쪽으로 90도 방향 전환을, 명령 F가 주어지면 바라보고 있는 방향으로 한칸 이동하게 됩니다. 
# 시작 위치를 포함하여 위치를 이동하게 될 때마다 해당 칸에 적혀있는 수를 계속 더한다고 헀을 때, 
# 이들의 총합을 구하는 프로그램을 구하는 프로그램을 작성해보세요. 
# 단, 격자의 범위를 벗어나게 하는 명령어는 무시해야함에 유의합니다.

n, t = map(int, input().split())
order_list = list(input())

input_list = []
for _ in range(n):
    input_list.append(list(map(int, input().split())))

sum_count = 0
direction = 0
x_y = [int(n/2), int(n/2)] # x, y

sum_count += input_list[x_y[1]][x_y[0]]
for idx in range(len(order_list)):
    if order_list[idx] == "L":
        direction -= 90
    elif order_list[idx] == "R":
        direction += 90
    elif order_list[idx] == "F":
        direction = direction % 360
        if direction == 0:
            move = (0,-1) # x, y
        elif direction == 90 or direction == -270:
            move = (1,0) # x, y

        elif direction == 180 or direction == -180:
            move = (0,1) # x, y

        elif direction == 270 or direction == -90:
            move = (-1,0) # x, y
        
        try:
            y = x_y[1]+move[1]
            x = x_y[0]+move[0]
            if x < 0 or y < 0:
                continue
            sum_count += input_list[y][x]
        except Exception:
            continue
        else:
            x_y = [x_y[0]+move[0], x_y[1]+move[1]]

print(sum_count)
```



### 문제 - 완전탐색
<img width="750" alt="스크린샷 2023-10-28 오후 12 06 56" src="https://github.com/briiidgehong/cote-essential/assets/73451727/171aaf7b-f9af-4899-992f-0ba38adbb0d7">

```
from itertools import combinations

n, s = map(int, input().split())
input_list = map(int, input().split())
# 6c4
combination_list = list(combinations(input_list, n-2))

diff_list = []
for each in combination_list:
    diff_list.append(abs(s - sum(each)))

print(min(diff_list))
```

---

<img width="754" alt="스크린샷 2023-10-28 오후 12 08 04" src="https://github.com/briiidgehong/cote-essential/assets/73451727/a868d918-8256-4f50-859d-6ec4e1505105">
<img width="741" alt="스크린샷 2023-10-28 오후 12 08 10" src="https://github.com/briiidgehong/cote-essential/assets/73451727/44a06205-7027-4b79-8e89-8a866d05d312">

```
from itertools import combinations
n, h, t = map(int, input().split())
h_list = list(map(int, input().split()))

# 기준 = 연속되는 구간
section_list = []
for start_idx in range(n):
    if start_idx+t <= n:
        section_list.append(list(each for each in range(start_idx, start_idx+t)))

cost_list = []
for section in section_list:
    cost = 0
    for idx in section:
        cost += abs(h_list[idx] - h)
    cost_list.append(cost)

print(min(cost_list))
```

---

<img width="748" alt="스크린샷 2023-10-28 오후 12 12 36" src="https://github.com/briiidgehong/cote-essential/assets/73451727/6172d1b8-4422-4e91-8d8c-056700a4c9b0">

```
from itertools import combinations

input_list = list(map(int, input().split()))
idx_list = list(idx for idx in range(len(input_list)))

c_list = list(map(list, list(combinations(idx_list, 2))))

gazisu_list = []
for each in c_list:
    temp_idx_list = idx_list[:]
    temp_idx_list.remove(each[0])
    temp_idx_list.remove(each[1])
    
    temp_c_list = list(map(list, list(combinations(temp_idx_list, 2))))
    for sub_each in temp_c_list:
        temp_temp_idx_list = temp_idx_list[:]
        temp_temp_idx_list.remove(sub_each[0])
        temp_temp_idx_list.remove(sub_each[1])   
        gazisu_list.append([each, sub_each, temp_temp_idx_list])

diff_list = []
for each in gazisu_list:
    sum_0 = sum(input_list[sub_each] for sub_each in each[0])
    sum_1 = sum(input_list[sub_each] for sub_each in each[1])
    sum_2 = sum(input_list[sub_each] for sub_each in each[2])

    if sum_0 != sum_1 and sum_1 != sum_2 and sum_0 != sum_2:  
        sum_list = [sum_0, sum_1, sum_2]
        diff_list.append(max(sum_list) - min(sum_list))

if len(diff_list) == 0:
    print(-1)
else:
    print(min(diff_list))
```

---

<img width="750" alt="스크린샷 2023-10-28 오후 12 13 40" src="https://github.com/briiidgehong/cote-essential/assets/73451727/04566567-2ed5-4bc5-bf1e-1fa556a12003">

```
start, end = map(int, input().split())

count = 0
for each in range(start, end+1):
    if str(each) == "".join(list(reversed(str(each)))):
        count += 1
print(count)
```

---
<img width="653" alt="스크린샷 2023-10-28 오후 12 17 19" src="https://github.com/briiidgehong/cote-essential/assets/73451727/e90b0fbd-62b7-4bd3-87b5-2018b62edcff">
<img width="655" alt="스크린샷 2023-10-28 오후 12 17 26" src="https://github.com/briiidgehong/cote-essential/assets/73451727/bbe83bea-4367-46a9-9ea4-d8c2acde1ea4">

```
# 선생님이 N명의 학생에게 B만큼의 예산으로 선물을 주려고 합니다. 
# 학생 i가 원하는 선물의 가격 P(i)와 배송비 S(i)가 있고, 선생님에게는 선물 하나를 정해서 반값으로 할인받을 수 있는 쿠폰이 있습니다. 
# 선생님이 선물 가능한 학생의 최대 명수를 구하는 프로그램을 작성해보세요. 단, 선물의 가격은 항상 짝수입니다.
import copy

# 학생수 n / 예산 b
n, b = map(int, input().split())

# 선물의 가격 p / 배송비 s
gift_list = []
for _ in range(n):
    gift = list(map(int, input().split()))
    gift_list.append(gift)

# coupon
count_list = []
for idx in range(n):
    temp_gift_list = copy.deepcopy(gift_list) # deep copy ! / call by value
    temp_gift_list[idx][0] = int(temp_gift_list[idx][0]/2)
    temp_gift_list.sort(key=lambda x:x[0]+x[1])
    sum_count = 0
    count = 0
    for each in temp_gift_list:
        sum_count += (each[0] + each[1])
        if sum_count <= b:
            count += 1
        else:
            break
    count_list.append(count)

print(max(count_list))
```

---
<img width="655" alt="스크린샷 2023-10-28 오후 12 18 59" src="https://github.com/briiidgehong/cote-essential/assets/73451727/0fbc4666-237a-4f73-bc70-2ce27d3f6aba">
<img width="651" alt="스크린샷 2023-10-28 오후 12 19 05" src="https://github.com/briiidgehong/cote-essential/assets/73451727/aa1e2837-42c0-4b60-a0d9-846d84c31448">

```
n, m = map(int, input().split())
input_list = list(map(int, input().split()))

result_list = []
for start_idx in range(n):
    idx = start_idx
    sum_num = 0
    for _ in range(m):
        sum_num += input_list[idx]
        idx = input_list[idx] - 1
    result_list.append(sum_num)
print(max(result_list))
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
### 정렬
---
<img width="658" alt="스크린샷 2023-10-28 오후 12 16 09" src="https://github.com/briiidgehong/cote-essential/assets/73451727/8a369071-f934-4962-ac53-899ad021f3c1">
<img width="656" alt="스크린샷 2023-10-28 오후 12 16 17" src="https://github.com/briiidgehong/cote-essential/assets/73451727/e9a8ee38-a626-4668-ad47-6fa77cc23ff3">


```
n = int(input())

temp_list = []
for idx in range(n):
    input_list = list(map(int, input().split())) # 키, 몸무게
    input_list.append(idx+1)
    temp_list.append(input_list)

temp_list = sorted(temp_list, key = lambda x:(x[0], -x[1]))

for each in temp_list:
    print(*each)
```
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
  
<img width="747" alt="스크린샷 2023-10-28 오후 12 02 39" src="https://github.com/briiidgehong/cote-essential/assets/73451727/2f2cad28-5e83-47e5-a714-d6a05e19619c">

```
a, b = map(int, input().split())
n = input() # a 진수로 표현된 n -> n 을 b 진수로 표현

# 1. a진수 n을 10진수로 변환
sum_num = 0
for idx, each in enumerate(reversed(str(n))):
    if idx == 0:
        sum_num += int(each)
    else:
        multiple = 1
        for _ in range(idx):
            multiple *= a
        sum_num += multiple * int(each)
num_10 = sum_num

# 2. 10진수 n을 b진수로 변환
# 16 / 4 = 4 ''' 0
# 4 / 4 = 1 ''' 0
# 1 / 4 = 0 ''' 1
# 1 0 0
num_b_list = []
while True:
    mock, namugi = divmod(num_10,b)
    num_b_list.append([mock, namugi])
    num_10 = mock
    if mock == 0:
        break
str_result = ''
for each in reversed(num_b_list):
    str_result += str(each[1])
print(str_result)
```

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






