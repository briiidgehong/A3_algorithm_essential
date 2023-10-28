## 코딩테스트 ESSENTAIL
주요 핵심 알고리즘을 문제를 기반으로 정리합니다. 

정말 실전에 쓰이고 필요한 핵심 내용들만 최대한 간추려서 작성합니다.

시간이 지나도 빠르게 본실력으로 돌아올수 있도록 오답노트를 함께 작성합니다.


"결국 코테는 문제를 추려서 지속적으로 반복하는것이 정답"


## 주요 알고리즘 유형
<details>
<summary> BFS </summary>
  
```
# 핵심개념

# 시간복잡도
# 핵심코드
```
---

### 문제


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

### 문제 - 시뮬레이션
<img width="653" alt="스크린샷 2023-10-28 오후 12 30 17" src="https://github.com/briiidgehong/cote-essential/assets/73451727/e61667db-73f8-499d-a49b-d4786da390ab">
<img width="833" alt="스크린샷 2023-10-28 오후 12 30 34" src="https://github.com/briiidgehong/cote-essential/assets/73451727/2028b430-c34e-42ea-a143-4057fe2303ef">

```
# A와 B가 동일한 시작점에서 같은 방향으로 출발합니다. 
# 도중에 방향을 바꾸는 경우는 없고, A, B는 각각 N번, M번에 걸쳐 주어지는 특정 속도로 특정 시간만큼 이동한다고 합니다. 
# 이 경기는 특이하게 매 시간마다 현재 선두인 사람들을 모아 명예의 전당에 그 이름을 올려줍니다. 
# A, B의 움직임에 대한 정보가 주어졌을 때 명예의 전당에 올라간 사람의 조합이 총 몇 번 바뀌었는지를 출력하는 프로그램을 작성해보세요.

n, m = map(int, input().split())

a_list = []
# A
for _ in range(n): # v t
    v, t = map(int, input().split())
    for each in range(t):
        if len(a_list) == 0:
            a_list.append(v)
        else:
            a_list.append(a_list[-1]+ v)
        
b_list = []
# B
for _ in range(m): # v t
    v, t = map(int, input().split())
    for each in range(t):
        if len(b_list) == 0:
            b_list.append(v)
        else:
            b_list.append(b_list[-1]+ v)

# A, B가 동시에 명예의 전당에 올라가게 됩니다. -> A / B / AB
glory_list = []
count = 0
for idx in range(len(a_list)):
    if idx == 0:
        if a_list[idx] == b_list[idx]:
            glory_list.append("ab")
        elif a_list[idx] > b_list[idx]:
            glory_list.append("a")
        elif a_list[idx] < b_list[idx]:
            glory_list.append("b")    
    else:
        if a_list[idx] == b_list[idx]:
            if glory_list[-1] != "ab":
                count += 1
            glory_list.append("ab")
        elif a_list[idx] > b_list[idx]:
            if glory_list[-1] != "a":
                count += 1
            glory_list.append("a")
        elif a_list[idx] < b_list[idx]:
            if glory_list[-1] != "b":
                count += 1
            glory_list.append("b")

print(count+1)
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

---

<img width="658" alt="스크린샷 2023-10-28 오후 12 20 39" src="https://github.com/briiidgehong/cote-essential/assets/73451727/2dacf817-1b57-4e5d-9186-ed3fc4619605">
<img width="657" alt="스크린샷 2023-10-28 오후 12 20 46" src="https://github.com/briiidgehong/cote-essential/assets/73451727/4d7b5f95-7cc0-4484-9e0c-d745a757eb24">

```
# n개의 수가 주어졌을 때, 각각의 수에 변화를 적절하게 주어, 최종적으로 나오는 수들 중 최대 최소간의 차가 k 이하가 되게끔 만들려고 합니다. 
# 수 a가 수 b로 바뀌는데 드는 비용이 |a - b|라 했을 때, 필요한 최소 비용을 구하는 프로그램을 작성해보세요.
n, k = map(int, input().split())
input_list = list(map(int, input().split()))

cost = 0
while True:
    max_num = max(input_list)
    min_num = min(input_list)
    if max_num - min_num <= k:
        break
    max_count = input_list.count(max_num)
    min_count = input_list.count(min_num)
    if max_count >= min_count:
        for idx, each in enumerate(input_list):
            if each == min_num:
                input_list[idx] += 1
                cost += 1
    else:
        for idx, each in enumerate(input_list):
            if each == max_num:
                input_list[idx] -= 1
                cost += 1
    
print(cost)
```

### 문제 - 케이스별로 나누기
---
<img width="656" alt="스크린샷 2023-10-28 오후 12 22 11" src="https://github.com/briiidgehong/cote-essential/assets/73451727/9b996ff6-738c-467a-9c15-8728202b1f65">
<img width="655" alt="스크린샷 2023-10-28 오후 12 22 18" src="https://github.com/briiidgehong/cote-essential/assets/73451727/8896b857-68de-43de-bd2e-9e97d56a8c95">

```
n = int(input())

input_list = []
for _ in range(n):
    input_list.append(list(map(int, input().split())))

import copy
yes_flag = False
for idx, each in enumerate(input_list):
    temp_list = copy.deepcopy(input_list)
    del temp_list[idx]
    duplicate_list = []
    for sub_idx, sub_each in enumerate(temp_list):
        if sub_idx == 0:
            duplicate_list = list(num for num in range(sub_each[0], sub_each[1]+1))
        else:
            temp_duplicate_list = []
            for num in range(sub_each[0], sub_each[1]+1):
                if num in duplicate_list:
                    temp_duplicate_list.append(num)
            duplicate_list = temp_duplicate_list
    
    if len(duplicate_list) >= 1:
        yes_flag = True
        break
if yes_flag:
    print("Yes")
else:
    print("No")
```

---
<img width="658" alt="스크린샷 2023-10-28 오후 12 23 18" src="https://github.com/briiidgehong/cote-essential/assets/73451727/25d4875a-3b2f-421d-86ef-c28375c846e9">
<img width="659" alt="스크린샷 2023-10-28 오후 12 23 25" src="https://github.com/briiidgehong/cote-essential/assets/73451727/1967adb7-3714-46f9-809d-5481f7fa63f5">

```
import copy
x = int(input()) # xm만큼 달리기 진행

# 10
# 시간 1 2 3 4 5 6 
# 속력 1 2 3 2 1 1
# 거리 1 2 3 2 1 1
# 누적 1 3 6 8 9 10

if x == 1:
    print(1)
else:
    import copy
    # + / 유지 / -
    # 판단
    result_list = [1]
    count = 0 
    while True:
        # +
        plus_list = copy.deepcopy(result_list)
        plus_list.append(result_list[-1]+1)
        sum_num = sum(plus_list)
        down_list = list(each for each in range(1, plus_list[-1]))
        sum_num += sum(down_list)
        if sum_num == x:
            plus_list.extend(down_list)
            result_list = plus_list
            break
        elif sum_num > x:
            pass
        elif sum_num < x:
            result_list = plus_list
            continue
        
        # 0
        keep_list = copy.deepcopy(result_list)
        keep_list.append(result_list[-1])
        sum_num = sum(keep_list)
        down_list = list(each for each in range(1, keep_list[-1]))
        sum_num += sum(down_list)
        if sum_num == x:
            keep_list.extend(down_list)
            result_list = keep_list
            break
        elif sum_num > x:
            pass
        elif sum_num < x:
            result_list = keep_list
            continue

        # -1
        minus_list = copy.deepcopy(result_list)
        minus_list.append(result_list[-1]-1)
        sum_num = sum(minus_list)
        down_list = list(each for each in range(1, minus_list[-1]))
        sum_num += sum(down_list)
        if sum_num == x:
            minus_list.extend(down_list)
            result_list = minus_list
            break
        elif sum_num > x:
            pass
        elif sum_num < x:
            result_list = minus_list
            continue


    print(len(result_list))
```

---

<img width="655" alt="스크린샷 2023-10-28 오후 12 24 27" src="https://github.com/briiidgehong/cote-essential/assets/73451727/10472869-a171-407b-b6ea-3dc57546fe8e">
<img width="657" alt="스크린샷 2023-10-28 오후 12 24 33" src="https://github.com/briiidgehong/cote-essential/assets/73451727/f83ea98a-ebdb-4f2c-bb8e-ffbe9487c8cb">


```
n = int(input())
# 인접한 두 사람의 위치를 계속 바꾸는 행위만 가능하다고 할 때, 가능한 최소 횟수를 구하는 프로그램
input_list = list(input().split())
sorted_list = sorted(input_list)
count = 0
for end_idx in reversed(range(0, n-1)):
    for idx, each in enumerate(input_list):
        if idx <= end_idx:
            if input_list[idx] > input_list[idx+1]:
                temp = input_list[idx+1]
                input_list[idx+1] = input_list[idx]
                input_list[idx] = temp
                count += 1
    if input_list == sorted_list:
        break

print(count)
```

### 문제 - ad hoc
---
#### - 지극히 최선인 전략이 확실해지는 경우
<img width="654" alt="스크린샷 2023-10-28 오후 12 26 31" src="https://github.com/briiidgehong/cote-essential/assets/73451727/f9021e36-74d9-44f3-98c9-4b30954d0da4">
<img width="653" alt="스크린샷 2023-10-28 오후 12 26 39" src="https://github.com/briiidgehong/cote-essential/assets/73451727/0e1d5803-3ca5-49c5-ac0e-c267f26d05e0">

```
groun_num = int(input())
input_list = list(map(int, input().split()))
input_list.sort()

interval_list = []
for idx in range(0, groun_num):
    interval_list.append(input_list[idx+groun_num] - input_list[idx])
print(min(interval_list))
```

---
#### - 고려해야 할 대상이 뚜렷이 정해지는 경우
<img width="655" alt="스크린샷 2023-10-28 오후 12 27 40" src="https://github.com/briiidgehong/cote-essential/assets/73451727/4fe9e5d3-050a-4c78-adef-0a0e68e04eb3">
<img width="654" alt="스크린샷 2023-10-28 오후 12 27 53" src="https://github.com/briiidgehong/cote-essential/assets/73451727/58cd7557-cb09-431b-a7cf-0ae3db5edfe0">

```
n = int(input())
input_list = []
for _ in range(n):
    input_list.append(list(map(int, input().split())))

import copy
interval_list = []
for idx in range(n):
    temp_list = copy.deepcopy(input_list)
    del temp_list[idx]
    interval = []
    for each in temp_list:
        interval.extend(each)
    interval.sort()
    interval_list.append(interval[-1]-interval[0])
        
print(min(interval_list))
```


</details>


<details>
<summary> 이진탐색 </summary>
  dsa
</details>


<details>
<summary> DP </summary>

### 문제 
---
<img width="791" alt="스크린샷 2023-10-28 오후 2 43 25" src="https://github.com/briiidgehong/cote-essential/assets/73451727/db9f6f39-c9aa-4a8a-97a8-6c3e256403a7">
<img width="774" alt="스크린샷 2023-10-28 오후 2 43 57" src="https://github.com/briiidgehong/cote-essential/assets/73451727/056f4d68-0bb4-42c0-85e3-2dd68a1dcc2b">
<img width="777" alt="스크린샷 2023-10-28 오후 2 44 02" src="https://github.com/briiidgehong/cote-essential/assets/73451727/ce1d719b-3511-43ae-aa5b-8933099f6d99">

---
<img width="594" alt="스크린샷 2023-10-28 오후 2 40 14" src="https://github.com/briiidgehong/cote-essential/assets/73451727/8275a1e6-bbce-4781-94cd-64f8e57bedce">
<img width="545" alt="스크린샷 2023-10-28 오후 2 41 15" src="https://github.com/briiidgehong/cote-essential/assets/73451727/23a7aba3-c042-4f91-8279-e8310932a7cf">
<img width="353" alt="스크린샷 2023-10-28 오후 2 42 26" src="https://github.com/briiidgehong/cote-essential/assets/73451727/fc417518-f907-49c8-b3df-62ae2c1c8693">

https://github.com/briiidgehong/cote-essential/assets/73451727/0c52acb3-1656-48c9-889c-f91dea5301d3

---
<img width="787" alt="스크린샷 2023-10-28 오후 2 45 11" src="https://github.com/briiidgehong/cote-essential/assets/73451727/cb2f11ea-6006-45c7-80a4-aae9f135d907">
<img width="764" alt="스크린샷 2023-10-28 오후 2 45 42" src="https://github.com/briiidgehong/cote-essential/assets/73451727/2266af8f-3619-41c4-baed-7621170f40a1">
<img width="433" alt="스크린샷 2023-10-28 오후 2 45 46" src="https://github.com/briiidgehong/cote-essential/assets/73451727/ee61d7c9-b05e-4cdf-8945-5a4035be4c32">

---
<img width="787" alt="스크린샷 2023-10-28 오후 2 46 24" src="https://github.com/briiidgehong/cote-essential/assets/73451727/201d72d9-2105-425d-9fd8-d360a5e18657">
<img width="750" alt="스크린샷 2023-10-28 오후 2 46 40" src="https://github.com/briiidgehong/cote-essential/assets/73451727/daa0d3ca-4189-47bc-b2d9-85c44a83bdaa">
<img width="554" alt="스크린샷 2023-10-28 오후 2 46 43" src="https://github.com/briiidgehong/cote-essential/assets/73451727/b3120ccb-81f0-4ed0-a909-b1315dccd118">

---
<img width="398" alt="스크린샷 2023-10-28 오후 2 56 15" src="https://github.com/briiidgehong/cote-essential/assets/73451727/2e476032-90eb-4247-bbb1-1c3d8d40a865">
<img width="759" alt="스크린샷 2023-10-28 오후 2 56 24" src="https://github.com/briiidgehong/cote-essential/assets/73451727/9aabed8f-7e5a-4753-958b-0ccd2324ea02">

---
<img width="792" alt="스크린샷 2023-10-28 오후 3 01 26" src="https://github.com/briiidgehong/cote-essential/assets/73451727/21202815-7f0c-4f69-a18e-51d4966babd1">
<img width="690" alt="스크린샷 2023-10-28 오후 3 01 55" src="https://github.com/briiidgehong/cote-essential/assets/73451727/69ab021b-2264-49fd-a2d7-7e73df869894">


</details>


<details>
<summary> 그리디 </summary>
### 문제
---
<img width="776" alt="스크린샷 2023-10-28 오후 3 08 08" src="https://github.com/briiidgehong/cote-essential/assets/73451727/aeb40e31-2acc-4209-b267-81ac8bbd84b6">

---
<img width="805" alt="스크린샷 2023-10-28 오후 3 04 49" src="https://github.com/briiidgehong/cote-essential/assets/73451727/34a0f117-a6be-4b34-8e27-e7133dd4da9f">
<img width="798" alt="스크린샷 2023-10-28 오후 3 04 35" src="https://github.com/briiidgehong/cote-essential/assets/73451727/e712b640-bee8-48c6-afa3-0a9c123d3cf1">

---
<img width="785" alt="스크린샷 2023-10-28 오후 3 06 10" src="https://github.com/briiidgehong/cote-essential/assets/73451727/faddc15a-dcea-4efa-9e75-27f001d5e142">
<img width="768" alt="스크린샷 2023-10-28 오후 3 06 28" src="https://github.com/briiidgehong/cote-essential/assets/73451727/3da91dc5-9a87-4890-98d0-cf43dba28a76">

---
<img width="788" alt="스크린샷 2023-10-28 오후 3 06 56" src="https://github.com/briiidgehong/cote-essential/assets/73451727/c056eeea-2eb6-4a66-811a-2fd574ebcb16">
<img width="778" alt="스크린샷 2023-10-28 오후 3 07 09" src="https://github.com/briiidgehong/cote-essential/assets/73451727/70bda0b2-54b5-4442-8603-c8720b9e8710">


</details>


<details>
<summary> MST (union-find / 크루스칼 / 프림) </summary>
### 문제 - union find
---
<img width="786" alt="스크린샷 2023-10-28 오후 2 51 05" src="https://github.com/briiidgehong/cote-essential/assets/73451727/fd8bec69-f538-4319-9b5a-038f2e0d9ccc">
<img width="653" alt="스크린샷 2023-10-28 오후 2 51 33" src="https://github.com/briiidgehong/cote-essential/assets/73451727/7b83e002-1048-44f2-a820-1e3812b112de">

### 문제 - 크루스칼
---

<img width="393" alt="스크린샷 2023-10-28 오후 2 52 45" src="https://github.com/briiidgehong/cote-essential/assets/73451727/16db225c-668b-4304-8153-3a72ea51fd8c">
<img width="754" alt="스크린샷 2023-10-28 오후 2 52 53" src="https://github.com/briiidgehong/cote-essential/assets/73451727/ef432a54-8f1c-4b40-9d3f-e1f439ba9a55">

### 문제 - 프림
<img width="790" alt="스크린샷 2023-10-28 오후 2 53 57" src="https://github.com/briiidgehong/cote-essential/assets/73451727/9207acee-38f8-4c1c-a645-d4f0f5010e96">
선택 순서는 3 - 2 - 5 - 6 - 4 - 7 이 됩니다.

https://github.com/briiidgehong/cote-essential/assets/73451727/a254fa84-38b7-4649-8fdb-0146a9e89a77

</details>

<details>
<summary> 최단거리(다익스트라, 플로이드 와샬) </summary>

### 문제 - 다익스트라
---
<img width="766" alt="스크린샷 2023-10-28 오후 2 48 17" src="https://github.com/briiidgehong/cote-essential/assets/73451727/2e359361-dd0f-43fe-b13f-c15ad9d18c12">
<img width="437" alt="스크린샷 2023-10-28 오후 2 49 11" src="https://github.com/briiidgehong/cote-essential/assets/73451727/893d1c3a-612c-4b77-ab79-62967fe5dace">
<img width="760" alt="스크린샷 2023-10-28 오후 2 49 29" src="https://github.com/briiidgehong/cote-essential/assets/73451727/f6c12dde-9d66-4a51-834f-65b9bd2013d6">

### 문제 - 플로이드 와샬
---
<img width="789" alt="스크린샷 2023-10-28 오후 2 50 12" src="https://github.com/briiidgehong/cote-essential/assets/73451727/831ff6bb-383f-4127-9146-db5171c62f73">
<img width="652" alt="스크린샷 2023-10-28 오후 2 50 26" src="https://github.com/briiidgehong/cote-essential/assets/73451727/d2b387af-62b0-4673-9bed-67755efc4bc0">

### 문제 - BFS 활용
---
<img width="786" alt="스크린샷 2023-10-28 오후 2 59 46" src="https://github.com/briiidgehong/cote-essential/assets/73451727/f6b75549-8fc6-47ea-97cf-5c805b122b64">
답: 1 4 5 6 7

https://github.com/briiidgehong/cote-essential/assets/73451727/e72568c3-54ea-4046-bfe6-3058c537a35a



</details>


## 기타 알고리즘 유형
<details>
<summary> 위상정렬 </summary>

### 문제
---
<img width="789" alt="스크린샷 2023-10-28 오후 2 57 24" src="https://github.com/briiidgehong/cote-essential/assets/73451727/b574c392-3a3b-4843-846c-3d2d1300eb35">
답: 3 4 5 6 2 1 8 9 7
https://github.com/briiidgehong/cote-essential/assets/73451727/7ed82f54-1e8f-4a38-84f3-e4e5d7f72ed1

---
<img width="782" alt="스크린샷 2023-10-28 오후 2 58 20" src="https://github.com/briiidgehong/cote-essential/assets/73451727/20248a27-a34a-46de-8ba5-6be2e27e55f5">
답: 3 4 5 2 1 6 8 9 7
https://github.com/briiidgehong/cote-essential/assets/73451727/1bc84762-755d-4161-abb6-4413380ff9cd



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






