## 코딩테스트 ESSENTAIL
주요 핵심 알고리즘을 문제를 기반으로 정리합니다. 

정말 실전에 쓰이고 필요한 핵심 내용들만 최대한 간추려서 작성합니다.

시간이 지나도 빠르게 본실력으로 돌아올수 있도록 오답노트를 함께 작성합니다.

"결국 코테는 문제를 추려서 지속적으로 반복하는것이 정답"

- 추리는 문제 기준: 백준 풀고 / 정답률이 45% 이상 / 최소 35% 이상만 골라풀자
- 문제접근:
시간복잡도 계산 / 종이와 펜으로 계산하고 알고리즘 정하고 코딩
1초에 2천만개 연산 가능
    주어진 시간제한이 2초이면 4천만회 연산이 한계
    데이터 개수가 1000개라면
        O(N^2) = 백만회
        O(NlogN) = 3000회
    데이터 개수가 10^5 10만회라면,
        O(N^2) = 10^10 -> 4천만회보다 크므로 쓸수 없음
        O(NlogN) = 50만 -> 4천회보다 작으므로 쓸수 있음 -> NlogN을 사용하는 알고리즘을 떠올려 코딩!

- 유형별로 10~15문제만 반복숙달(거의 외울정도로) / 총 80문제정도만 하면된다.
- "코딩테스트 준비" 가 아니라 한입거리를 반복적으로 진행 -> 백준문제2문제 풀기

1. 탐색 - 이진탐색 / 완전탐색 / 그래프탐색 / 백트래킹
2. 자료구조 - 해싱/파싱/스택/힙/트리/문자열
3. BFS / DFS
4. 그리디
5. DP

해당 유형의 골드 4~5정도 풀수 있는 수준이 되면 합격권
이외 크루스칼 / 다익스트라 / 트라이 등의 기타 알고리즘을 더하면 합격 안정권

```
백준 10816, 백준 1439, 백준 10799, 백준 1992, 백준 9012, 백준 2447, 백준 10101,
백준 14503, 백준 3040, 백준 11403, 백준 11651, 백준 1789, 백준 15649, 백준 15650,
백준 11497, 프로그래머스 42895, 백준 9663, 백준 2630, 백준 1446, 백준 1094, 백준 4307,
백준 4485, 백준 1520, 백준 9084, 백준 11758, 백준 1991, 백준 17779, 백준 11053, 백준 13869,
백준 9251, 백준 1766, 백준 11725, 백준 10845, 백준 1004, 백준 4256, 백준 11724, 백준 1337,
백준 7490, 백준 2022, 백준 1340, 백준 17140, 백준 1389, 백준 1158, 백준 1316, 백준 10815,
백준 17266, 백준 7795, 백준 11054, 백준 8958, 백준 7576, 백준 1912, 백준 9020, 백준 16236,
백준 1167, 백준 2437, 백준 15686, 백준 9095, 백준 1194, 백준 25644, 백준 2170, 백준 1922,
백준 17070, 백준 1525, 백준 18352, 백준 1918, 백준 14725, 백준 7569, 백준 2887, 백준 12015,
백준 1007, 백준 1197, 백준 10942, 백준 1509, 백준 1194, 백준 1562, 백준 1799
```

<img width="1309" alt="스크린샷 2023-10-31 오후 3 39 27" src="https://github.com/briiidgehong/cote-essential/assets/73451727/438397b0-79b2-4ad2-a129-54e3a89652c0">

## 주요 알고리즘 유형
<details>
<summary> 그리디 </summary>

```
PREVIEW:
"때로는 당장 눈앞의 최선이, 최고의 결과를 가져온다."
반례 찾아보고 없으면 그리디로 풀고 있다면 다른방법(DP-메모이제이션) 풀어보기
```
---

## 기본문제1 - 최소동전갯수 - 백준
<img width="1178" alt="스크린샷 2023-11-01 오후 3 22 45" src="https://github.com/briiidgehong/cote-essential/assets/73451727/9235a7be-7063-47be-a109-84f8e6b8fbac">

```
# 그리디 -> k를 큰동전부터 시작해서 동전갯수대로 나눈뒤 나머지 값을 갱신

n, k = map(int, input().split())
coin_list = []
for each in range(n):
    coin_list.append(int(input()))

count = 0
for each in reversed(coin_list):
    if k >= each:
        count += int(k // each) # 몫
        k = k % each # 나머지
    if k == 0:
        break
print(count)
```
---

## 기본문제2 - 1이 될때까지 - 이코테
<img width="786" alt="스크린샷 2023-11-01 오후 4 43 59" src="https://github.com/briiidgehong/cote-essential/assets/73451727/32290d1d-690c-4c8c-9725-c372d86cf827">

```
# 그리디 -> 주어진 N에 대해서 최대한 많이 나누기 진행
# N이 100억 이상의 큰수라 가정하고 O(N)이 아닌 O(logN)을 가지도록 코드 작성
# 즉, 나누기횟수 만큼의 시간복잡도를 가지도록 코드를 작성한다.
# while loop 한번 돌때마다 나누기가 시행되도록 n이 아닌 logN 시간 복잡도를 가질수 있다.
n, k = map(int, input().split())
result = 0

while True:
    # (N == K로 나누어떨어지는 수)가 될 때까지 1씩 빼기
    target = (n // k) * k
    result += (n - target)
    n = target
    # N이 k보다 작을 때(더 이상 나눌 수 없을 때) 반복문 탈출
    if n < k:
        break
    # K로 나누기
    result += 1
    n //= k

# 마지막으로 남은 수에 대하여 1씩 빼기
result += (n - 1)
print(result)
```
---

## 기본문제3 - 모험가길드 - 이코테
<img width="736" alt="스크린샷 2023-11-01 오후 4 57 14" src="https://github.com/briiidgehong/cote-essential/assets/73451727/2e8f6793-6167-40bc-a9d9-2dc3fa7493aa">

```
# 그리디
# 오름차순 정렬 이후에 가장 공포도가 낮은 모험가부터 확인
# '현재 그룹에 포함된 모험가의 수' 가 '현재 확인하고 있는 공포도' 보다 크거나 같다면 이를 그룹으로 설정

group = 0 # 총 그룹수
current_group_num = 0 # 현재 그룹에 포함된 모험가의 수

for each_panic_rate in list:
    current_group_num += 1
        if current_group_num >= each_panic_rate:
            group += 1
            current_group_num = 0
print(group)
```

---

## 기본문제4 - 1로 만들기 - 백준 - 그리디로 착각할만한 DP
<img width="1162" alt="스크린샷 2023-11-02 오전 12 03 15" src="https://github.com/briiidgehong/cote-essential/assets/73451727/5cef18bd-7889-4035-b657-6da8d6dcd459">

```
# 그리디로 접근
# 무조건 큰수로 먼저 나누는것이 가장 최선이다.
# 반례를 찾아보자.
# 10 -> 5 -> 4 -> 2 -> 1
# 10 -> 9 -> 3 -> 1
# 반례가 있다. 그리디로 풀긴 어렵다.
# DP 상향식으로 아래서부터 DP 테이블을 채우자.

# DP 접근
# DP[N] = min((N - 1을 1로 만들기 위한 최소횟수 + 1), 2나 3으로 나누어지면 DP[몫] + 1)

# DP[8] = min(DP[8/2]+1, DP[7]+1)
# DP[9] = min(DP[9/3]+1, DP[8]+1)
# DP[10] = min(DP[10/2]+1, DP[9]+1)
# DP[11] = min(DP[10]+1)
# DP[12] = min(DP[12/3]+1, DP{12/2}+1, DP[11]+1)

```
---

## 기본문제5 fraction knapsack vs 0/1 knapsack
```
knapsack 배낭 알고리즘
배낭에 담을 수 있는 n개의 물건이 존재
각 물건에는 "무게"와 "가치"가 존재한다.
어떤것을 어떻게 담아야 가장 많은 가치를 담을 수 있을까?

물건을 쪼갤수 있는 fraction knapsack 알고리즘과 쪼갤수 없는 0/1 knapsack 알고리즘으로 나뉜다.

fraction knapsack 은 그리디풀이로 무게대비가격(가격/무게)이 높은것부터 담으면 쉽게 풀이가 가능하다.

0/1 knapsack 의 경우는
무게가 가벼운 것 부터 담았을 때가 최적이지 않은 경우도 있고,
가격이 높은 것 부터 담았을 때 역시 최적이 아닌 경우가 있으므로,
동적계획법을 통해 풀이가 가능하다.


```
## fraction knapsack - greedy
<img width="805" alt="스크린샷 2023-10-28 오후 3 04 49" src="https://github.com/briiidgehong/cote-essential/assets/73451727/34a0f117-a6be-4b34-8e27-e7133dd4da9f">
답: 10.333

## 0/1 knapsack - dp
<img width="525" alt="스크린샷 2023-11-02 오후 4 00 17" src="https://github.com/briiidgehong/cote-essential/assets/73451727/b97e5bb4-9466-4323-985c-ad0ea5fe4f9a">
<img width="723" alt="스크린샷 2023-11-02 오후 4 01 04" src="https://github.com/briiidgehong/cote-essential/assets/73451727/7bc6da4d-bf0c-4035-a865-00e43a086ac8">
<img width="717" alt="스크린샷 2023-11-02 오후 4 01 15" src="https://github.com/briiidgehong/cote-essential/assets/73451727/28f2b198-6c87-4c27-9573-c88c0e76dde0">

<img width="647" alt="스크린샷 2023-11-02 오후 4 31 13" src="https://github.com/briiidgehong/cote-essential/assets/73451727/f7fb0611-077c-4772-a083-67910d8db035">

```
보석 = [1,2,3,4,5]
value = [4,1,2,6,3]
weight = [3,1,4,5,2]

경우의수 = 각각의 보석을 넣지않는경우/넣는경우
= [0/1,0/1,0/1,0/1,0/1] = 2^5 = 32

DP[보석][지금까지 선택한 보석 무게의 합] 
= 가능한 보석 가치의 최대 합

i번째 보석을 가방에 넣는 경우 
DP[i][j]는 DP[i-1][j-weight[i]] + value[i]
다만, 이 경우에는 j가 weight[i]보다 같거나 커야지만 말이 되므로
이러한 경우에만 고려가 가능합니다.

i번째 보석을 가방에 넣지 않는 경우
보석을 가방에 넣지 않았으므로 DP[i][j]는 dp[i-1][j]

DP[i][j] = 
max(DP[i-1][j-weight[i]] + value[i], dp[i-1][j])

1) DP[2][3] = max(DP[1][2]+1, DP[1][3]) = 4
2) DP[3][5] = max(DP[2][1]+2, DP[2][5]) = 3
3) DP[3][8] = max(DP[2][4]+2, DP[2][8]) = 7
4) DP[5][1] = max(DP[4][1]) X: DP[4][1-2]+3 = 1
5) DP[5][7] = max(DP[4][5]+3, DP[4][7]) = 9

DP[1][1]=-
DP[1][2]=-
DP[1][3]=4
DP[1][4]=-
DP[1][5]=-
DP[1][6]=-
DP[1][7]=-
DP[1][8]=-

DP[2][1]=1
DP[2][2]=1
DP[2][3]=4
DP[2][4]=5
DP[2][5]=-
DP[2][6]=-
DP[2][7]=-
DP[2][8]=-
```
</details>

<details>
<summary> DP </summary>

```
PREVIEW:
"DP는 이전값 점화식"

점화식 생성 / 이전 값을 재활용하는 메모이제이션
직접해봐야 점화식 나온다. -> n=1 부터 하나씩 그려보면서 규칙 찾기

예시: 1~10 숫자 중, 각각 이전값들을 합한 값 구하기
시간복잡도:
단순 for loop: O(N^2) [N개 각각에 대해 최대 N번(10+9+8+''')의 연산이 들어간다.]
DP:O(N) [DP 테이블 활용]

```
---

## 기본문제1 - 백준 11726 - 2xN 타일링
```
1. 아이디어
점화식: An = (An-1) + (An-2)
N값 구하기 위해, for문 3부터 N까지의 값을 구해주기
2. 시간복잡도
- O(N)
3. 자료구조
- DP table [0]*N
```
```
n = int(input())

dp_table = [0] * (n+1)

# DPn = DP(n-1) + DP(n-2)
# DP1 = 1 / DP2 = 2 / DP3 = 3 / DP4=5
dp_table[1] = 1
if n > 1:
    dp_table[2] = 2
for idx in range(3, n+1):
    dp_table[idx] = dp_table[idx-1] + dp_table[idx-2]

print(dp_table[n] % 10007)
```

```
# 피보나치 수열 - 바텀업
n = 99
dp_table = [0] * (n+1)

dp_table[1] = 1
dp_table[2] = 1

for idx in range(3, n+1):
    dp_table[idx] = dp_table[idx-1] + dp_table[idx-2]

print(dp_table[n]) # 218922995834555169026
```
---

## 기본문제2 - 이코테 - 개미전사
<img width="671" alt="스크린샷 2023-08-20 오전 10 14 07" src="https://github.com/briiidgehong/algorithm/assets/73451727/e346740a-5357-4ca0-b981-41a377382e15">

```
# (자기 자신 + 누적값)과 (자기 자신 바로 앞 값)을 비교
def solution(array):
    accumulate_sum = []
    for idx, each in enumerate(array):
        if idx == 0:
            accumulate_sum.append(each)
        elif idx == 1:
            accumulate_sum.append(max(array[0], array[1]))
        else:  # idx >= 2
            accumulate_sum.append(
                max((accumulate_sum[idx - 2] + each), accumulate_sum[idx - 1])
            )
    return accumulate_sum[-1]

print(solution([1, 3, 1, 5]))  # 8

```
---

## 기본문제3 - 이코테 - 1로 만들기
```
# bottom - up !!!
# dp - table !!!
```

<img width="474" alt="스크린샷 2023-08-20 오전 11 02 25" src="https://github.com/briiidgehong/algorithm/assets/73451727/db40415a-460f-4db5-96b7-0aa87387efbc">

```
def solution(num):
    dp_table = [0] * 30001
    for each_num in range(2, num + 1):
        temp_list = []
        # /5
        if each_num % 5 == 0:
            temp_list.append(dp_table[each_num // 5])
        # /3
        if each_num % 3 == 0:
            temp_list.append(dp_table[each_num // 3])
        # /2
        if each_num % 2 == 0:
            temp_list.append(dp_table[each_num // 2])
        # -1
        temp_list.append(dp_table[each_num - 1])
        dp_table[each_num] = min(temp_list) + 1
    return dp_table[num]

print(solution(26))  # 3

```
---

## 기본문제4 - 이코테 - 효율적인 화폐구성

<img width="504" alt="스크린샷 2023-08-20 오후 12 19 46" src="https://github.com/briiidgehong/algorithm/assets/73451727/08894c38-8dff-479e-bf9f-8ce4e3df6e16">

```
# bottom - up !!!
# 최소 화폐 개수
def solution(array, target_num):
    dp_table = [0] * 10001
    min_num = min(array)
    for each in array:
        dp_table[each] = 1
    for num in range(target_num + 1):
        temp_list = []
        for each in array:
            if num >= each and num - each >= min_num:
                temp_list.append(dp_table[num - each] + 1)
        if len(temp_list) >= 1:
            dp_table[num] = min(temp_list)
    if dp_table[target_num] == 0:
        return -1
    else:
        return dp_table[target_num]

print(solution([2, 3, 5], 7))  # 5

```
---

## 기본문제5 - 이코테 - 금광

<img width="629" alt="스크린샷 2023-08-20 오후 4 20 19" src="https://github.com/briiidgehong/algorithm/assets/73451727/3b9c7fdb-baea-4465-9cd1-a43e973783be">

```
def solution(array):
    for idx in range(1, len(array[0])):
        for sub_idx, each in enumerate(array):
            temp_list = []
            if sub_idx == 0:
                temp_list.append(array[sub_idx][idx - 1])
                temp_list.append(array[sub_idx + 1][idx - 1])
            elif sub_idx == len(array) - 1:
                temp_list.append(array[sub_idx][idx - 1])
                temp_list.append(array[sub_idx - 1][idx - 1])
            else:
                temp_list.append(array[sub_idx - 1][idx - 1])
                temp_list.append(array[sub_idx][idx - 1])
                temp_list.append(array[sub_idx + 1][idx - 1])
            array[sub_idx][idx] += max(temp_list)

    temp_list = []
    for each in array:
        temp_list.append(each[-1])
    return max(temp_list)

print(solution([[1, 3, 3, 2], [2, 1, 4, 1], [0, 6, 4, 7]]))  # 19
print(solution([[1, 3, 1, 5], [2, 2, 4, 1], [5, 0, 2, 3], [0, 6, 1, 2]]))  # 16

```
---

## 기본문제6 - 이코테 - 병사 배치하기

<img width="763" alt="스크린샷 2023-08-21 오후 2 19 19" src="https://github.com/briiidgehong/algorithm/assets/73451727/efeaf4f3-019a-49cd-94e4-1bc2cfcc8807">

```
def solution(array):
    reverse_list = []
    for each in reversed(array):
        reverse_list.append(each)

    # LIS (가장 긴, 증가하는 부분수열 알고리즘)
    dp_table = [1] * len(reverse_list)
    for idx in range(1, len(reverse_list)):
        for sub_idx in range(idx):
            if reverse_list[idx] > reverse_list[sub_idx]:
                dp_table[idx] = max(dp_table[idx], dp_table[sub_idx] + 1)

    return len(reverse_list) - max(dp_table)

# 4 2 5 8 4 11 15
# 1 1 1 1 1 1  1
# 1 1 2 3 2 4  5
print(solution([15, 11, 4, 8, 5, 2, 4]))  # 2

```
---

## 기본문제7 - 코드트리 - 은행
<img width="791" alt="스크린샷 2023-10-28 오후 2 43 25" src="https://github.com/briiidgehong/cote-essential/assets/73451727/db9f6f39-c9aa-4a8a-97a8-6c3e256403a7">
<img width="774" alt="스크린샷 2023-10-28 오후 2 43 57" src="https://github.com/briiidgehong/cote-essential/assets/73451727/056f4d68-0bb4-42c0-85e3-2dd68a1dcc2b">
<img width="777" alt="스크린샷 2023-10-28 오후 2 44 02" src="https://github.com/briiidgehong/cote-essential/assets/73451727/ce1d719b-3511-43ae-aa5b-8933099f6d99">

---

## 기본문제8 - 코드트리 - 숫자 암호 만들기
<img width="594" alt="스크린샷 2023-10-28 오후 2 40 14" src="https://github.com/briiidgehong/cote-essential/assets/73451727/8275a1e6-bbce-4781-94cd-64f8e57bedce">
<img width="545" alt="스크린샷 2023-10-28 오후 2 41 15" src="https://github.com/briiidgehong/cote-essential/assets/73451727/23a7aba3-c042-4f91-8279-e8310932a7cf">
<img width="353" alt="스크린샷 2023-10-28 오후 2 42 26" src="https://github.com/briiidgehong/cote-essential/assets/73451727/fc417518-f907-49c8-b3df-62ae2c1c8693">

https://github.com/briiidgehong/cote-essential/assets/73451727/0c52acb3-1656-48c9-889c-f91dea5301d3

---

## 기본문제9 - 코드트리 - 가장 긴 증가하는 부분수열(LIS)
<img width="787" alt="스크린샷 2023-10-28 오후 2 45 11" src="https://github.com/briiidgehong/cote-essential/assets/73451727/cb2f11ea-6006-45c7-80a4-aae9f135d907">
<img width="764" alt="스크린샷 2023-10-28 오후 2 45 42" src="https://github.com/briiidgehong/cote-essential/assets/73451727/2266af8f-3619-41c4-baed-7621170f40a1">
<img width="433" alt="스크린샷 2023-10-28 오후 2 45 46" src="https://github.com/briiidgehong/cote-essential/assets/73451727/ee61d7c9-b05e-4cdf-8945-5a4035be4c32">

<img width="787" alt="스크린샷 2023-10-28 오후 2 46 24" src="https://github.com/briiidgehong/cote-essential/assets/73451727/201d72d9-2105-425d-9fd8-d360a5e18657">
<img width="750" alt="스크린샷 2023-10-28 오후 2 46 40" src="https://github.com/briiidgehong/cote-essential/assets/73451727/daa0d3ca-4189-47bc-b2d9-85c44a83bdaa">
<img width="554" alt="스크린샷 2023-10-28 오후 2 46 43" src="https://github.com/briiidgehong/cote-essential/assets/73451727/b3120ccb-81f0-4ed0-a909-b1315dccd118">

---

## 기본문제10 - 코드트리 - 정수 사각형
<img width="398" alt="스크린샷 2023-10-28 오후 2 56 15" src="https://github.com/briiidgehong/cote-essential/assets/73451727/2e476032-90eb-4247-bbb1-1c3d8d40a865">
<img width="759" alt="스크린샷 2023-10-28 오후 2 56 24" src="https://github.com/briiidgehong/cote-essential/assets/73451727/9aabed8f-7e5a-4753-958b-0ccd2324ea02">

---

## 기본문제11 - 코드트리 - 편집거리
<img width="792" alt="스크린샷 2023-10-28 오후 3 01 26" src="https://github.com/briiidgehong/cote-essential/assets/73451727/21202815-7f0c-4f69-a18e-51d4966babd1">
<img width="690" alt="스크린샷 2023-10-28 오후 3 01 55" src="https://github.com/briiidgehong/cote-essential/assets/73451727/69ab021b-2264-49fd-a2d7-7e73df869894">

---

</details>

<details>
<summary> 이진탐색 </summary>

```
PREVIEW:
정렬되어있을 경우 사용
인데스 기준, 시작점 / 끝점 / 중간점을 이용해 탐색 범위를 설정

*파라메트릭 서치
  - 최적화 문제를 결정 문제(예/아니오) 로 바꾸어 해결하는 기법
  - 최적화 문제: 어떤 함수의 값을 최대한 낮추거나/높이는 문제
  - 특정한 조건을 만족하는 가장 알맞은 값을 빠르게 찾는 최적화 문제
  - 일반적으로 코테에서 파라메트릭 서치 문제는 이진탐색을 이용한다.

* lower bound / upper bound = bisect left / right

시간복잡도:
어떤값 찾을때 O(N) -> O(logN)

핵심코드: *외워놓기
result = 0
while start <= end:
    mid = (start + end) // 2
    if list[mid] == target:
        result = mid
        break
    elif list[mid] > target:
        end = mid - 1
    else:
        start = mid + 1
```
---

## 기본문제1 - 백준 1920 - 수 찾기
```
a_num = int(input())
a_list = []
a_list = list(map(int, input().split()))
a_list.sort()

n = int(input())
n_list = list(map(int, input().split()))

result_list = []
for target in n_list:
    result = -1
    start = 0
    end = len(a_list) - 1
    while start <= end:
        mid = (start + end) // 2
        if a_list[mid] == target:
            result = mid
            break
        elif a_list[mid] > target:
            end = mid - 1
        else:
            start = mid + 1
    if result == -1: 
        result_list.append(0)
    else:
        result_list.append(1)
        
for each in result_list:
    print(each)
```
---

## 기본문제2 - 이코테 - 떡볶이 떡 자르기
```
def solution(target_meter, dduk_list):
    dduk_list.sort()

    # # 일반풀이: index[0] ~ index[-1] 까지 탐색
    # for each in reversed(range(dduk_list[0], dduk_list[-1])):
    #     if (
    #         sum([sub_each - each for sub_each in dduk_list if sub_each - each >= 1])
    #         == target_meter
    #     ):
    #         return each

    # 이진탐색 풀이
    start = dduk_list[0]
    end = dduk_list[-1]
    while start <= end:
        middle = int((start + end) / 2)
        calc = sum([each - middle for each in dduk_list if each - middle >= 0])
        if calc == target_meter:
            return middle
        elif calc > target_meter:
            start = middle + 1
        else:  # calc < target_meter
            end = middle - 1
    return 0

print(solution(6, [19, 15, 10, 17]))
```
---

## 기본문제3 - 이코테 - 정렬된 배열에서 특정 수의 개수 구하기
```
from bisect import bisect_left, bisect_right

## 정렬된 배열에서 특정 수의 개수 구하기
def solution(target_number, list):
    def count_by_range(list, left_num, right_num):
        # 2를 넣는다고 할때 왼쪽 어떤 인덱스에 넣어야 할지
        left_idx = bisect_left(list, left_num)
        # 2를 넣는다고 할때 오른쪽 어떤 인덱스에 넣어야 할지
        right_idx = bisect_right(list, right_num)
        return right_idx - left_idx
    return count_by_range(list, target_number, target_number)

print(solution(2, [1, 1, 2, 2, 2, 2, 3]))
```
---

</details>


<details>
<summary> BFS / DFS </summary>
  
```
PREVIEW:
    그래프 탐색:
    어떤것들이 연속해서 이어질때, 모두 확인하는 방법
    Graph: Vertex(어떤것) + Edge(이어지는것)
    
    그래프 탐색 종류:
    BFS - 너비우선탐색 - queue 
    DFS - 깊이우선탐색 - 재귀(백트래킹에 유용)

    단순 그래프 탐색이 필요한 경우에는 BFS로 풀고,
    백트래킹이 필요한 경우에는 DFS+재귀+백트래킹 으로 푼다.

시간복잡도:
    BFS: O(V+E)
    DFS: O(V+E)
핵심코드:

def DFS_solution(graph):
    dfs_search_list = []
    visited = [False] * len(graph) # 각 노드가 방문된 정보

    def dfs(graph, idx, visited):
        visited[idx] = True  # 현재노드를 방문처리
        dfs_search_list.append(idx)
        for sub_idx in graph[idx]:  # 현재노드와 연결된 다른 노드를 재귀적으로 방문
            if not visited[sub_idx]:
                dfs(graph, sub_idx, visited)
    dfs(graph, 1, visited)
    return dfs_search_list

from collections import deque
def BFS_solution(graph):
    dfs_search_list = []
    visited = [False] * len(graph)
    queue = deque([1]) 
    visited[1] = True # 현재 노드를 방문 처리
    while queue: # 큐가 빌때까지 반복
        poped = queue.popleft() # 하나의 원소를 뽑는다.
        dfs_search_list.append(poped)
        # 아직 방문하지 않은 인접한 원소들을 큐에 삽입
        for sub_idx in graph[poped]:
            if not visited[sub_idx]:
                queue.append(sub_idx)
                visited[sub_idx] = True
    return dfs_search_list

# 1~8
array = [[1, 2], [1, 3], [1, 8], [2, 7], [3, 4], [3, 5], [4, 5], [6, 7], [7, 8]]
grpah = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7],
]

# grapn 만들기
graph = []
temp_list = []
for each in array:
    temp_list.append(each[0])
    temp_list.append(each[1])
max_num = max(temp_list)  # 8

for each in range(0, max_num + 1):
    graph.append([])
for each in array:
    graph[each[0]].append(each[1])
    graph[each[1]].append(each[0])
for idx, each in enumerate(graph):
    if each is not None:
        graph[idx] = sorted(each)

print(DFS_solution(grpah))  # 12768345
print(BFS_solution(graph))  # 12387456

from itertools import product
product_list = list(product(*case_list))
```

<img width="921" alt="스크린샷 2023-07-19 오후 12 06 50" src="https://github.com/briiidgehong/cote/assets/73451727/9beb58cd-9a1f-473d-89f9-539d4d4c3f31">
<img width="917" alt="스크린샷 2023-07-19 오후 12 07 17" src="https://github.com/briiidgehong/cote/assets/73451727/5a3c4bdc-a472-4dd2-b11d-3f681ea07161">

---

### 문제


</details>


<details>
<summary> 백트래킹 </summary>

```
PREVIEW:
시간복잡도:
핵심코드:
```
---

### 문제
---

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
<summary> 구현 / 시뮬레이션 / 완전탐색 </summary>
https://www.youtube.com/watch?v=2zjoKjt97vQ&list=PLRx0vPvlEmdAghTr5mXQxGpHjWqSz0dgC&index=2

```
PREVIEW:
시간복잡도:
핵심코드:
```
---

### 문제
---

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

```
PREVIEW:
시간복잡도:
핵심코드:
```
---

### 문제
---

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
PREVIEW:
시간복잡도:
핵심코드:
```
---

### 문제
---

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

### PREVIEW
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

<details>
<summary> 가장 긴 증가하는 / 감소하는 부분수열 </summary>
ㅁㄴㅇ
</details>

## 오답노트






