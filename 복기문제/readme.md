## USE
```
빠르게 훑어볼 수 있도록 복기문제 운용 // 중복되는 내용은 제외시키고 되도록 핵심만 취한다.
```

```
구현/시뮬레이션:
2979, 1157, 10431, 8979, 7568
소수판별: 1978, 1929, 2581, 4948, 11653, 2023
최대공약수, 최소공배수: 2609, 9613, 2485, 1735, 3036
진수변환: 2998, 3460, 8741

완전탐색:
2231, 2309, 10448, 1436, 1018, 17484

그리디:
11047, 1343, 14916, 2828, 이코테 모험가길드, 이코테 1이 될떄까지

DP:
2193, 9655, 1463, 11726, 2579, 12865
LIS, LCS: 11053, 9251, 9252
이코테 개미전사, 1로 만들기, 효율적인 화폐구성, 금광

이진탐색:
30792, 1920, 2805, 10816

투포인터:
2559, 3273, 2470, 1806, 1644

BFS, DFS:
1926, 2667, 2178, 1260, 2644, 1012, 7562, 11724, 11403

백트래킹:
15649, 15650, 15651, 15652, 15654, 1182, 9663, 프로그래머스 여행경로

다익스트라, 플로이드 와샬 (최단경로 알고리즘):
1238, 4485, 11404, 이코테 전보, 이코테 미래도시

union find:
1717, 1976, 10775

크루스칼 / 프림 (MST)
1197, 1922, 1647, 4386



```

## ref code
```
N = 3
M = 2 #or 3
n_list = [2,4,3]

# 순열 (15649, 15654)
from itertools import permutations
print(f"itertools순열: {list(permutations(n_list, M))}")

result_list = []
visited = [False] * N # index visited
def permutations_2(num_list):
	if len(num_list) == M:
		result_list.append(num_list)
		return
	for idx in range(N):
		if visited[idx] == False:
			visited[idx] = True
			permutations_2(num_list + [n_list[idx]])
			visited[idx] = False
permutations_2([])
print(f"순열구현: {result_list}")		

# 조합 (15650)
from itertools import combinations
print(f"itertools조합: {list(combinations(n_list, M))}")

result_list = []
def combinations_2(start, num_list):
	if len(num_list) == M:
		result_list.append(num_list)
		return
	for idx in range(start, N):
        # check point! start + 1 이 아닌 idx + 1
		combinations_2(idx+1, num_list+[n_list[idx]]) 
combinations_2(0, [])
print(f"조합구현: {result_list}")

# 중복순열 (15651)
from itertools import product
# repeat은 중복 할 수 있는 최대 횟수
print(f"itertools중복순열: {list(product(n_list, repeat=M))}")

result_list = []
def permutations_2(num_list):
    if len(num_list) == M:
        result_list.append(num_list)
        return
    for idx in range(N):
        permutations_2(num_list + [n_list[idx]])
permutations_2([])
print(f"중복순열구현: {result_list}")	

# 중복조합 (15652)
from itertools import combinations_with_replacement
print(f"itertools중복조합: {list(combinations_with_replacement(n_list, M))}")

result_list = []
def combinations_2(start, num_list):
	if len(num_list) == M:
		result_list.append(num_list)
		return
	for idx in range(start, N):
		combinations_2(idx, num_list+[n_list[idx]]) 
combinations_2(0, [])
print(f"중복조합구현: {result_list}")
```
