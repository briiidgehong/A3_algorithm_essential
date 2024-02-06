## USE
```
빠르게 훑어볼 수 있도록 복기문제 운용 // 중복되는 내용은 제외시키고 되도록 핵심만 취한다.
3T까지만 순서대로 풀고 4T부터는 랜덤풀이
```

```
구현/시뮬레이션:
2979P, 1157P, 10431P, 8979PT, 7568P
소수판별: 1978P, 1929P, 2581P, 4948, 11653, 2023
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
15649, 15650, 15651, 15652, 15654, 1182, 9663P, 프로그래머스 여행경로PT

다익스트라, 플로이드 와샬 (최단경로 알고리즘):
1238PT, 4485PT, 11404PT, 이코테 전보P, 이코테 미래도시PT

union find:
1717PT, 1976PT, 10775T

크루스칼 / 프림 (MST)
1197PT, 1922PT, 1647PT, 4386PT



```

## ref code
```
1. 소수판별
# 특정 수의 소수 판별
def is_prime(x):
	import math
    for i in range(2, int(math.sqrt(x)) + 1):
        if x % i == 0:
            return False 
    return True

# 에라토스테네스의 체 (N보다 작거나 같은 모든 소수를 찾을 때 사용)
def is_prime_range(N):
	import math
	prime_table = [True for _ in range(N+1)]
	for idx in range(2, int(math.sqrt(N)) + 1):
		if prime_table[idx] == True:
			sum_idx = (2 * idx)
			while sum_idx <= N:
				prime_table[sum_idx] = False
				sum_idx += idx
	primes = []
	for idx in range(2, N+1):
		if prime_table[idx] == True:
			primes.append(idx)
	return primes
```
```
2. 순열 / 조합 / 중복순열 / 중복조합

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

# 조합 (15650, 1182)
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
```
3. 최단경로 알고리즘 / 다익스트라, 플로이드 와샬
가중치가 존재하지 않는 간선의 최단경로는 BFS depth 로 구하고,
가중치가 존재하는 간선의 최단경로는 다익스트라, 플로이드 와샬을 사용한다.

다익스트라는 한 정점에서 다른 모든 정점 사이의 최단거리를 판별, 시간복잡도는 O(ElogV)
구현은 "min_table과 heapq를 이용한 다익스트라" 을 이용한다.
*힙을 사용하는 이유 -> 이전에 계산해둔 값이 그 단계에서 최소값이라는 것이 보장되기 때문에 갱신 횟수가 현저히 적어진다.
 (https://jaegualgo.blogspot.com/2017/07/dijkstra-priority-queue.html)

def dij(start, end):
	import heapq
	queue = []
	min_table = [int(1e9)] * (N+1)
	min_table[start] = 0
	heapq.heappush(queue, (0, start))
	while queue:
		poped_cost, poped_node = heapq.heappop(queue)
		for next_cost, next_node in graph[poped_node]:
			if min_table[next_node] <= next_cost + poped_cost:
				continue
			else:
				min_table[next_node] = next_cost + poped_cost
				heapq.heappush(queue, (next_cost + poped_cost, next_node))
	return min_table[end]

플로이드 와샬은 모든 정점에서 다른 모든 정점 사이의 최단거리를 판별, 시간복잡도는 O(V^3)
구현은 "점화식을 이용한 플로이드 와샬, 점화식 Dab = min(Dab, Dak + Dkb)" 을 이용한다.

for k in range(1, N+1):
	for a in range(1, N+1):
		for b in range(1, N+1):
			dp[a][b] = min(dp[a][b], dp[a][k] + dp[k][b])
```
```
4. union find
parent_table = [0] * (N+1)
for idx in range(1, N+1):
	parent_table[idx] = idx
 
def find_parent(x):
	if parent_table[x] != x:
		parent_table[x] = find_parent(parent_table[x])
	return parent_table[x]
 
def union_parent(a,b):
	a = find_parent(a)
	b = find_parent(b)
	if a <= b:
		parent_table[b] = a
	else:
		parent_table[a] = b
```
```
5. 최소 스패닝 트리 (MST): // MST 문제는 우선 크루스칼로 푼다. (프림 구현시 헷갈림)
	사이클:
		시작점과 끝이 같은 노드인 형태
	트리 < 그래프:
		트리는 그래프의 특수한 형태
		그래프: 사이클이 존재
		트리: 사이클이 존재하지 않음
	
	스패닝 트리: <<사이클이 없고, 노드가 모두 이어져 있는 그래프>>
	최소 스패닝 트리: 스패닝 트리중 가중치의 합이 가장 작은 스패닝 트리

	즉, 최소스패닝트리란
		<<사이클이없고, 노드가 모두 이어진, 가중치의 합이 가장 적은 트리>>
	
	MST 알고리즘
		크루스칼: 
			<<"parent/union/find 사용, 소팅 후 사이클 판별해가며 전체 간선 중 작은것부터 연결">>
			O(ElogE) = 100 000 * 5
		프림:
			"힙 사용하여, 현재 연결된 트리에 이어진 간선중 가장 작은것을 추가"
			O(ElogV) = 100 000 * 4

<<프림은 시작점을 정하고, 시작점에서 가까운 정점을 선택하면서 트리르 구성 하므로 그 과정에서 사이클을 이루지 않지만 크루스칼은 시작점을 따로 정하지 않고 최소 비용의 간선을 차례로 대입 하면서 트리를 구성하기 때문에 사이클 생성 여부를 항상 확인 해야한다.>>

크루스칼:
e_list = []
for _ in range(E):
	a, b, cost = map(int, input().split())
	e_list.append((cost, a, b))
e_list.sort()

parent = [0] * (N+1)
for idx in range(1, N+1):
	parent[idx] = idx

def find(x):
	if parent[x] != x:
		parent[x] = find(parent[x])
	return parent[x]
	
def union(a,b):
	a = find(a)
	b = find(b)
	if a <= b:
		parent[b] = a
	else:
		parent[a] = b

sum_cost = 0
for cost, a, b in e_list:
	a = find(a)
	b = find(b)
	if a == b:
		# union 시, 사이클 발생
		continue
	else:
		union(a,b)
		sum_cost += cost
print(sum_cost)

프림:
graph = {}
for idx in range(1, N+1):
	graph[idx] = []
for _ in range(E):
	a, b, cost = map(int, input().split())
	graph[a].append((cost, b))
	graph[b].append((cost, a))

import heapq
queue = []
heapq.heappush(queue, (0, 1)) # (cost, node)
visited = [False] * (N+1)
sum_cost = 0
while queue:
    poped_cost, poped_node = heapq.heappop(queue)
    # 이미 방문된 노드들이, queue에는 들어있어서 걸러줘야 함
    if visited[poped_node] == False: 
        visited[poped_node] = True
        sum_cost += poped_cost

        for each_cost, each_node in graph[poped_node]:
            heapq.heappush(queue, (each_cost, each_node))
print(sum_cost)

```
[ref. 크루스칼 vs 프림](https://velog.io/@fldfls/%EC%B5%9C%EC%86%8C-%EC%8B%A0%EC%9E%A5-%ED%8A%B8%EB%A6%AC-MST-%ED%81%AC%EB%A3%A8%EC%8A%A4%EC%B9%BC-%ED%94%84%EB%A6%BC-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)


