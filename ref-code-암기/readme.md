
```
1. 10진수에서 N진수 변환함수 예시코드를 작성하시오.
2. BFS 예시코드를 작성하시오.
3. 백트래킹 예시코드를 작성하시오.
4. 다익스트라 예시코드를 작성하시오.
5. union-find 예시코드를 작성하시오.
6. 최소스패닝트리(=크루스칼) 예시코드를 작성하시오.
7. 소수판별 예시코드를 작성하시오. (개별판별, 범위판별 에라토스테네스의 체)

```

```
1.
def convert(num, criteria):
	str_num = ''
	while True:
		num, rest = divmod(num, criteria)
		str_num = str(rest) + str_num
		if num == 0:
			break
	return str_num

2.
# BFS
from collections import deque
queue = deque()
queue.append((idx_y, idx_x))
count = 1
while queue:
	poped_y, poped_x = queue.popleft()
	for move_y, move_x in move:
		moved_y = move_y + poped_y
		moved_x = move_x + poped_x
		if moved_y >= 0 and moved_y < y and moved_x >= 0 and moved_x < x:
			if visited[moved_y][moved_x] == False and map_list[moved_y][moved_x] == 1:
				count += 1
				visited[moved_y][moved_x] = True
				queue.append((moved_y, moved_x))
count_list.append(count)

3.
count = 0
def dfs_recur(num_list):
	global count
	if len(num_list) == N:
		count += 1
		return
	idx_y = len(num_list)
	for idx_x in range(N):
		flag = True
		for each_y, each_x in num_list:
			if each_x == idx_x:
				flag = False
				break
			if abs(idx_y - each_y) == abs(idx_x - each_x):
				flag = False
				break
		if flag:
			dfs_recur(num_list+[(idx_y, idx_x)])
dfs_recur([])

4.
"min_table과 heapq를 이용한 다익스트라"

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

5.
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

6.
최소스패닝 트리란?
"사이클이 없고, 노드가 모두 이어져 있는 그래프 (최소의 가중치로)"
"MST는 크루스칼로 푼다."
"parent/union/find 사용, 소팅 후 사이클 판별해가며 전체 간선 중 작은것부터 연결"

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


```
