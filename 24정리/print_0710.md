```
# 1238 파티 / 다익스트라

N, M, X = map(int, input().split())
graph = {}
for idx in range(1, N+1):
	graph[idx] = []
for _ in range(M):
	start, end, cost = map(int, input().split())
	graph[start].append((cost, end))


def dij(start):
	import heapq
	queue = []
	min_table = [int(1e9)] * (N+1)
	min_table[start] = 0
	heapq.heappush(queue, (0, start))
	while queue:
		poped_cost, poped_node = heapq.heappop(queue)
		for next_cost, next_node in graph[poped_node]:
			if min_table[next_node] <= poped_cost + next_cost:
				continue
			else:
				min_table[next_node] = poped_cost + next_cost
				heapq.heappush(queue, (min_table[next_node], next_node))
	return min_table

x_min_table = dij(X)

result_list = []
for idx in range(1, N+1):
	if idx == X:
		continue
	total_cost = 0
	total_cost += dij(idx)[X]
	total_cost += x_min_table[idx]
	result_list.append(total_cost)
	
print(max(result_list))
```

```
# 1926 그림 / BFS

"""
BFS: 너비우선탐색

6 5
1 1 0 1 1
0 1 1 0 0
0 0 0 0 0
1 0 1 1 1
0 0 1 1 1
0 0 1 1 1

"""
y, x = map(int, input().split())
map_list = []
for _ in range(y):
    input_list = list(map(int, input().split()))
    map_list.append(input_list)

move = [(-1,0),(1,0),(0,-1),(0,1)] # (y,x)상하좌우
visited = list([False] * x for _ in range(y))

count_list = []
for idx_y in range(y):
    for idx_x in range(x):
        if visited[idx_y][idx_x] == False and map_list[idx_y][idx_x] == 1:
            visited[idx_y][idx_x] = True

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

print(len(count_list))
if len(count_list) == 0:
    print(0)
else:
    print(max(count_list))
```

```
# 1197 최소 스패닝 트리 / MST - 크루스칼

"""
MST: 최소 스패닝 트리: 크루스칼
가중치가 적은것부터 연결, 사이클이 생기지 않게, parent / union-find 사용

3 3
1 2 1
2 3 2
1 3 3
"""

V, E = map(int, input().split())

parent = [0] * (V+1) # [0,1,2,3 ''']
for idx in range(1, V+1):
    parent[idx] = idx
def find(x):
    if x != parent[x]:
        parent[x] = find(parent[x])
    return parent[x]
def union(a, b):
    a = find(a)
    b = find(b)
    if a <= b:
        parent[b] = a
    else:
        parent[a] = b

edges = []
for _ in range(E):
    start, end, cost = map(int, input().split())
    edges.append((cost, start, end))

edges.sort()
mst_cost = 0
for each_cost, each_start, each_end in edges:
    start = find(each_start)
    end = find(each_end)
    if start == end:
        continue
    else:
        mst_cost += each_cost
        union(each_start, each_end)
print(mst_cost)
```

```
# 9663 N-Queen

N = int(input())
count = 0
def dfs_recur(num_list): # [(idx_y, idx_x)]
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
			if abs(each_y - idx_y) == abs(each_x - idx_x):
				flag = False
				break
		if flag:
			dfs_recur(num_list+[(idx_y, idx_x)])
dfs_recur([])
print(count)
```

```
# 15649 N과 M

from itertools import permutations
N, M = map(int, input().split())

visited = [False] * (N+1)
result_list = []
def dfs_recur(num_list):
    if len(num_list) == M:
        result_list.append(num_list)
        return
    
    for idx in range(1, N+1):
        if visited[idx] == False:
            visited[idx] = True
            dfs_recur(num_list + [idx])
            visited[idx] = False
            
dfs_recur([])
for each in result_list:
    print(*each)
```

```
# 1717 집합의 표현

import sys
input = sys.stdin.readline
sys.setrecursionlimit(int(1e6))

# N+1의 집합 / M개의 연산
N, M = map(int, input().split())

# 합집합은 0 a b의 형태
# 두 원소가 한 집합안에 속하는지 확인 1 a b

parent = [0] * (N+1)
for idx in range(1, N+1):
    parent[idx] = idx

def find(x):
    if x != parent[x]:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    a = find(a)
    b = find(b)
    if a <= b:
        parent[b] = a
    else:
        parent[a] = b

for _ in range(M):
    criteria, a, b = map(int, input().split())
    if criteria == 0:
        union(a,b)
    else:
        a = find(a)
        b = find(b)
        if a == b:
            print("YES")
        else:
            print("NO")
```

```
# 1260 DFS와 BFS

# 정점, 간선, 시작 정점
N, M, start = map(int, input().split())

graph = {}
for idx in range(1, N+1):
    graph[idx] = []
for _ in range(M):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)
for key, value in graph.items():
    graph[key] = sorted(value)

# DFS
dfs_result = [start]
visited = [False] * (N+1)
visited[start] = True

def dfs(node):
    for each_node in graph[node]:
        if visited[each_node] == False:
            visited[each_node] = True
            dfs_result.append(each_node)
            dfs(each_node)

dfs(start)
print(*dfs_result)

# BFS
bfs_result = [start]
visited = [False] * (N+1)
visited[start] = True

def bfs(node):
    from collections import deque
    queue = deque()
    queue.append(node)
    visited[node] = True
    while queue:
        poped_node = queue.popleft()
        for next_node in graph[poped_node]:
            if visited[next_node] == False:
                visited[next_node] = True
                queue.append(next_node)
                bfs_result.append(next_node)
bfs(start)
print(*bfs_result)
```

```
# 15650 N과 M 2

N, M = map(int, input().split())

visited = [False] * (N+1)
result_list = []
def dfs_recur(num_list):
    if len(num_list) == M:
        result_list.append(num_list)
        return
    for idx in range(1, N+1):
        if visited[idx] == False:
            if len(num_list) == 0:
                visited[idx] = True
                dfs_recur(num_list + [idx])
                visited[idx] = False
            else:
                if idx >= num_list[-1]:
                    visited[idx] = True
                    dfs_recur(num_list + [idx])
                    visited[idx] = False

dfs_recur([])
for each in result_list:
    print(*each)
```

```
# 2998 8진수

# 2진수 -> 10진수 -> 8진수

input_num = input()
num_10 = int(input_num, 2)

def convert(num, criteria):
    str_num = ''
    while True:
        num, rest = divmod(num, criteria)
        str_num = str(rest) + str_num
        if num == 0:
            break
    return str_num

print(convert(num_10, 8))
```

```
# 2644 촌수계산

final_result = []
N = int(input())
start, end = map(int, input().split())
M = int(input())
graph = {}
for idx in range(1, N+1):
    graph[idx] = []
for _ in range(M):
    a, b  = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

"""
DFS - BackTracking
"""
visited = [False] * (N+1)
visited[start] = True
result = -1

def dfs(node, depth):
    global result
    if node == end:
        result = depth
    global count
    for next_node in graph[node]:
        if visited[next_node] == False:
            visited[next_node] = True
            dfs(next_node, depth + 1)
            # visited[next_node] = False
dfs(start, 0)
final_result.append(result)

"""
BFS
"""
visited = [False] * (N+1)
visited[start] = True
result = -1

from collections import deque
queue = deque()
queue.append((start, 0)) # node, depth
while queue:
    poped_node, poped_depth = queue.popleft()
    if poped_node == end:
        result = poped_depth
        break
    for next_node in graph[poped_node]:
        if visited[next_node] == False:
            visited[next_node] = True
            queue.append((next_node, poped_depth + 1))
final_result.append(result)

import random
print(random.choice(final_result))
```
