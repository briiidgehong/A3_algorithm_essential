
## MST - 크루스칼 - 백준 1197
---

![스크린샷 2024-06-14 오후 11 00 06](https://github.com/briiidgehong/A3_algorithm_essential/assets/73451727/9284e441-b884-465e-8649-5304479af685)

```
"""
MST = 최소 스패닝 트리
사이클이 없고, 모든 노드가 최소의 가중치로 연결된 그래프
MST는 크루스칼로 푼다.

크루스칼:
전체 간선 중 작은것부터 연결, 사이클 판별해가며
parent/union-find 를 사용

3 3       : 정점의 개수, 간선의 개수
1 2 1     : A, B, A와 B가 이어진 가중치
2 3 2
1 3 3
"""

# 정점, 간선
V, E = map(int, input().split())
edges = []
for _ in range(E):
    start, end, cost = map(int, input().split())
    edges.append((start, end, cost))

parent = [0] * (V+1)
for idx in range(1, V+1):
    parent[idx] = idx

def union(a, b):
    a = find(a)
    b = find(b)
    if a <= b:
        parent[b] = a
    else:
        parent[a] = b
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

edges.sort()
total_cost = 0
for each_cost, each_start, each_end in edges:
    start = find(each_start)
    end = find(each_end)
    if start == end:
        continue
    else:
        union(each_start, each_end)
        total_cost += each_cost
print(total_cost)

```
---

## 다익스트라, 플로이드 와샬 - 백준 1238
```
가중치 그래프의 최단거리를 구할때는 -> 다익스트라, 플로이드 와샬
다익스트라 -> 한 정점에서 모든 정점까지의 최단거리, O(ElogV)
    parent table / union-find
플로이드 와샬 -> 모든 정점에서 모든 정점까지의 최단거리, O(V^3)
    점화식:
        for k in range(1, N+1):
            for a in range(1, k+1):
                for b in range(a+1, k+1):
                    Dab = min(Dak + Dab, Dab)
```

### 1. 다익스트라
<img width="933" alt="스크린샷 2024-06-15 오후 8 45 26" src="https://github.com/briiidgehong/A3_algorithm_essential/assets/73451727/d36797b7-429f-4745-b016-db6cc80cfeb8">

```
"""
다익스트라
"""

# N 마을갯수, M 간선갯수, X 파티가 열리는 마을
N, M, X = map(int, input().split())

graph_dij = {}
for idx in range(N+1): # 0 ~ N
    graph_dij[idx] = []

for _ in range(M):
    start, end, cost = map(int, input().split())
    graph_dij[start].append((cost, end))

def dij(start):
    import heapq
    queue = []
    min_table = [int(1e9)] * (N+1)
    min_table[start] = 0
    heapq.heappush(queue, (0, start))
    while queue:
        poped_cost, poped_node = heapq.heappop(queue)
        for next_cost, next_node in graph_dij[poped_node]:
            if poped_cost + next_cost < min_table[next_node]:
                min_table[next_node] = poped_cost + next_cost
                heapq.heappush(queue, (poped_cost + next_cost, next_node))
    return min_table
print(dij(X))


"""
플로이드 와샬
"""
N, M, X = map(int, input().split())


graph_fw = {}
for idx in range(N+1): # 0 ~ N
    graph_fw[idx] = [int(1e9)] * (N+1) # 0 ~ N

for _ in range(M):
    start, end, cost = map(int, input().split())
    graph_fw[start][end] = cost

# 자기 자신의 cost는 0으로 초기화
for idx_x in range(1, N+1):
    for idx_y in range(1, N+1):
        if idx_x == idx_y:
            graph_fw[idx_x][idx_y] = 0

# 플로이드 와샬 알고리즘 수행
for k in range(1, N+1):
    for a in range(1, N+1):
        for b in range(1, N+1):
            # Dab = min(Dak + Dkb, Dab)
            graph_fw[a][b] = min(graph_fw[a][b], graph_fw[a][k] + graph_fw[k][b])

for key, value in graph_fw.items():
    print(key, value)

```
---

## BFS, DFS - 1260, 2644 + TODO: 백트래킹
```
"""
BFS , DFS

4 5 1
1 2
1 3
1 4
2 4
3 4
"""

# N: 정점의 개수 / M: 간선의 개수 / V: 탐색 시작 노드
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

dfs_result = [start]
visited = [False] * (N+1)
visited[start] = True

# DFS
def dfs(node):
    for each_node in graph[node]:
        if visited[each_node] == False:
            visited[each_node] = True
            dfs_result.append(each_node)
            dfs(each_node)
dfs(start)
print(*dfs_result)

bfs_result = [start]
visited = [False] * (N+1)
visited[start] = True

def bfs(node):
    from collections import deque
    queue = deque()
    queue.append(node)
    visited[node] = True
    while queue:
        pooped_node = queue.popleft()
        for next_node in graph[pooped_node]:
            if visited[next_node] == False:
                visited[next_node] = True
                queue.append(next_node)
                bfs_result.append(next_node)

bfs(start)
print(*bfs_result)
```

```
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
DFS 
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







---
