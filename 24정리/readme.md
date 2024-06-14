
## MST - 크루스칼
---

![스크린샷 2024-06-14 오후 11 00 06](https://github.com/briiidgehong/A3_algorithm_essential/assets/73451727/9284e441-b884-465e-8649-5304479af685)

```
"""
최소스패닝 트리?
사이클이 없고, 모든 노드가 연결된 그래프
MST는 크루스칼로 푼다.
크루스칼:
parent/union/find 를 사용해서, 소팅후 사이클 판별해가며 
간선이 적은 순서대로 연결한다.

"""
V, E = map(int, input().split())
e_list = []
for _ in range(E):
	start, end, cost = map(int, input().split())
	e_list.append((cost, start, end))
e_list.sort()

parent = [0] * (V+1)
for idx in range(1, V+1):
	parent[idx] = idx

def union(a,b):
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

total_cost = 0
for each_cost, each_start, each_end in e_list:
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
