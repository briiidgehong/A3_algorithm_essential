
```
1. 10진수에서 N진수 변환함수 예시코드를 작성하시오.
2. BFS 예시코드를 작성하시오.
3. 백트래킹 예시코드를 작성하시오.
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
```
