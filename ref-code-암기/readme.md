
```
1. 10진수에서 N진수 변환함수
2. BFS 예시코드 작성해보시오.
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

```
