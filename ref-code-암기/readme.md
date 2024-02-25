
```
1. 10진수에서 N진수 변환함수

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

```
