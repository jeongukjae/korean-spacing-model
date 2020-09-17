```
>>> from collections import Counter
>>> c = Counter()
>>> f = open("namuwikitext_20200302.train.processed")
>>> for line in f:
>>>     c.update(line)
>>> sum([v for _, v in c.most_common(4996)]) / sum([v for _, v in c.most_common()])
0.9996295887538321
```

5000개의 vocab만 가져도 char coverage 0.9996가 넘는다. (3개는 unk + bos + eos + pad)
