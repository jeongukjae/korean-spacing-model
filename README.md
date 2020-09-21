# Korean Spacing Model

## 학습용 파일들

이 폴더는 학습용 파일들이 모여있는 파일입니다. 워낙 간단한 모델이고, 대부분 TF의 기본기능만 사용하였기 때문에 패키지 혹은 테스트 코드는 존재하지 않습니다.

### 모델

사용한 모델은 [training/train.py#L14](https://github.com/jeongukjae/korean-spacing-model/blob/master/training/train.py#L14)를 참고하시기 바랍니다. 여러개의 Conv1D + MaxPool1D 결과물을 Concat 한 뒤 FFN을 거칩니다.

### `chars-4997`

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

### `config.json`
