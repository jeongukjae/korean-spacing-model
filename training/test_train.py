import tensorflow as tf
import pytest

from train import string_to_example

@pytest.fixture
def vocab_table():
    with open("./resources/chars-4997") as f:
        content = f.read()
        keys = ["<pad>", "<s>", "</s>", "<unk>"] + list(content)
        values = list(range(len(keys)))

        vocab_initializer = tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.string, value_dtype=tf.int32)
        vocab_table = tf.lookup.StaticHashTable(vocab_initializer, default_value=3)

    return vocab_table

def test_string_to_example(vocab_table):
    string_to_example(vocab_table)(tf.constant("안녕 하세요, 이 프로 젝트는 프로젝 트입 니다."))
