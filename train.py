import json
from argparse import ArgumentParser
from typing import List, Tuple

import tensorflow as tf

parser = ArgumentParser()
parser.add_argument("--train-file", type=str, required=True)
parser.add_argument("--dev-file", type=str, required=True)
parser.add_argument("--training-config", type=str, required=True)
parser.add_argument("--char-file", type=str, required=True)


class SpacingModel(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_classes: int = 3,
        conv_activation: str = "relu",
        dense_activation: str = "relu",
        conv_kernel_and_filter_sizes: List[Tuple[int, int]] = [
            (2, 8),
            (3, 8),
            (4, 8),
            (5, 8),
        ],
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.embeddings = tf.keras.layers.Embedding(vocab_size, hidden_size)
        self.convs = [
            tf.keras.layers.Conv1D(
                filter_size,
                kernel_size,
                padding="same",
                activation=conv_activation,
            )
            for kernel_size, filter_size in conv_kernel_and_filter_sizes
        ]
        self.pools = [
            tf.keras.layers.MaxPooling1D(pool_size=filter_size, data_format="channels_first")
            for _, filter_size in conv_kernel_and_filter_sizes
        ]
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.output_dense1 = tf.keras.layers.Dense(hidden_size, activation=dense_activation)
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.output_dense2 = tf.keras.layers.Dense(num_classes)

    def call(self, input_tensor):
        """
        input_tensor: Tokenized Sequences, Shape: (Batch Size, Sequence Length)
        """

        # embeddings: (Batch Size, Sequence Length, Hidden Size)
        embeddings = self.embeddings(input_tensor)
        # features: (Batch Size, Sequence Length, sum(#filters))
        features = self.dropout1(
            tf.concat([pool(conv(embeddings)) for conv, pool in zip(self.convs, self.pools)], axis=-1)
        )
        # projected: (Batch Size, Sequence Length, Hidden Size)
        projected = self.dropout2(self.output_dense1(features))
        # (Batch Size, Sequence Length, 2)
        return self.output_dense2(projected)


def string_to_example(
    vocab_table: tf.lookup.StaticHashTable,
    encoding: str = "UTF-8",
    max_length: int = 256,
    delete_prob: float = 0.5,
    add_prob: float = 0.15,
):
    @tf.function
    def _inner(tensors: tf.Tensor):
        bytes_array = tf.strings.unicode_split(tf.strings.regex_replace(tensors, " +", " "), encoding)
        space_positions = bytes_array == " "
        sequence_length = tf.shape(space_positions)[0]

        while_condition = lambda i, *_: i < sequence_length

        def while_body(i, strings, labels):
            # 다음 char가 space가 아니고, 문장 끝이 아닐 때 add_prob의 확률로 space 추가
            # 이번 char가 space일 때
            is_next_char_space = tf.cond(i < sequence_length - 1, lambda: bytes_array[i + 1] == " ", lambda: False)

            state = tf.cond(
                is_next_char_space,
                lambda: tf.cond(tf.random.uniform([]) < delete_prob, lambda: 2, lambda: 0),
                lambda: tf.cond(bytes_array[i] != " " and tf.random.uniform([]) < add_prob, lambda: 1, lambda: 0),
            )
            # 0: 그대로 진행
            # 1: 다음 인덱스에 space 추가
            # 2: 다음 space 삭제
            strings = tf.cond(
                state != 1,
                lambda: tf.concat([strings, [bytes_array[i]]], axis=0),
                lambda: tf.concat([strings, [bytes_array[i], " "]], axis=0),
            )
            # label 0: 변화 x
            # label 1: 다음 인덱스에 space 추가
            # label 2: 현재 space 삭제
            labels = tf.cond(
                state == 0,
                lambda: tf.concat([labels, [0]], axis=0),
                lambda: tf.cond(
                    state == 1,
                    lambda: tf.concat([labels, [0, 2]], axis=0),
                    lambda: tf.concat([labels, [1]], axis=0),
                ),
            )
            i += tf.cond(state == 2, lambda: 2, lambda: 1)

            return (i, strings, labels)

        i, strings, labels = tf.while_loop(
            while_condition,
            while_body,
            (
                tf.constant(0),
                tf.constant([], dtype=tf.string),
                tf.constant([], dtype=tf.int32),
            ),
            shape_invariants=(tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([None])),
        )

        strings = vocab_table.lookup(tf.concat([["<s>"], strings, ["</s>"]], axis=0))
        labels = tf.concat([[0], labels, [0]], axis=0)

        strings = tf.cond(tf.shape(strings)[0] > max_length, lambda: strings[:max_length], lambda: strings)
        labels = tf.cond(tf.shape(labels)[0] > max_length, lambda: labels[:max_length], lambda: labels)

        length_to_pad = max_length - tf.shape(strings)[0]
        strings = tf.pad(strings, [[0, length_to_pad]])
        labels = tf.pad(labels, [[0, length_to_pad]], constant_values=-1)

        return (strings, labels)

    return _inner


def sparse_categorical_crossentropy_with_ignore(y_true, y_pred, from_logits=False, axis=-1, ignore_id=-1):
    positions = tf.where(y_true != ignore_id)

    y_true = tf.gather_nd(y_true, positions)
    y_pred = tf.gather_nd(y_pred, positions)

    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits, axis=axis)


def sparse_categorical_accuracy_with_ignore(y_true, y_pred, ignore_id=-1):
    positions = tf.where(y_true != ignore_id)

    y_true = tf.gather_nd(y_true, positions)
    y_pred = tf.gather_nd(y_pred, positions)

    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


class SparseCategoricalCrossentropyWithIgnore(tf.python.keras.losses.LossFunctionWrapper):
    def __init__(
        self,
        from_logits=False,
        reduction=tf.keras.losses.Reduction.AUTO,
        ignore_id=-1,
        name="sparse_categorical_crossentropy_with_ignore",
    ):
        super(SparseCategoricalCrossentropyWithIgnore, self).__init__(
            sparse_categorical_crossentropy_with_ignore,
            name=name,
            reduction=reduction,
            ignore_id=ignore_id,
            from_logits=from_logits,
        )


def main():
    args = parser.parse_args()
    with open(args.training_config) as f:
        config = json.load(f)

    with open(args.char_file) as f:
        content = f.read()
        keys = ["<pad>", "<s>", "</s>", "<unk>"] + list(content)
        values = list(range(len(keys)))

        vocab_initializer = tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.string, value_dtype=tf.int32)
        vocab_table = tf.lookup.StaticHashTable(vocab_initializer, default_value=3)

    train_dataset = (
        tf.data.TextLineDataset(tf.constant([args.train_file]))
        .shuffle(10000)
        .map(
            string_to_example(vocab_table),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .batch(config["train_batch_size"])
    )
    dev_dataset = (
        tf.data.TextLineDataset(tf.constant([args.dev_file]))
        .shuffle(10000)
        .map(
            string_to_example(vocab_table),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .batch(config["val_batch_size"])
        .take(4)
    )

    model = SpacingModel(
        config["vocab_size"],
        config["hidden_size"],
        conv_activation=config["conv_activation"],
        dense_activation=config["dense_activation"],
        conv_kernel_and_filter_sizes=config["conv_kernel_and_filter_sizes"],
        dropout_rate=config["dropout_rate"],
    )
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=SparseCategoricalCrossentropyWithIgnore(from_logits=True, ignore_id=-1),
        metrics=[sparse_categorical_accuracy_with_ignore],
    )
    model.fit(
        train_dataset,
        epochs=config["epochs"],
        validation_data=dev_dataset,
        steps_per_epoch=400,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(filepath="./models/checkpoint-{epoch}.ckpt"),
            tf.keras.callbacks.TensorBoard(log_dir="./logs"),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2, verbose=1),
        ],
    )


if __name__ == "__main__":
    main()
