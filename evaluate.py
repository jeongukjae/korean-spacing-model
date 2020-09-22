import json
from argparse import ArgumentParser

import tensorflow as tf

from train import (
    SpacingModel,
    string_to_example,
    sparse_categorical_accuracy_with_ignore,
    SparseCategoricalCrossentropyWithIgnore,
)

parser = ArgumentParser()
parser.add_argument("--char-file", type=str, required=True)
parser.add_argument("--model-file", type=str, required=True)
parser.add_argument("--training-config", type=str, required=True)
parser.add_argument("--test-file", type=str, required=True)
parser.add_argument("--add-prob", type=float, required=True)
parser.add_argument("--delete-prob", type=float, required=True)


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

    test_dataset = (
        tf.data.TextLineDataset(tf.constant([args.test_file]))
        .shuffle(10000)
        .map(
            string_to_example(vocab_table, delete_prob=args.delete_prob, add_prob=args.add_prob),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .batch(config["val_batch_size"])
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

    model.load_weights(args.model_file)
    model(tf.keras.Input([None], dtype=tf.int32))
    model.summary()
    model.evaluate(test_dataset)


if __name__ == "__main__":
    main()
