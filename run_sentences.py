import json
from argparse import ArgumentParser

import tensorflow as tf

from train import SpacingModel

parser = ArgumentParser()
parser.add_argument("--char-file", type=str, required=True)
parser.add_argument("--model-file", type=str, required=True)
parser.add_argument("--training-config", type=str, required=True)

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

        inverse_vocab_initializer = tf.lookup.KeyValueTensorInitializer(
            values, keys, value_dtype=tf.string, key_dtype=tf.int32
        )
        inverse_vocab_table = tf.lookup.StaticHashTable(inverse_vocab_initializer, default_value="<unk>")

    model = SpacingModel(
        config["vocab_size"],
        config["hidden_size"],
        conv_activation=config["conv_activation"],
        dense_activation=config["dense_activation"],
        conv_kernel_and_filter_sizes=config["conv_kernel_and_filter_sizes"],
        dropout_rate=config["dropout_rate"],
    )

    model.load_weights(args.model_file)
    model(tf.keras.Input([None], dtype=tf.int32))
    print(model.summary())

    inference = get_inference_fn(model, vocab_table)

    while True:
        input_str = input("Str: ")
        input_str = tf.constant(input_str)
        result = inference(input_str).numpy()
        print(b"".join(result).decode("utf8"))


def get_inference_fn(model, vocab_table):
    @tf.function
    def inference(tensors):
        byte_array = tf.concat(
            [["<s>"], tf.strings.unicode_split(tf.strings.regex_replace(tensors, " +", " "), "UTF-8"), ["</s>"]], axis=0
        )
        strings = vocab_table.lookup(byte_array)[tf.newaxis, :]

        model_output = tf.argmax(model(strings), axis=-1)[0]
        return convert_output_to_string(byte_array, model_output)

    return inference

def convert_output_to_string(byte_array, model_output):
    sequence_length = tf.size(model_output)
    while_condition = lambda i, *_: i < sequence_length

    def while_body(i, o):
        o = tf.cond(
            model_output[i] == 1,
            lambda: tf.concat([o, [byte_array[i], " "]], axis=0),
            lambda: tf.cond(
                (model_output[i] == 2) and (byte_array[i] == " "),
                lambda: o,
                lambda: tf.concat([o, [byte_array[i]]], axis=0),
            ),
        )
        return i + 1, o

    _, strings_result = tf.while_loop(
        while_condition,
        while_body,
        (tf.constant(0), tf.constant([], dtype=tf.string)),
        shape_invariants=(tf.TensorShape([]), tf.TensorShape([None])),
    )
    return strings_result

if __name__ == "__main__":
    main()
