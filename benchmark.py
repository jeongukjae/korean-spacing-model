import json
import time
from argparse import ArgumentParser

import tensorflow as tf

from train import SpacingModel

parser = ArgumentParser()
parser.add_argument("--training-config", type=str, required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--sequence-length", type=int, required=True)

args = parser.parse_args()

with open(args.training_config) as f:
    config = json.load(f)

model = SpacingModel(
    config["vocab_size"],
    config["hidden_size"],
    conv_activation=config["conv_activation"],
    dense_activation=config["dense_activation"],
    conv_kernel_and_filter_sizes=config["conv_kernel_and_filter_sizes"],
    dropout_rate=config["dropout_rate"],
)
model(tf.keras.Input([None], dtype=tf.int32))


@tf.function
def run(batch):
    return model(batch)


print("Warmup stage (10 iteration)")
for _ in range(10):
    run(tf.random.uniform((args.batch_size, args.sequence_length), maxval=config["vocab_size"], dtype=tf.int32))

print("Benchmark model speed with random input (1000 iteration)")
s = time.time()
for _ in range(1000):
    run(tf.random.uniform((args.batch_size, args.sequence_length), maxval=config["vocab_size"], dtype=tf.int32))
elapsed = time.time() - s
print("Elapsed:", elapsed, "s")
print("Per batch:", elapsed / 1000, "s")
print("Per sentence:", elapsed / 1000 / args.batch_size, "s")
