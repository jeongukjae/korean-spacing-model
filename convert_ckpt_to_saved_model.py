import json
import tensorflow as tf
from train import SpacingModel
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--training-config", type=str, required=True)
parser.add_argument("--input-path", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)

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
model.load_weights(args.input_path)


@tf.function()
def serve(input_tensor):
    return model(input_tensor)


tf.saved_model.save(
    model,
    args.output_path,
    serve.get_concrete_function(tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="input_tensor")),
)
