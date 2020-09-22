#!/bin/bash

tensorflowjs_converter \
    --input_format=tf_saved_model \
    --saved_model_tags=serve \
    ./spacing-model/1/ \
    ./docs/spacing-model-web
