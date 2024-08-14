#! /bin/bash

model_name=Pixart_embedding_layer
test_input_path=../data/embedder_input.npz
test_result_path=../data/embedder_output.npz

model_deploy.py \
   --mlir ../output/${model_name}.mlir \
	--processor bm1684\
	--quantize F32\
   --model ../output/${model_name}_fp32.bmodel \
   --disable_layer_group

# model_name=Pixart_block
# test_input_path=../data/input_test.npz
# test_result_path=../data/output_test.npz

# model_deploy.py \
#    --mlir ../output/${model_name}.mlir \
# 	--processor bm1684\
# 	--quantize F32\
#    --model ../output/${model_name}_fp32.bmodel
#    # --disable_layer_group\
