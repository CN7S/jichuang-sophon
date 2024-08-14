#! /bin/bash

model_name=model_vae
test_input_path=../data/input_data.npz
test_result_path=../data/output_test.npz

model_deploy.py \
   --mlir ../output/${model_name}.mlir \
	--processor bm1684\
	--quantize F32\
	--test_input $test_input_path \
	--test_reference $test_result_path \
   --model ../output/${model_name}_fp32.bmodel \
   # --disable_layer_group

# model_name=Pixart_block
# test_input_path=../data/input_test.npz
# test_result_path=../data/output_test.npz

# model_deploy.py \
#    --mlir ../output/${model_name}.mlir \
# 	--processor bm1684\
# 	--quantize F32\
#    --model ../output/${model_name}_fp32.bmodel
#    # --disable_layer_group\
