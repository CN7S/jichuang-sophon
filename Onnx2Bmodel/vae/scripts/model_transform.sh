#! /bin/bash

model_name=model_vae
model_def_path=/workspace/PixArt-sigma_custom/Onnx/VAE/fp32/VAE.onnx
test_input_path=../data/input_data.npz
test_result_path=../data/output_test.npz

model_transform.py \
	--model_name $model_name\
	--model_def $model_def_path\
	--input_shapes [[1,4,64,64]] \
	--input_types float32 \
	# --test_input $test_input_path\
	# --test_result $test_result_path\
	--channel_format none\
	--mlir ../output/$model_name.mlir

# model_name=Pixart_block
# model_def_path=/workspace/PixArt-sigma_custom/Onnx/Pixart/fp32/Pixart_1024_MS.onnx
# test_input_path=../data/input_test.npz
# test_result_path=../data/output_test.npz

# model_transform.py \
# 	--model_name $model_name\
# 	--model_def $model_def_path\
# 	--input_shapes [[2,1024,1152],[2,1152],[1,100,1152],[1,16,2048,100]] \
# 	--input_types float32,float32,float32,float32 \
# 	--channel_format none\
# 	--mlir ../output/$model_name.mlir
	# --test_input $test_input_path\
	# --test_result $test_result_path\
	# --input_shapes [[2,4,128,128],[2],[2,1,300,4096],[1,300]] \
	# [2, 4096, 1152]) torch.Size([1, 18, 1152]) torch.Size([2, 6912]) torch.Size([1, 16, 8192, 18]
