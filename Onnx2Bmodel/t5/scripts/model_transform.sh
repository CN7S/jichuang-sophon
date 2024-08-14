#! /bin/bash

layer_id=0
while (( $layer_id < 24))
do
	model_name=T5-block-$layer_id
	model_def_path=/workspace/Onnx/T5/fp32/${model_name}.onnx
	test_input_path=../data/input_test.npz
	test_result_path=../data/output_test.npz
	input_shape='[[1,300,4096],[1,64,300,300]]'
	if [ $layer_id -eq 0 ];
	then
		input_shape='[[1,300,4096],[1,300]]'
	fi

	model_transform.py \
		--model_name $model_name\
		--model_def $model_def_path\
		--input_shapes $input_shape \
		--input_types float32,float32 \
		--channel_format none\
		--mlir ../output/$model_name.mlir
	
	echo -n "$layer_id done."
	((layer_id=$layer_id+1))
done


model_name=T5-final-layer
model_def_path=/workspace/Onnx/T5/fp32/${model_name}.onnx
test_input_path=../data/input_test.npz
test_result_path=../data/output_test.npz
input_shape='[[1,300,4096]]'
model_transform.py \
	--model_name $model_name\
	--model_def $model_def_path\
	--input_shapes $input_shape \
	--input_types float32 \
	--channel_format none\
	--mlir ../output/$model_name.mlir



# model_name=T5-block-2
# model_def_path=/workspace/Onnx/T5/fp32/${model_name}.onnx
# test_input_path=../data/input_test.npz
# test_result_path=../data/output_test.npz

# model_transform.py \
# 	--model_name $model_name\
# 	--model_def $model_def_path\
# 	--input_shapes [[1,300,4096],[1,64,300,300]] \
# 	--input_types float32,float32 \
# 	--channel_format none\
# 	--mlir ../output/$model_name.mlir
# 	# --test_input $test_input_path\
# 	# --test_result $test_result_path\


# layer_id=0
# while (( $layer_id < 24))
# do
# 	echo -n "$layer_id"
# 	((layer_id=$layer_id+1))
# done
