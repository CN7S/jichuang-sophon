#! /bin/bash 

model_name=PixArt_block_0
model_def_path=../models/fp32/Pixart_Block_0_MS.onnx

model_transform.py \
	--model_name $model_name\
	--model_def $model_def_path\
	--input_shapes [[2,1024,1152],[2,6912],[1,100,1152]] \
	--input_types float32,float32,float32 \
	--channel_format none\
	--mlir ../output/model/$model_name.mlir
	# --input_shapes [[2,1024,1152],[2,6912],[1,100,1152]] \
	# --input_types float32,float32,float32 \










