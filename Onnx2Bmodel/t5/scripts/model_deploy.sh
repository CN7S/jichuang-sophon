#! /bin/bash
layer_id=0
while (( $layer_id < 24))
do
	model_name=T5-block-$layer_id
   model_deploy.py \
      --mlir ../output/${model_name}.mlir \
      --chip bm1684\
      --quantize F32\
      --skip_validation\
      --model ../output/${model_name}_fp32.bmodel
	
	echo -n "$layer_id done."
	((layer_id=$layer_id+1))
done


model_name=T5-final-layer
model_deploy.py \
   --mlir ../output/${model_name}.mlir \
	--chip bm1684\
	--quantize F32\
    --skip_validation\
   --model ../output/${model_name}_fp32.bmodel