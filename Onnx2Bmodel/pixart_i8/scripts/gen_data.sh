#! /bin/bash

model_name=PixArt_block_0
data_dir=../data/dataset/cali/${model_name}
if [ ! -d $data_dir ];
    then
    mkdir $data_dir
    echo "create dir";
    else
    echo "dir exists";
fi
iter=0
while (( $iter < 10))
do
    gen_rand_input.py \
        --mlir ../output/model/$model_name.mlir \
        --ranges [[-1,1],[-1,1],[-1,1]]\
        --input_types f32,f32,f32\
        --output ${data_dir}/input_$iter.npz
    ((iter=$iter+1))
done