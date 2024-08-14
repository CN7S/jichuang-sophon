#! /bin/bash

model_name=PixArt_block_0
data_dir=../data/dataset/cali/${model_name}
run_qtable.py ../output/model/$model_name.mlir \
    --dataset ${data_dir} \
    --input_num 1 \
    --calibration_table ../output/cali/$model_name.cali\
    --processor bm1684 \
    --loss_table ../output/cali/$model_name.loss\
    --base_quantize_table ../output/cali/$model_name.qtable\
    -o ../output/cali/$model_name.qtable