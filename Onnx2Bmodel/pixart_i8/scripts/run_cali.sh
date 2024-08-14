#! /bin/bash

model_name=PixArt_block_0
data_dir=../data/dataset/cali/${model_name}
echo ${data_dir}
run_calibration.py ../output/model/$model_name.mlir \
    --dataset ${data_dir} \
    --input_num 1 \
    -o ../output/cali/$model_name.cali \



# run_qtable.py ../output/model/$model_name.mlir \
#     --dataset ../data/ \
#     --input_num 50 \
#     --calibration_table ../output/cali/$model_name.cali\
#     --processor bm1684 \
#     --loss_table ../output/cali/$model_name.loss\
#     -o ../output/cali/$model_name.qtable


    