#! /bin/bash
model_name=PixArt_block_0
quantize=MIX
if [ $quantize == "F32" ];
then
   model_deploy.py \
      --mlir ../output/model/$model_name.mlir \
      --chip bm1684\
      --quantize ${quantize}\
      --model ../output/model/${model_name}_${quantize}.bmodel
      # --calibration_table ../output/cali/$model_name.cali \
      # --quantize_table ../output/cali/$model_name.qtable \
else 
   if [ $quantize == "INT8" ];
   then
      model_deploy.py \
         --mlir ../output/model/$model_name.mlir \
         --chip bm1684\
         --quant_input\
         --quant_output\
         --quantize ${quantize}\
         --calibration_table ../output/cali/$model_name.cali \
         --model ../output/model/${model_name}_${quantize}.bmodel \
         # --quantize_table ../output/cali/$model_name.qtable
   else 
      if [ $quantize == "MIX" ];
      then
         model_deploy.py \
            --mlir ../output/model/$model_name.mlir \
            --chip bm1684\
            --quantize INT8\
            --quant_input\
            --quant_output\
            --calibration_table ../output/cali/$model_name.cali \
            --quantize_table ../output/cali/$model_name.qtable \
            --model ../output/model/${model_name}_${quantize}.bmodel 
      fi
   fi
fi

