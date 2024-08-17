CICC2221 算能杯 核心代码
共有5类程序代码
backend_tpu: 
	tpu后端代码，主要实现tpu模型部署及运算和TPU端的server程序编写源码。
frontend_webui: 
	前端ui界面源码，webui运行app程序位于app目录下的app.py
models_src:
	模型结构源码，包括pixart的GPU加速源码和TPU加速结构源码。
Onnx2Bmodel:
	模型从Onnx结构转换并部署为Bmodel的脚本源码，该脚本运行在sophon的docker环境下，使用tpu-mlir 1.7.
transformers:
	来自huggingface提供的python库 transformers， 修改了该路径下t5模型的model源码，使其结构可以被TPU加速与部署。
	