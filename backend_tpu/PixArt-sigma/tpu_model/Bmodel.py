import numpy as np
import sophon.sail as sail
import os
import gc
class BMODEL():
    def __init__(
            self,
            model_path=None,
    ):
        # net load
        # if model_path is not None:
        self.net = sail.Engine(model_path, 0, sail.IOMode.SYSI)
        self.net_handle = self.net.get_handle()
        self.net_graph_name = self.net.get_graph_names()[0]
        self.net_input_names = self.net.get_input_names(self.net_graph_name)
        self.net_output_names = self.net.get_output_names(self.net_graph_name)
        self.net_input_shape = []
        for index, input_name in enumerate(self.net_input_names):
            self.net_input_shape.append(self.net.get_input_shape(self.net_graph_name, input_name))
        self.net_outputs = []
        self.net_output_shape = []
        self.net_output_dtype = []
        self.net_output_tensors = {}
        for index, output_name in enumerate(self.net_output_names):
            self.net_output_shape.append(self.net.get_output_shape(self.net_graph_name, output_name))
            self.net_output_dtype.append(self.net.get_output_dtype(self.net_graph_name, output_name))
            self.net_outputs.append(sail.Tensor(self.net_handle, self.net_output_shape[index], self.net_output_dtype[index], True, True))
            self.net_output_tensors[output_name] = self.net_outputs[index] 
        print('Loading Finished.')
        # else:
        #     self.net = sail.Engine(0)

    # def load_model(self, model_path):        
    #     # net load
    #     self.net.load(model_path)
    #     input()
    #     self.net_handle = self.net.get_handle()
    #     self.net_graph_name = self.net.get_graph_names()[0]
    #     self.net_input_names = self.net.get_input_names(self.net_graph_name)
    #     self.net_output_names = self.net.get_output_names(self.net_graph_name)
    #     self.net_input_shape = []
    #     for index, input_name in enumerate(self.net_input_names):
    #         self.net_input_shape.append(self.net.get_input_shape(self.net_graph_name, input_name))
    #     self.net_outputs = []
    #     self.net_output_shape = []
    #     self.net_output_dtype = []
    #     self.net_output_tensors = {}
    #     for index, output_name in enumerate(self.net_output_names):
    #         self.net_output_shape.append(self.net.get_output_shape(self.net_graph_name, output_name))
    #         self.net_output_dtype.append(self.net.get_output_dtype(self.net_graph_name, output_name))
    #         self.net_outputs.append(sail.Tensor(self.net_handle, self.net_output_shape[index], self.net_output_dtype[index], True, True))
    #         self.net_output_tensors[output_name] = self.net_outputs[index] 
    #     print('Loading Finished.')


    def forward(self, **kwargs):
        ref_data = [kwargs[x] for x in kwargs]
        input_tensors = {}
        input_shapes = {}
        for index, input_name in enumerate(self.net_input_names):
            input_tensors[input_name] = sail.Tensor(self.net_handle, ref_data[index])
            input_shapes[input_name] = self.net_input_shape[index]
        
        self.net.process(self.net_graph_name, input_tensors, input_shapes, self.net_output_tensors)
        return [x.asnumpy(self.net_output_shape[index]) for index, x in enumerate(self.net_outputs)]