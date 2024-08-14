import numpy as np
import sophon.sail as sail
import os


#############################################################################
#                                 VAE Model                                #
#################################################################################
class VAE_DECODE():
    def __init__(
            self,
            model_path
    ):
        model_path = os.path.join(model_path, 'model_vae_fp32.bmodel')
        # vae model load
        self.vae = sail.Engine(model_path, 0, sail.IOMode.SYSI)
        self.vae_handle = self.vae.get_handle()
        self.vae_graph_name = self.vae.get_graph_names()[0]
        self.vae_input_names = self.vae.get_input_names(self.vae_graph_name)
        self.vae_output_names = self.vae.get_output_names(self.vae_graph_name)
        self.vae_input_shape = []
        for index, input_name in enumerate(self.vae_input_names):
            self.vae_input_shape.append(self.vae.get_input_shape(self.vae_graph_name, input_name))
        self.vae_outputs = []
        self.vae_output_shape = []
        self.vae_output_dtype = []
        self.vae_output_tensors = {}
        for index, output_name in enumerate(self.vae_output_names):
            self.vae_output_shape.append(self.vae.get_output_shape(self.vae_graph_name, output_name))
            self.vae_output_dtype.append(self.vae.get_output_dtype(self.vae_graph_name, output_name))
            self.vae_outputs.append(sail.Tensor(self.vae_handle, self.vae_output_shape[index], self.vae_output_dtype[index], True, True))
            self.vae_output_tensors[output_name] = self.vae_outputs[index] 
        print('Loading Finished.')
        
    def forward(self, samples):
        ref_data = [samples]
        input_tensors = {}
        input_shapes = {}
        for index, input_name in enumerate(self.vae_input_names):
            input_tensors[input_name] = sail.Tensor(self.vae_handle, ref_data[index])
            input_shapes[input_name] = self.vae_input_shape[index]
        
        self.vae.process(self.vae_graph_name, input_tensors, input_shapes, self.vae_output_tensors)
        return [x.asnumpy(self.vae_output_shape[index]) for index, x in enumerate(self.vae_outputs)]