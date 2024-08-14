import numpy as np
import sophon.sail as sail
import os

#############################################################################
#                                 text_encoer Model                                #
#################################################################################
class T5_Final_Norm():
    def __init__(
            self,
            model_path,
    ):
        # embed_path = os.path.join(model_path, 'embedding.npy')
        model_name = f'T5-final-layer_fp32.bmodel'
        model_path = os.path.join(model_path, model_name)
        # embedding load
        # self.embedding = 2 #np.load(embed_path)

        # text_encoer model load
        self.text_encoer = sail.Engine(model_path, 0, sail.IOMode.SYSI)
        self.text_encoer_handle = self.text_encoer.get_handle()
        self.text_encoer_graph_name = self.text_encoer.get_graph_names()[0]
        self.text_encoer_input_names = self.text_encoer.get_input_names(self.text_encoer_graph_name)
        self.text_encoer_output_names = self.text_encoer.get_output_names(self.text_encoer_graph_name)
        self.text_encoer_input_shape = []
        for index, input_name in enumerate(self.text_encoer_input_names):
            self.text_encoer_input_shape.append(self.text_encoer.get_input_shape(self.text_encoer_graph_name, input_name))
        self.text_encoer_outputs = []
        self.text_encoer_output_shape = []
        self.text_encoer_output_dtype = []
        self.text_encoer_output_tensors = {}
        for index, output_name in enumerate(self.text_encoer_output_names):
            self.text_encoer_output_shape.append(self.text_encoer.get_output_shape(self.text_encoer_graph_name, output_name))
            self.text_encoer_output_dtype.append(self.text_encoer.get_output_dtype(self.text_encoer_graph_name, output_name))
            self.text_encoer_outputs.append(sail.Tensor(self.text_encoer_handle, self.text_encoer_output_shape[index], self.text_encoer_output_dtype[index], True, True))
            self.text_encoer_output_tensors[output_name] = self.text_encoer_outputs[index] 
        print('Loading Finished.')

    def forward(self, **kwargs):
        ref_data = [kwargs[x] for x in kwargs]
        input_tensors = {}
        input_shapes = {}
        for index, input_name in enumerate(self.text_encoer_input_names):
            input_tensors[input_name] = sail.Tensor(self.text_encoer_handle, ref_data[index])
            input_shapes[input_name] = self.text_encoer_input_shape[index]
        
        self.text_encoer.process(self.text_encoer_graph_name, input_tensors, input_shapes, self.text_encoer_output_tensors)
        return [x.asnumpy(self.text_encoer_output_shape[index]) for index, x in enumerate(self.text_encoer_outputs)]

class T5_ENCODE_Block():
    def __init__(
            self,
            model_path,
            layer_id # 0-24
    ):
        # embed_path = os.path.join(model_path, 'embedding.npy')
        model_name = f'T5-block-{layer_id}_fp32.bmodel'
        model_path = os.path.join(model_path, model_name)
        # embedding load
        # self.embedding = 2 #np.load(embed_path)

        # text_encoer model load
        self.text_encoer = sail.Engine(model_path, 0, sail.IOMode.SYSI)
        self.text_encoer_handle = self.text_encoer.get_handle()
        self.text_encoer_graph_name = self.text_encoer.get_graph_names()[0]
        self.text_encoer_input_names = self.text_encoer.get_input_names(self.text_encoer_graph_name)
        self.text_encoer_output_names = self.text_encoer.get_output_names(self.text_encoer_graph_name)
        self.text_encoer_input_shape = []
        for index, input_name in enumerate(self.text_encoer_input_names):
            self.text_encoer_input_shape.append(self.text_encoer.get_input_shape(self.text_encoer_graph_name, input_name))
        self.text_encoer_outputs = []
        self.text_encoer_output_shape = []
        self.text_encoer_output_dtype = []
        self.text_encoer_output_tensors = {}
        for index, output_name in enumerate(self.text_encoer_output_names):
            self.text_encoer_output_shape.append(self.text_encoer.get_output_shape(self.text_encoer_graph_name, output_name))
            self.text_encoer_output_dtype.append(self.text_encoer.get_output_dtype(self.text_encoer_graph_name, output_name))
            self.text_encoer_outputs.append(sail.Tensor(self.text_encoer_handle, self.text_encoer_output_shape[index], self.text_encoer_output_dtype[index], True, True))
            self.text_encoer_output_tensors[output_name] = self.text_encoer_outputs[index] 
        print('Loading Finished.')

    def forward(self, **kwargs):
        ref_data = [kwargs[x] for x in kwargs]
        input_tensors = {}
        input_shapes = {}
        for index, input_name in enumerate(self.text_encoer_input_names):
            input_tensors[input_name] = sail.Tensor(self.text_encoer_handle, ref_data[index])
            input_shapes[input_name] = self.text_encoer_input_shape[index]
        
        self.text_encoer.process(self.text_encoer_graph_name, input_tensors, input_shapes, self.text_encoer_output_tensors)
        return [x.asnumpy(self.text_encoer_output_shape[index]) for index, x in enumerate(self.text_encoer_outputs)]

class T5_ENCODE():
    def __init__(
            self,
            model_path,
            device_map=None,
    ):
        self.model_path = model_path
        self.num_layers = 24
        self.device_map = device_map
        self.T5_Block_on_device = {}
        if device_map is not None:
            for layer in device_map:
                self.T5_Block_on_device[f'layer{layer}'] = T5_ENCODE_Block(model_path, layer)
        
    
    def forward(self, inputs_embeds, attention_mask):
        sail.set_print_flag(True)
        import time
        start_time = time.time()
        layer_time = []
        hidden_state = None
        position_bias = None
        # first block to convert position_bias
        if self.device_map is not None and 0 in self.device_map:
            [hidden_state, position_bias] = self.T5_Block_on_device[f'layer{0}'].forward(inputs_embeds, attention_mask)
        else:
            tempT5Block = T5_ENCODE_Block(self.model_path, 0)
            [hidden_state, position_bias] = tempT5Block.forward(a=inputs_embeds, b=attention_mask)
            del tempT5Block


        for layer in range(1, self.num_layers):
            layer_start = time.time()
            if self.device_map is not None and layer in self.device_map:
                [hidden_state] = self.T5_Block_on_device[f'layer{layer}'].forward(hidden_state, position_bias)
            else:
                tempT5Block = T5_ENCODE_Block(self.model_path, layer)
                [hidden_state] = tempT5Block.forward(a=hidden_state, b=position_bias)
                del tempT5Block
            layer_end = time.time()
            layer_duration = layer_end - layer_start
            layer_time.append(layer_duration)
        
        # final norm layer
        final_Norm = T5_Final_Norm(self.model_path)
        [hidden_state] = final_Norm.forward(a=hidden_state)

        end_time = time.time()
        duration = end_time - start_time
        print(f'T5 Encoder Total Time: {duration}s')
        for i, time in enumerate(layer_time):
            print(f'T5 layer {i+1} execute time : {time}s')
        return hidden_state

class T5_ENCODE_Slice():
    def __init__(
            self,
            model_path,
            layer_range=[0,23], # 0-23
            do_final_norm=True,
            on_device=False,
    ):
        print(f'T5 load config = {{layer_range}}:{layer_range}, {{do_final_norm}}: {do_final_norm}')
        self.start_layer = layer_range[0]
        self.end_layer = layer_range[1]
        self.model_path = model_path
        self.num_layers = 24
        self.do_final_norm = do_final_norm
        self.on_device = on_device
        self.T5_Block_on_device = {}

        if self.start_layer == 0:
            self.inputs_embedding = np.load(os.path.join(model_path, 'input_embeddings_weight.npy'))
        
        if self.on_device:
            for layer in range(self.start_layer, self.end_layer+1):
                self.T5_Block_on_device[f'layer{layer}'] = T5_ENCODE_Block(model_path, layer)
            if do_final_norm:
                self.final_norm = T5_Final_Norm(self.model_path)
        
    
    def forward(self, input_arg1, input_arg2 = None):
        import time
        start_time = time.time()
        layer_time = []
        hidden_state = None
        position_bias = None

        if self.start_layer == 0:
            # inputs_embeds = input_arg1
            inputs_embeds = self.inputs_embedding[input_arg1]
            attention_mask = input_arg2
        else:
            hidden_state = input_arg1
            position_bias = input_arg2
        
        # if self.end_layer == 0:
        #     raise 'Error, Slice need do more than two layers'
        
        # layer 0 - 23
        for layer in range(self.start_layer, self.end_layer+1): 
            if(layer == 0):
                # first block to convert position_bias
                if self.on_device:
                    [hidden_state, position_bias] = self.T5_Block_on_device[f'layer{layer}'].forward(a=inputs_embeds, b=attention_mask)
                else:
                    tempT5Block = T5_ENCODE_Block(self.model_path, layer)
                    [hidden_state, position_bias] = tempT5Block.forward(a=inputs_embeds, b=attention_mask)
                    del tempT5Block
            elif(layer < self.num_layers):
                layer_start = time.time()
                if self.on_device:
                    [hidden_state] = self.T5_Block_on_device[f'layer{layer}'].forward(a=hidden_state, b=position_bias)
                else:
                    tempT5Block = T5_ENCODE_Block(self.model_path, layer)
                    [hidden_state] = tempT5Block.forward(a=hidden_state, b=position_bias)
                    del tempT5Block
                layer_end = time.time()
                layer_duration = layer_end - layer_start
                layer_time.append(layer_duration)
        
        # final norm layer
        if self.do_final_norm:
            if self.on_device:
                [hidden_state] = self.final_norm.forward(a=hidden_state)
            else:
                final_norm = T5_Final_Norm(self.model_path)
                [hidden_state] = final_norm.forward(a=hidden_state)
                del final_norm

        end_time = time.time()
        duration = end_time - start_time
        print(f'T5 Encoder Total Time: {duration}s')
        for i, time in enumerate(layer_time):
            print(f'T5 layer {i+1} execute time : {time}s')

        if self.do_final_norm:
            return hidden_state
        else:
            return hidden_state, position_bias