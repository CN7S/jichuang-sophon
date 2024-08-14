import numpy as np
import sophon.sail as sail


#############################################################################
#                                 Core PixArt Model                                #
#################################################################################
class PixArtMS():
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            embedding_path,
            block_path,
            attn_bias
    ):
        
        # embedding model load
        self.embedding = sail.Engine(embedding_path, 0, sail.IOMode.SYSI)
        self.embedding_handle = self.embedding.get_handle()
        self.embedding_graph_name = self.embedding.get_graph_names()[0]
        self.embedding_input_names = self.get_input_names(self.embedding_graph_name)
        self.embedding_output_names = self.get_output_names(self.embedding_graph_name)
        self.embedding_input_shape = []
        for index, input_name in enumerate(self.embedding_input_names):
            self.embedding_input_shape[index] = self.embedding.get_input_shape(self.embedding_graph_name, input_name)
        self.embedding_outputs = []
        self.embedding_output_shape = []
        self.embedding_output_dtype = []
        self.embedding_output_tensors = {}
        for index, output_name in enumerate(self.embedding_output_names):
            self.embedding_output_shape[index] = self.embedding.get_output_shape(self.embedding_graph_name, output_name)
            self.embedding_output_dtype[index] = self.embedding.get_output_dtype(self.embedding_graph_name, output_name)
            self.embedding_outputs[index] = sail.Tensor(self.embedding_handle, self.embedding_output_shape[index], self.embedding_output_dtype[index], True, True)
            self.embedding_output_tensors[output_name] = self.embedding_outputs[index] 
        
        # block model load
        self.block = sail.Engine(block_path, 0, sail.IOMode.SYSI)
        self.block_handle = self.block.get_handle()
        self.block_graph_name = self.block.get_graph_names()[0]
        self.block_input_names = self.get_input_names(self.block_graph_name)
        self.block_output_names = self.get_output_names(self.block_graph_name)
        self.block_input_shape = []
        for index, input_name in enumerate(self.block_input_names):
            self.block_input_shape[index] = self.block.get_input_shape(self.block_graph_name, input_name)
        self.block_outputs = []
        self.block_output_shape = []
        self.block_output_dtype = []
        self.block_output_tensors = {}
        for index, output_name in enumerate(self.block_output_names):
            self.block_output_shape[index] = self.block.get_output_shape(self.block_graph_name, output_name)
            self.block_output_dtype[index] = self.block.get_output_dtype(self.block_graph_name, output_name)
            self.block_outputs[index] = sail.Tensor(self.block_handle, self.block_output_shape[index], self.block_output_dtype[index], True, True)
            self.block_output_tensors[output_name] = self.block_outputs[index] 

        # attn_bias
        self.attn_bias = attn_bias
        
    def embedding_forward(self, x, t, y):
        # print("use numpy data as input")
        # ref_data = np.load("./np_input.npy")
        # print(ref_data.shape)

        # input = sail.Tensor(self.handle, ref_data)
        # input_tensors = {self.input_name: input}
        # input_shapes = {self.input_name: self.input_shape}

        # self.net.process(self.graph_name, input_tensors,
        #                     input_shapes, self.output_tensors)
        ref_data = [x,t,y]
        input_tensors = {}
        input_shapes = {}
        for index, input_name in enumerate(self.embedding_input_names):
            input_tensors[input_name] = sail.Tensor(self.embedding_handle, ref_data[index])
            input_shapes[input_name] = self.embedding_input_shape[index]
        
        self.embedding.process(self.embedding_graph_name, input_tensors, input_shapes, self.embedding_output_tensors)
        return [x.asnumpy(self.embedding_output_shape[index]) for index, x in enumerate(self.embedding_outputs)]

    def net_forward(self, x, t, y, mask):
        ref_data = [x,t,y,mask]
        input_tensors = {}
        input_shapes = {}
        for index, input_name in enumerate(self.block_input_names):
            input_tensors[input_name] = sail.Tensor(self.block_handle, ref_data[index])
            input_shapes[input_name] = self.block_input_shape[index]
        
        self.block.process(self.block_graph_name, input_tensors, input_shapes, self.block_output_tensors)
        return [x.asnumpy(self.block_output_shape[index]) for index, x in enumerate(self.block_outputs)]

    def forward(self, x, timestep, y, mask=None):
        import numpy as np

        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        # embedding layer
        
        x,t,y = self.embedding_forward(x,timestep,y)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 0)
            y = np.squeeze(y, axis=1)[mask != 0].reshape(1, -1, x.shape[-1])
        # else:
        #     y_lens = [y.shape[2]] * y.shape[0]
        #     y = y.squeeze(1).view(1, -1, x.shape[-1])

        # blcok layer and final layer

        x = self.net_forward(x,t,y,self.attn_bias)[0]

        return x

    def forward_with_dpmsolver(self, x, timestep, y):
        """
        dpm solver donnot need variance prediction
        """

        model_out = self.forward(x, timestep, y)
        return np.split(model_out, 2, dim=1)[0]

