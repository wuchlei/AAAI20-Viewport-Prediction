import torch
import torch.nn as nn
import numpy as np
from torch.nn import init

from s2cnn import S2Convolution, SO3Convolution, so3_integrate
from s2cnn import s2_near_identity_grid, so3_near_identity_grid

import constant

class s2_conv_rnn_cell(nn.Module):
    def __init__(self, f_in, f_hidden, bandwidth):
        """
        Initialize the ConvLSTM cell
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super().__init__()
        self.input_features = f_in
        self.hidden_features = f_hidden
        self.bandwidth = bandwidth

        grid = s2_near_identity_grid(n_alpha=2 * bandwidth)  
        self.reset_gate = S2Convolution(f_in+f_hidden, f_hidden, bandwidth, bandwidth, grid)
        self.update_gate = S2Convolution(f_in+f_hidden, f_hidden, bandwidth, bandwidth, grid)
        self.output_gate = S2Convolution(f_in+f_hidden, f_hidden, bandwidth, bandwidth, grid)


    def init_hidden(self, batch_size):
        state_size = [batch_size, self.hidden_features] + [self.bandwidth*2]*2
        if constant.IS_GPU:
            hidden_state = torch.zeros(state_size).cuda()
        else:
            hidden_state = torch.zeros(state_size)
        return hidden_state

    def forward(self, inputs, prev_state):
        stacked_inputs = torch.cat([inputs, prev_state], dim=1)
        update = torch.max(torch.sigmoid(self.update_gate(stacked_inputs)), -1).unsqueeze(-1)
        reset = torch.max(torch.sigmoid(self.reset_gate(stacked_inputs)), -1).unsqueeze(-1)
        out_inputs = torch.max(torch.tanh(self.output_gate(torch.cat([inputs, prev_state * reset], dim=1))), -1).unsqueeze(-1)
        new_state = prev_state * (1 - update) + out_inputs * update

        return update
        # return new_state

class s2_conv_rnn(nn.Module):
    def __init__(self, f_in, f_hidden, bandwidth, n_layers, return_all_layers=False):
        super().__init__()
        
        self.f_in = f_in
        if type(f_hidden) != list:
            self.f_hidden = [f_hidden]*n_layers
        else:
            assert len(f_hidden) == n_layers, '`f_hidden` must have the same length as n_layers'
            self.f_hidden = f_hidden

        self.bandwidth = bandwidth
        self.n_layers = n_layers

        self.return_all_layers = return_all_layers
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.f_in
            else:
                input_dim = self.f_hidden[i-1]

            cell = s2_conv_rnn_cell(input_dim, self.f_hidden[i], self.bandwidth)
            cells.append(cell)

        self.cell_list = nn.ModuleList(cells)

    def forward(self, inputs, hidden_state=None):
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=inputs.size(0))

        layer_output_list = []
        # last_state_list   = []

        seq_len = inputs.size(1)
        cur_layer_input = inputs

        for layer_idx in range(self.n_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](inputs=cur_layer_input[:, t], prev_state=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            # last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            # last_state_list   = last_state_list[-1]

        return layer_output_list #, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.n_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states


class s2_rnn_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = s2_conv_rnn(*constant.S2_RNN_ENCODER)

    def forward(self, x):  # pylint: disable=W0221
        h = self.rnn(x)
        return h[:, -1]

class conv_rnn_cell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        """
        super().__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def init_hidden(self, batch_size):
        if constant.IS_GPU:
            return torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda()
        else:
            return torch.zeros(batch_size, self.hidden_dim, self.height, self.width)

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class conv_rnn(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        """
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super().__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(conv_rnn_cell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], # (b,t,c,h,w)
                                              h_cur=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            # last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            # last_state_list   = last_state_list[-1:]

        return layer_output_list #, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class rnn_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        height = constant.ENCODER_BANDWIDTH[-1]*2
        width = height
        channels = constant.ENCODER_FEATURES[-1]
        hidden_dim = channels
        kernel_size = (3,3)
        num_layers = 1

        self.rnn = conv_rnn(input_size=(height, width),
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False)

    def forward(self, x):  # pylint: disable=W0221
        h = self.rnn(x)
        return h[:, -1]

class preference_rnn_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        height = constant.ENCODER_BANDWIDTH[-1]*2
        width = height
        channels = constant.ENCODER_FEATURES[-1]
        hidden_dim = channels
        kernel_size = (3,3)
        num_layers = 1

        self.rnn = conv_rnn(input_size=(height, width),
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False)

    def forward(self, x):  # pylint: disable=W0221
        h = self.rnn(x)
        return h
