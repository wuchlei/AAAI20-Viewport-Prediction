import torch
import torch.nn as nn
import torch.nn.functional as F
from s2cnn import s2_equatorial_grid, S2Convolution, so3_equatorial_grid, SO3Convolution, so3_integrate
from s2cnn import s2_near_identity_grid, so3_near_identity_grid
from s2cnn import s2_soft_grid, so3_soft_grid
import numpy as np
import constant
import lie_learn.spaces.S3 as S3

nonlinear = nn.ReLU
# nonlinear = nn.Sigmoid

def cal_haar_measure_weight(b):
    import lie_learn.spaces.S3 as S3
    w = torch.tensor(S3.quadrature_weights(b))

    if constant.IS_GPU:
        return w.cuda()
    else:
        return w    

def s2_integrate(x):
    """
    Integrate a signal on SO(3) using the Haar measure
    """
    device_type=x.device.type
    device_index=x.device.index

    b = x.size(-2) // 2
    w =  torch.tensor(S3.quadrature_weights(b), dtype=torch.float32, device=torch.device(device_type, device_index))

    x = torch.sum(x, dim=-1).squeeze(-1) 

    sz = x.size()
    x = x.view(-1, 2 * b)
    w = w.view(2 * b, 1)
    x = torch.mm(x, w).squeeze(-1)
    x = x.view(*sz[:-1])
    return x

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        #m.weight.data.fill_(1.)
        m.bias.data.fill_(0.1)
    elif type(m) == S2Convolution or type(m) == SO3Convolution:
        torch.nn.init.xavier_normal_(m.kernel)
        m.bias.data.fill_(0.1)


class spherical_residual_image_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = constant.ENCODER_FEATURES
        self.bandwidths = constant.ENCODER_BANDWIDTH
        self.groups = constant.ENCODER_GROUPS
        assert len(self.bandwidths) == len(self.features)

        sequence = []

        # S2 layer
        grid = s2_near_identity_grid(n_alpha=2 * self.bandwidths[0], n_beta=2) #, max_beta=0.2)
        # grid = s2_near_identity_grid(n_alpha=self.bandwidths[0], n_beta=4) #, max_beta=0.2)
        # grid = s2_equatorial_grid(n_alpha=2 * self.bandwidths[0], n_beta=1)

        sequence.append(S2Convolution(self.features[0], self.features[1], self.bandwidths[0], self.bandwidths[1], grid))
        # sequence.append(nn.BatchNorm3d(self.features[1], affine=True))
        sequence.append(nn.GroupNorm(self.groups[0], self.features[1]))
        sequence.append(nonlinear())

        # SO3 layers
        for l in range(1, len(self.features)-1, 2):
            f_in = self.features[l]
            f_mid = self.features[l + 1]
            f_out = self.features[l + 2]

            b_in = self.bandwidths[l]
            b_mid = self.bandwidths[l + 1]
            b_out = self.bandwidths[l + 2]

            n_group_mid = self.groups[l]
            n_group_out = self.groups[l + 1]

            # sequence.append(so3_residual_block(f_in, f_mid, f_out, b_in, b_mid, b_out, n_group_mid, n_group_out))
            sequence.append(so3_residual_block(f_in, f_mid, f_out, b_in, b_mid, b_out, n_group_mid, n_group_out))
        self.sequential = nn.Sequential(*sequence)

    def forward(self, x):  # pylint: disable=W0221
        x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        x = so3_integrate(x)  # [batch, feature]
        #x = x.view(x.size(0), x.size(1), -1).max(-1)[0]
        # x = x.view(x.size(0), -1)

        return x

class so3_residual_block(nn.Module):
    def __init__(self, f_in, f_mid, f_out, b_in, b_mid, b_out, n_group_mid, n_group_out):
        super().__init__()
        grid1 = so3_near_identity_grid(n_alpha=2 * b_in, n_beta=2, n_gamma=2) #max_beta=0, max_gamma=0, 
        grid2 = so3_near_identity_grid(n_alpha=2 * b_mid, n_beta=2, n_gamma=2) #max_beta=0, max_gamma=0, 

        # grid1 = so3_near_identity_grid(n_alpha=b_in, n_beta=3, n_gamma=3) #max_beta=0, max_gamma=0, 
        # grid2 = so3_near_identity_grid(n_alpha=b_mid, n_beta=3, n_gamma=3) #max_beta=0, max_gamma=0, 

        # grid1 = so3_equatorial_grid(n_alpha=2 * b_in, n_beta=1, n_gamma=1)
        # grid2 = so3_equatorial_grid(n_alpha=2 * b_in, n_beta=1, n_gamma=1)

        self.left = nn.Sequential(
            SO3Convolution(f_in, f_mid, b_in, b_mid, grid1),
            # nn.BatchNorm3d(f_mid, affine=True),
            nn.GroupNorm(n_group_mid, f_mid),
            SO3Convolution(f_mid, f_out, b_mid, b_out, grid2),
            # nn.BatchNorm3d(f_out, affine=True),
            nn.GroupNorm(n_group_out, f_out),
            nonlinear()
        )

        self.shortcut = SO3Shortcut(f_in, f_out, b_in, b_out)

    def forward(self, x):
        x = self.left(x) + self.shortcut(x)
        # x += self.shortcut(x)
        x = F.relu(x)
        return x

class spherical_image_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = constant.ENCODER_FEATURES
        self.bandwidths = constant.ENCODER_BANDWIDTH
        self.groups = constant.ENCODER_GROUPS
        assert len(self.bandwidths) == len(self.features)

        sequence = []

        # S2 layer
        # grid = s2_near_identity_grid(n_alpha = self.bandwidths[0], n_beta= 4)
        grid = s2_near_identity_grid(n_alpha=2 * self.bandwidths[0], n_beta=2)
        # grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1)
        # grid = s2_soft_grid(self.bandwidths[0])
        sequence.append(S2Convolution(self.features[0], self.features[1], self.bandwidths[0], self.bandwidths[1], grid))
        sequence.append(nn.BatchNorm3d(self.features[1], affine=True))
        # sequence.append(nn.GroupNorm(self.groups[0], self.features[1]))
        sequence.append(nonlinear())

        # SO3 layers
        for l in range(1, len(self.features)-1):
            f_in = self.features[l]
            f_out = self.features[l + 1]

            b_in = self.bandwidths[l]
            b_out = self.bandwidths[l + 1]
            
            n_group = self.groups[l]

            # grid = so3_near_identity_grid(n_alpha= b_in, n_beta= 3, n_gamma= 3)  
            grid = so3_near_identity_grid(n_alpha=2 * b_in, n_beta=2, n_gamma=2)  
            # grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1)

            sequence.append(SO3Convolution(f_in, f_out, b_in, b_out, grid))

            sequence.append(nn.BatchNorm3d(f_out, affine=True))
            # sequence.append(nn.GroupNorm(n_group, f_out))

            sequence.append(nonlinear())

        self.sequential = nn.Sequential(*sequence)

    def forward(self, x):  # pylint: disable=W0221
        x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        # x = so3_integrate(x)  # [batch, feature]
        #x = x.view(x.size(0), x.size(1), -1).max(-1)[0]
        #x = x.view(x.size(0), -1)

        return x

class ucf_image_predictor(nn.Module):
    def __init__(self):
        super().__init__()
        predict = []

        for i in range(len(constant.UCF_PREDICTOR_FC)-2):
            predict.append(nn.Linear(constant.UCF_PREDICTOR_FC[i], constant.UCF_PREDICTOR_FC[i+1]))
            predict.append(nn.BatchNorm1d(constant.UCF_PREDICTOR_FC[i+1], affine=True))
            predict.append(nonlinear())
            predict.append(nn.Dropout(0.5))

        predict.append(nn.Linear(constant.UCF_PREDICTOR_FC[-2], constant.UCF_PREDICTOR_FC[-1]))
        predict.append(nn.Softmax(dim=-1))

        self.predictor = nn.Sequential(*predict)

    def forward(self, x):
        # x = so3_integrate(x)
        h = self.predictor(x)
        return h

class spherical_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = constant.ENCODER_FEATURES
        self.bandwidths = constant.ENCODER_BANDWIDTH
        assert len(self.bandwidths) == len(self.features)

        sequence = []

        # S2 layer
        grid = s2_near_identity_grid(n_alpha=2 * self.bandwidths[0], n_beta=2)
        # grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1)
        # grid = s2_soft_grid(self.bandwidths[0])
        sequence.append(S2Convolution(self.features[0], self.features[1], self.bandwidths[0], self.bandwidths[1], grid))
        sequence.append(nn.BatchNorm3d(self.features[1], affine=True))
        sequence.append(nn.ReLU())

        # SO3 layers
        for l in range(1, len(self.features)-1):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l + 1]
            b_in = self.bandwidths[l]
            b_out = self.bandwidths[l + 1]

            grid = so3_near_identity_grid(n_alpha=2 * b_in, n_beta=2, n_gamma=2)  
            # grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1)
            # grid = so3_soft_grid(b_in)
            sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid))
            sequence.append(nn.BatchNorm3d(nfeature_out, affine=True))
            sequence.append(nn.ReLU())

        self.sequential = nn.Sequential(*sequence)

        self.input_size = constant.REPRESENTATION_SIZE        
        self.hidden_size = constant.ENCODER_HIDDEN_SIZE
        self.n_layers = constant.ENCODER_LAYERS
        self.rnn = nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.n_layers, batch_first=True)

        # output_features = self.features[-2]
        # self.out_layer = nn.Linear(output_features, self.features[-1])

    def forward(self, x):  # pylint: disable=W0221
        input_shape = x.shape
        assert len(input_shape) == 5

        x = x.contiguous().view(input_shape[0]*input_shape[1], input_shape[2], input_shape[3], input_shape[4])

        x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        # x = so3_integrate(x)  # [batch, feature]
        x = x.view(x.size(0), x.size(1), -1).max(-1)[0]

        x = x.contiguous().view(input_shape[0], input_shape[1], -1)
        
        h0 = torch.zeros(self.n_layers, input_shape[0], self.hidden_size)
        if constant.IS_GPU:
            h0 = h0.cuda()
        
        # print(x.shape, h0.shape)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x, h0)
        x = x[:,-1,:]

        return x
        # x = x.contiguous().view(h.size(0)*h.size(1), h.size(2))
        # x = self.out_layer(x)
        # return F.log_softmax(x, dim=1)

class SO3Shortcut(nn.Module):
    def __init__(self, f_in, f_out, b_in, b_out):
        super().__init__()
        assert b_out <= b_in

        self.tmp = [f_in, f_out, b_in, b_out]
        if (f_in != f_out) or (b_in != b_out):
            self.conv = nn.Sequential(
                SO3Convolution(nfeature_in=f_in, nfeature_out=f_out, 
                b_in=b_in, b_out=b_out, grid=((0, 0, 0), )),
                nn.BatchNorm3d(f_out, affine=True)
            )
        else:
            self.conv = None

    def forward(self, x):
        '''
        :x:      [batch, feature_in,  beta, alpha, gamma]
        :return: [batch, feature_out, beta, alpha, gamma]
        '''
        if self.conv is not None:
            return self.conv(x)
        else:
            return x

class spherical_residual_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = constant.ENCODER_FEATURES
        self.bandwidths = constant.ENCODER_BANDWIDTH
        assert len(self.bandwidths) == len(self.features)

        sequence = []

        # S2 layer
        grid = s2_near_identity_grid(n_alpha=2 * self.bandwidths[0], n_beta=2, max_beta=0.2)
        # grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1)
        sequence.append(S2Convolution(self.features[0], self.features[1], self.bandwidths[0], self.bandwidths[1], grid))
        sequence.append(nn.BatchNorm3d(self.features[1], affine=True))
        sequence.append(nn.ReLU())

        # SO3 layers
        for l in range(1, len(self.features)-1):
            f_in = self.features[l]
            f_out = self.features[l + 1]
            b_in = self.bandwidths[l]
            b_out = self.bandwidths[l + 1]

            sequence.append(so3_residual_block(f_in, f_out, b_in, b_out))

        self.sequential = nn.Sequential(*sequence)

        self.input_size = constant.REPRESENTATION_SIZE
        self.hidden_size = constant.ENCODER_HIDDEN_SIZE
        self.n_layers = constant.ENCODER_LAYERS

        self.rnn = nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.n_layers, batch_first=True)

    def forward(self, x):  # pylint: disable=W0221
        input_shape = x.shape
        assert len(input_shape) == 5

        x = x.contiguous().view(input_shape[0]*input_shape[1], input_shape[2], input_shape[3], input_shape[4])

        x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        # x = so3_integrate(x)  # [batch, feature]

        x = x.contiguous().view(input_shape[0], input_shape[1], -1)
        h0 = torch.zeros(self.n_layers, input_shape[0], self.hidden_size)
        if constant.IS_GPU:
            h0 = h0.cuda()

        self.rnn.flatten_parameters()
        x, _ = self.rnn(x, h0)
        x = x[:,-1,:]

        return x

class mdn(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_size = constant.PREDICTOR_HIDDEN_SIZE
        self.n_layers = constant.PREDICTOR_LAYERS
        self.input_size = constant.ENCODER_HIDDEN_SIZE
        self.n_mixtures = constant.PREDICTOR_MIXTURES

        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.n_layers, batch_first=True) #, dropout=0.5)
        # self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.weight = nn.Linear(self.hidden_size, self.n_mixtures)
        self.mus= nn.ModuleList([nn.Linear(self.hidden_size, 3) for i in range(self.n_mixtures)])
        self.tau = nn.Linear(self.hidden_size, self.n_mixtures)

        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()
        
        self.dropout = nn.Dropout(p=0.5)
        self.cnt = 0

    def forward(self, h):
        assert len(h.shape) == 3
        
        if constant.IS_GPU:
            h0 = torch.zeros(self.n_layers, h.size(0), self.hidden_size).cuda()
        else:
            h0 = torch.zeros(self.n_layers, h.size(0), self.hidden_size)

        self.rnn.flatten_parameters()
        h, _ = self.rnn(h, h0)
        h = h[:,:,:]
        # h = h[:,1:,:]
        h = h.contiguous().view(h.size(0)*h.size(1), h.size(2)) # batch_size * seq_len, gru_hidden_size

        mus = []
        for i in range(self.n_mixtures):
            mu = self.tanh(self.mus[i](h))
            mu_norm = torch.norm(mu, p=2, dim=1).detach()
            mu = mu.div(mu_norm.unsqueeze(-1).expand(-1, 3))
            mus.append(mu)

        mux = torch.stack([x[:, 0] for x in mus]).transpose(0,1)
        muy = torch.stack([x[:, 1] for x in mus]).transpose(0,1)
        muz = torch.stack([x[:, 2] for x in mus]).transpose(0,1)

        # weight = self.softmax(self.weight(h))
        # tau = self.exp_activation(self.tau(h))
        tau = torch.ones((mux.shape))*4
        weight = torch.ones((mux.shape))*0.5

        if constant.IS_GPU:
            tau = tau.cuda()
            weight = weight.cuda()

        idx = 1
        if self.cnt % 10 == 0:
            print('vMF parameters', mux[idx].cpu().detach().numpy(), muy[idx].cpu().detach().numpy(), muz[idx].cpu().detach().numpy(),
            weight[idx].cpu().detach().numpy(), tau[idx].cpu().detach().numpy())
            cnt = 0
        else:
            cnt += 1

        return weight, mux, muy, muz, tau

    def exp_activation(self, x):
        return torch.exp(x)

def vmf_to_pdf(weight, mux, muy, muz, tau):
    forward_vec = cal_forward_vector()

    lth = weight.shape[0]

    mux = mux.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, constant.LABEL_HEI, constant.LABEL_WID)
    muy = muy.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, constant.LABEL_HEI, constant.LABEL_WID)
    muz = muz.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, constant.LABEL_HEI, constant.LABEL_WID)

    weight = weight.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, constant.LABEL_HEI, constant.LABEL_WID)
    tau = tau.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, constant.LABEL_HEI, constant.LABEL_WID)

    x = forward_vec[0].unsqueeze(0).unsqueeze(0).repeat(lth, constant.PREDICTOR_MIXTURES, 1, 1)
    y = forward_vec[1].unsqueeze(0).unsqueeze(0).repeat(lth, constant.PREDICTOR_MIXTURES, 1, 1)
    z = forward_vec[2].unsqueeze(0).unsqueeze(0).repeat(lth, constant.PREDICTOR_MIXTURES, 1, 1)

    if constant.IS_GPU:
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()

    exp_tau = torch.exp(tau)
    
    dot_product = mux*x + muy*y + muz*z
    exp = torch.exp(tau*dot_product)
    
    # z3 = tau/(2*np.pi*(np.exp(tau)-np.exp(-tau))+constant.EPSILON)
    z3 = tau/(2*np.pi*(torch.exp(tau)-torch.exp(-tau))+constant.EPSILON)

    # h = exp*z3*weight
    h = weight*z3*exp
    h = torch.sum(h, 1)

    max_prob = torch.max(h.view(-1, 800), 1)[0]
    min_prob = torch.min(h.view(-1, 800), 1)[0]
    range_prob = max_prob-min_prob+constant.EPSILON

    min_prob = min_prob.unsqueeze(-1).unsqueeze(-1).expand(-1, 20, 40)
    range_prob = range_prob.unsqueeze(-1).unsqueeze(-1).expand(-1, 20, 40)

    h = (h-min_prob)/range_prob

    return h

        
def cal_forward_vector():
    img_wid, img_hei = constant.LABEL_WID, constant.LABEL_HEI

    x_range = np.arange(0, img_wid)
    y_range = np.arange(img_hei / 2, -img_hei / 2, -1)

    tx, ty = np.meshgrid(x_range, y_range)
    alpha = tx.astype(float) / img_wid * 2 * np.pi
    beta = ty.astype(float) / img_hei * np.pi

    x = np.cos(alpha) * np.cos(beta)
    y = np.sin(beta)
    z = -np.sin(alpha) * np.cos(beta)
    
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    z = torch.from_numpy(z).float()

    return x, y, z


class neg_log_likely_hood(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        loss = output
        loss = -torch.log(loss+constant.EPSILON) * target
        loss = s2_integrate(loss)
        loss = loss.mean()
        return loss


class spherical_mse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        loss = torch.pow(output-target, 2)
        loss = s2_integrate(loss)
        loss = loss.mean()
        return loss

class g_mdn(nn.Module):
    def __init__(self, input_size=constant.ENCODER_HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = constant.PREDICTOR_HIDDEN_SIZE
        self.num_layers = constant.PREDICTOR_LAYERS
        self.input_size = input_size
        self.num_mixtures = constant.PREDICTOR_MIXTURES

        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True) #, dropout=0.5)
        # self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.weight = nn.Linear(self.hidden_size, self.num_mixtures)
        self.mux = nn.Linear(self.hidden_size, self.num_mixtures)
        self.muy = nn.Linear(self.hidden_size, self.num_mixtures)
        self.sigx = nn.Linear(self.hidden_size, self.num_mixtures)
        self.sigy = nn.Linear(self.hidden_size, self.num_mixtures)
        # self.ro = nn.Linear(hidden_size, num_mixtures)

        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.5)
        self.EPSILON = constant.EPSILON

        self.cnt = 0

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if constant.IS_GPU:
            h0 = h0.cuda()
            c0 = c0.cuda()
        
        h = x
        
        self.rnn.flatten_parameters()
        h, _ = self.rnn(h, h0)
        h = h[:,:,:]
        # h = h[:,1:,:]
        h = h.contiguous().view(h.size(0)*h.size(1), h.size(2))

        weight = self.softmax(self.weight(h))

        mux = self.sigmoid(self.mux(h))
        muy = self.sigmoid(self.muy(h))
        sigx = self.exp_activation(self.sigx(h))
        sigy = self.exp_activation(self.sigy(h))

        idx = 1
        if self.cnt % 10 == 0:
            print('GMM parameters', mux[idx].cpu().detach().numpy(), muy[idx].cpu().detach().numpy(), 
            sigx[idx].cpu().detach().numpy(), sigy[idx].cpu().detach().numpy(),
            weight[idx].cpu().detach().numpy())
            cnt = 0
        else:
            cnt += 1

        return weight, mux, muy, sigx, sigy 

    def exp_activation(self, x):
        return torch.exp(x)


class ucf_predictor(nn.Module):
    def __init__(self, input_size=constant.ENCODER_HIDDEN_SIZE):
        super().__init__()

        self.fc1 = nn.Linear(input_size, constant.UCF_FC_1)
        self.fc2 = nn.Linear(constant.UCF_FC_1, constant.UCF_FC_2)
        self.fc3 = nn.Linear(constant.UCF_FC_2, constant.UCF_CLASS)
        
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        # h = self.dropout(h)

        h = self.relu(self.fc2(h))
        # h = self.dropout(h)

        h = self.softmax(self.fc3(h))

        return h


def gmm_to_pdf(weight, mux, muy, sigx, sigy):
    lth = weight.shape[0]

    x = torch.linspace(0, constant.LABEL_WID-1, constant.LABEL_WID).repeat(constant.LABEL_HEI, 1)/constant.LABEL_WID
    x = x.unsqueeze(0).unsqueeze(0).repeat(lth, constant.PREDICTOR_MIXTURES, 1, 1)

    y = torch.linspace(0, constant.LABEL_HEI-1, constant.LABEL_HEI).view(-1, 1).repeat(1, constant.LABEL_WID)/constant.LABEL_HEI
    y = y.unsqueeze(0).unsqueeze(0).repeat(lth, constant.PREDICTOR_MIXTURES, 1, 1)

    if constant.IS_GPU:
        x = x.cuda()
        y = y.cuda()

    weight = weight.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, constant.LABEL_HEI, constant.LABEL_WID)
    mux = mux.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, constant.LABEL_HEI, constant.LABEL_WID)
    muy = muy.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, constant.LABEL_HEI, constant.LABEL_WID)
    sigx = sigx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, constant.LABEL_HEI, constant.LABEL_WID)
    sigy = sigy.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, constant.LABEL_HEI, constant.LABEL_WID)
    
    x_minus_mux = x - mux      
    y_minus_muy = y - muy
    exp = -0.5 * (torch.pow(x_minus_mux, 2)/(torch.pow(sigx, 2)+constant.EPSILON) +
                    torch.pow(y_minus_muy, 2)/(torch.pow(sigy, 2)+constant.EPSILON))

    h = weight*torch.exp(exp)/(constant.EPSILON+2*np.pi*sigx*sigy)
    h = torch.sum(h, 1)
    return h

if __name__ == '__main__':
    encoder = spherical_residual_encoder()
    import ucf_dataset
    dataset = ucf_dataset.ucf_dataset()
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=6)
    inputs, labels = next(iter(loader))

    res = encoder(inputs)
    print(res.shape)
