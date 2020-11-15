import torch
import torch.nn as nn
import numpy as np
import constant
from s2_conv_rnn import preference_rnn_encoder
from s2_model import vmf_to_pdf

nonlinear = nn.ReLU()
dropout = nn.Dropout(0.7)

class motion_fov_prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_embed = nn.Linear(90, 128) 
        self.motion_norm = nn.BatchNorm1d(128)
        constant.FOV_FINAL_EMBED = 128+1

    def forward(self, fov, motion, video):  # pylint: disable=W0221
        batch = fov.shape[0]
        sequnce = fov.shape[1]
        prediction = 4
        tot_lth = batch * sequnce * prediction

        motion = motion.unsqueeze(2)
        motion = motion.expand(-1, -1, 4, -1, -1)
        motion = motion.contiguous().view(-1, 90)
        motion = self.motion_embed(motion)
        motion = nonlinear(self.motion_norm(motion))
        motion = motion.view(tot_lth, -1)
        
        device_type=fov.device.type
        device_index=fov.device.index
        timewindow = torch.tensor(constant.PRED_TIME_WINDOW, dtype=torch.float32, device=torch.device(device_type, device_index)).unsqueeze(0)
        timewindow = timewindow.expand(batch*sequnce, -1).contiguous().view(-1, 1)
        
        h = torch.cat([timewindow, motion], 1)
        return h

class naive_fov_prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_embed = nn.Linear(90, 128) 
        self.motion_norm = nn.BatchNorm1d(128)

        self.fov_embed = nn.Conv2d(528, 4, 1)
        self.fov_norm = nn.BatchNorm2d(4)

        self.video_embed = nn.Conv2d(528, 4, 1)
        self.video_norm = nn.BatchNorm2d(4)

        constant.FOV_FINAL_EMBED = 784*2+128+1
        constant.INTEGRATE_FOV = False

    def forward(self, fov, motion, video):  # pylint: disable=W0221
        batch = fov.shape[0]
        sequnce = fov.shape[1]
        prediction = constant.NUM_FOV_PREDICTION
        tot_lth = batch * sequnce * prediction

        fov = fov.unsqueeze(2)
        fov = fov.expand(-1, -1, 4, -1, -1, -1)
        fov = fov.contiguous().view(-1, 528, 14, 14)
        fov = self.fov_embed(fov)
        fov = nonlinear(self.fov_norm(fov))
        fov = fov.view(tot_lth, -1)

        motion = motion.unsqueeze(2)
        motion = motion.expand(-1, -1, 4, -1, -1)
        motion = motion.contiguous().view(-1, 90)
        motion = self.motion_embed(motion)
        motion = nonlinear(self.motion_norm(motion))
        motion = motion.view(tot_lth, -1)

        video = video.view(-1, 528, 14, 14)
        video = self.video_embed(video)
        video = nonlinear(self.video_norm(video))
        video = video.view(tot_lth, -1)
        
        device_type=fov.device.type
        device_index=fov.device.index
        timewindow = torch.tensor(constant.PRED_TIME_WINDOW, dtype=torch.float32, device=torch.device(device_type, device_index)).unsqueeze(0)
        timewindow = timewindow.expand(batch*sequnce, -1).contiguous().view(-1, 1)
        
        h = torch.cat([timewindow, fov, motion, video], 1)
        return h

class basic_preference_fov_prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_embed = nn.Linear(90, 128) 
        self.motion_norm = nn.BatchNorm1d(128)

        self.n_layers = 1
        self.preference_rnn = nn.GRU(528, 784, self.n_layers, batch_first=True)

        self.video_embed = nn.Conv2d(528, 4, 1)
        self.video_norm = nn.BatchNorm2d(4)

        constant.FOV_FINAL_EMBED = 784*2+128+1
        constant.INTEGRATE_FOV = True

    def forward(self, fov, motion, video):  # pylint: disable=W0221
        device_type=fov.device.type
        device_index=fov.device.index

        batch = fov.shape[0]
        sequnce = fov.shape[1]
        prediction = constant.NUM_FOV_PREDICTION
        tot_lth = batch * sequnce * prediction

        h0 = torch.zeros(self.n_layers, batch, 784, dtype=torch.float32, device=torch.device(device_type, device_index))
        self.preference_rnn.flatten_parameters()
        preference, _ = self.preference_rnn(fov, h0)
        preference = preference.unsqueeze(2)
        preference = preference.expand(-1, -1, prediction, -1)
        preference = preference.contiguous().view(tot_lth, 784)
        preference = nonlinear(preference)

        motion = motion.unsqueeze(2)
        motion = motion.expand(-1, -1, prediction, -1, -1)
        motion = motion.contiguous().view(-1, 90)
        motion = self.motion_embed(motion)
        motion = nonlinear(self.motion_norm(motion))
        motion = motion.view(tot_lth, -1)

        video = video.view(-1, 528, 14, 14)
        video = self.video_embed(video)
        video = nonlinear(self.video_norm(video))
        video = video.view(tot_lth, -1)
        
        timewindow = torch.tensor(constant.PRED_TIME_WINDOW, dtype=torch.float32, device=torch.device(device_type, device_index)).unsqueeze(0)
        timewindow = timewindow.expand(batch*sequnce, -1).contiguous().view(-1, 1)
        
        h = torch.cat([timewindow, preference, motion, video], 1)
        return h

class attention_preference_fov_prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_embed = nn.Linear(90, 128) 
        self.motion_norm = nn.BatchNorm1d(128)

        self.n_layers = 1
        self.preference_rnn = nn.GRU(528, 784, self.n_layers, batch_first=True)

        self.video_embed = nn.Conv2d(528, 4, 1)
        self.video_norm = nn.BatchNorm2d(4)

        #attention for preference and motion
        self.video_feature = 528
        self.preference_feature = 784
        self.mapping_feature = 128

        self.W_p = nn.Linear(self.preference_feature, self.mapping_feature, bias=False)
        self.W_v = nn.Linear(self.video_feature, self.mapping_feature)
        self.W_pv = nn.Linear(self.mapping_feature, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        constant.FOV_FINAL_EMBED = 784*2+128+1
        constant.INTEGRATE_FOV = True

    def forward(self, fov, motion, video):  # pylint: disable=W0221
        device_type=fov.device.type
        device_index=fov.device.index

        batch = fov.shape[0]
        sequnce = fov.shape[1]
        prediction = constant.NUM_FOV_PREDICTION
        tot_lth = batch * sequnce * prediction

        h0 = torch.zeros(self.n_layers, batch, 784, dtype=torch.float32, device=torch.device(device_type, device_index))
        self.preference_rnn.flatten_parameters()
        preference, _ = self.preference_rnn(fov, h0)
        preference = preference.unsqueeze(2)
        preference = preference.expand(-1, -1, prediction, -1)

        preference = preference.contiguous().view(tot_lth, 784)
        p_att = self.W_p(preference).unsqueeze(1).expand(-1, 196, -1)

        v_att = video.view(tot_lth, 528, 196).transpose(-2, -1)
        v_att = v_att.contiguous().view(tot_lth*196, 528)
        v_att = self.W_v(v_att).view(tot_lth, 196, 128)

        v_att = self.tanh(v_att + p_att).view(tot_lth*196, 128)
        v_att = self.W_pv(v_att).view(tot_lth, 196)
        v_att = self.softmax(v_att).unsqueeze(-1)

        video = video.view(tot_lth, 196, 528)
        video = v_att*video

        video = torch.transpose(video, -2, -1)
        video = video.contiguous().view(tot_lth, 528, 14, 14)
        video = self.video_embed(video)
        video = nonlinear(self.video_norm(video))
        video = video.view(tot_lth, -1)

        motion = motion.unsqueeze(2)
        motion = motion.expand(-1, -1, prediction, -1, -1)
        motion = motion.contiguous().view(-1, 90)
        motion = self.motion_embed(motion)
        motion = nonlinear(self.motion_norm(motion))
        motion = motion.view(tot_lth, -1)
        
        timewindow = torch.tensor(constant.PRED_TIME_WINDOW, dtype=torch.float32, device=torch.device(device_type, device_index)).unsqueeze(0)
        timewindow = timewindow.expand(batch*sequnce, -1).contiguous().view(-1, 1)
        
        h = torch.cat([timewindow, preference, motion, video], 1)
        return h


class simple_predict(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_embed = 784*2+128+1
        self.fc = nn.Linear(self.final_embed, 1024)
        self.fc_norm = nn.BatchNorm1d(1024)

        self.predict = nn.Linear(1024, 800)

    def forward(self, h):  # pylint: disable=W0221
        h = self.fc(h)
        h = dropout(nonlinear(self.fc_norm(h)))
        h = nonlinear(self.predict(h))

        return h.view(-1, 20, 40)

class vmf_mdn(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_embed = constant.FOV_FINAL_EMBED

        self.fc1 = nn.Linear(self.final_embed, 1024)
        self.fc1_norm = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 1024)
        self.fc2_norm = nn.BatchNorm1d(1024)

        self.predictor_input = 1024
        self.n_mixtures = 3

        self.weight = nn.Linear(self.predictor_input, self.n_mixtures)
        self.mus= nn.ModuleList([nn.Linear(self.predictor_input, 3) for i in range(self.n_mixtures)])
        self.tau = nn.Linear(self.predictor_input, self.n_mixtures)

        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.cnt = 0

    def forward(self, h):  # pylint: disable=W0221
        h = self.fc1(h)
        h = dropout(nonlinear(self.fc1_norm(h)))

        h = self.fc2(h)
        h = dropout(nonlinear(self.fc2_norm(h)))

        mus = []
        for i in range(self.n_mixtures):
            mu = self.tanh(self.mus[i](h))
            mu_norm = torch.norm(mu, p=2, dim=1).detach()
            mu = mu.div(mu_norm.unsqueeze(-1).expand(-1, 3))
            mus.append(mu)

        mux = torch.stack([x[:, 0] for x in mus]).transpose(0,1)
        muy = torch.stack([x[:, 1] for x in mus]).transpose(0,1)
        muz = torch.stack([x[:, 2] for x in mus]).transpose(0,1)

        weight = self.softmax(self.weight(h))
        tau = self.relu(self.tau(h))
        return vmf_to_pdf(weight, mux, muy, muz, tau)

def cal_prediction_metric(prediction, label):
    fov = prediction>0.5

    label = label.type(torch.uint8)
    correct = torch.sum(fov&label, dim=-1).squeeze(-1)
    correct = torch.sum(correct, dim=-1).squeeze(-1)
    correct = correct.type(torch.float32)

    accuracy = torch.sum(fov == label, dim=-1).squeeze(-1)
    accuracy = torch.sum(accuracy, dim=-1).squeeze(-1)
    accuracy = accuracy.type(torch.float32)
    accuracy = accuracy/800.

    iou = torch.sum(fov | label, dim=-1).squeeze(-1)
    iou = torch.sum(iou, dim=-1).squeeze(-1)
    iou = iou.type(torch.float32)
    iou = correct/iou

    label = label.type(torch.float32)
    fov = fov.type(torch.float32)

    tot_fov = torch.sum(fov, dim=-1).squeeze(-1)
    tot_fov = torch.sum(tot_fov, dim=-1).squeeze(-1)+constant.EPSILON

    tot_label = torch.sum(label, dim=-1).squeeze(-1)
    tot_label = torch.sum(tot_label, dim=-1).squeeze(-1)

    precision  = correct/tot_fov
    recall = correct/tot_label
    f1 = 2*precision*recall/(precision+recall)

    return accuracy, iou, precision, recall, f1
    
    
