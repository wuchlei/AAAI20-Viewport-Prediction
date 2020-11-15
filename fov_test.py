import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import torch.nn.functional as f
import time, constant, cv_toolkits

from fov_dataset import s2_video_dataset
from fov_model import naive_fov_prediction, vmf_mdn, simple_predict, cal_prediction_metric, basic_preference_fov_prediction, motion_fov_prediction, attention_preference_fov_prediction
from s2_model import neg_log_likely_hood, vmf_to_pdf, spherical_mse

import json

def train(args):
    if args.model == 'attention':
        encoder = attention_preference_fov_prediction()
        output_model_path = 'model/attention/'
    elif args.model == 'basic':
        encoder = basic_preference_fov_prediction()
        output_model_path = 'model/basic/'
    elif args.model == 'motion':
        encoder = motion_fov_prediction()
        output_model_path = 'model/motion/'
    elif args.model == 'naive':
        encoder = naive_fov_prediction()
        output_model_path = 'model/naive/'
    predictor = vmf_mdn()

    encoder.load_state_dict(torch.load(output_model_path+'encoder.pth'))
    predictor.load_state_dict(torch.load(output_model_path+'predictor.pth'))

    gpu_cnt = 1
    gpu_ids = [0]
    if args.gpu_ids != None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    if args.multi_gpu: # and torch.cuda.device_count() > 1:
        if len(gpu_ids) == 1:
            gpu_ids = [0, 1]

        gpu_cnt = len(gpu_ids)
        encoder = nn.DataParallel(encoder, device_ids=gpu_ids)
        predictor = nn.DataParallel(predictor, device_ids=gpu_ids)
        
    print('Using {} GPUs'.format(gpu_cnt))
    encoder.cuda()
    predictor.cuda()

    if args.batch != None:
        test_batch_size = int(args.batch) * gpu_cnt
    else:
        test_batch_size = constant.FOV_VALID_BATCH_SIZE * gpu_cnt

    test_dataset = s2_video_dataset(stage=2)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=6)

    min_loss = np.Infinity
    countdown = 0
   
    t_accuracies = [[] for i in range(constant.NUM_FOV_PREDICTION)]
    t_ious = [[] for i in range(constant.NUM_FOV_PREDICTION)]
    t_precisions = [[] for i in range(constant.NUM_FOV_PREDICTION)]
    t_recalls = [[] for i in range(constant.NUM_FOV_PREDICTION)]

    p_accuracies = [[list() for j in range(constant.NUM_FOV_PREDICTION)] for i in range(constant.FOV_SEQ_LEN)]
    p_ious = [[list() for j in range(constant.NUM_FOV_PREDICTION)] for i in range(constant.FOV_SEQ_LEN)]
    p_precisions = [[list() for j in range(constant.NUM_FOV_PREDICTION)] for i in range(constant.FOV_SEQ_LEN)]
    p_recalls = [[list() for j in range(constant.NUM_FOV_PREDICTION)] for i in range(constant.FOV_SEQ_LEN)]

    with torch.no_grad():
        encoder.eval()
        predictor.eval()

        for i, (fov, motion, video, label, id) in enumerate(tqdm(test_loader, ascii = True)):  
            fov = fov.cuda()
            video = video.cuda()
            motion = motion.cuda()
            label = label.view(-1, 20, 40).cuda()
            inputs = [fov, motion, video]

            embedding = encoder(*inputs)
            outputs = predictor(embedding)

            accuracy, iou, precision, recall, f1 = cal_prediction_metric(outputs, label)
            
            step = constant.NUM_FOV_PREDICTION
            for j in range(constant.NUM_FOV_PREDICTION):
                t_accuracies[j].append(float(accuracy[j::step].cpu().mean().data.numpy()))
                t_ious[j].append(float(iou[j::step].cpu().mean().data.numpy()))
                t_precisions[j].append(float(precision[j::step].cpu().mean().data.numpy()))
                t_recalls[j].append(float(recall[j::step].cpu().mean().data.numpy()))

            step = constant.FOV_SEQ_LEN * constant.NUM_FOV_PREDICTION
            for j in range(constant.FOV_SEQ_LEN):
                for k in range(constant.NUM_FOV_PREDICTION):
                    p_accuracies[j][k].append(float(accuracy[j*constant.NUM_FOV_PREDICTION+k::step].cpu().mean().data.numpy()))
                    p_ious[j][k].append(float(iou[j*constant.NUM_FOV_PREDICTION+k::step].cpu().mean().data.numpy()))
                    p_precisions[j][k].append(float(precision[j*constant.NUM_FOV_PREDICTION+k::step].cpu().mean().data.numpy()))
                    p_recalls[j][k].append(float(recall[j*constant.NUM_FOV_PREDICTION+k::step].cpu().mean().data.numpy()))

        for t in range(constant.NUM_FOV_PREDICTION):
            print ('Time Window: [%d/%d], Accuracy: %.6f, IOU: %.6f, Precision: %.6f, Recall: %.6f'
                %(t+1, constant.NUM_FOV_PREDICTION, np.mean(t_accuracies[t]), np.mean(t_ious[t]), np.mean(t_precisions[t]), np.mean(t_recalls[t])))
        print()

    result = {}
    result['t_accuracies'] = t_accuracies
    result['t_ious'] = t_ious
    result['t_precisions'] = t_precisions
    result['t_recalls'] = t_recalls
    
    result['p_accuracies'] = p_accuracies
    result['p_ious'] = p_ious
    result['p_precisions'] = p_precisions
    result['p_recalls'] = p_recalls

    json.dump(result, open('result_naive.json', 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--gpu-ids")
    parser.add_argument("--model")
    parser.add_argument("--batch")
    args = parser.parse_args()

    train(args)
