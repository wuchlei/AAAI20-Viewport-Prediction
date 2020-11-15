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
    criterion = spherical_mse()
    
    if args.restore:
        print('Restore model')
        encoder.load_state_dict(torch.load('model/attention/encoder.pth'))
        predictor.load_state_dict(torch.load('model/attention/predictor.pth'))

    learning_rate = constant.LEARNING_RATE

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate) #, weight_decay=1e-4)
    encoder_lr_scheduler = CosineAnnealingLR(encoder_optimizer, constant.FOV_NUM_EPOCHS, constant.LEARNING_RATE_MIN)

    predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate) #, weight_decay=1e-4)
    predictor_lr_scheduler = CosineAnnealingLR(predictor_optimizer, constant.FOV_NUM_EPOCHS, constant.LEARNING_RATE_MIN)

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
        train_batch_size = int(args.batch) * gpu_cnt
        valid_batch_size = int(args.batch) * gpu_cnt
    else:
        train_batch_size = constant.FOV_TRAIN_BATCH_SIZE * gpu_cnt
        valid_batch_size = constant.FOV_VALID_BATCH_SIZE * gpu_cnt

    train_dataset = s2_video_dataset()
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=6)

    valid_dataset = s2_video_dataset(stage=1)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True, num_workers=6)

    min_loss = np.Infinity
    countdown = 0
    for epoch in range(constant.FOV_NUM_EPOCHS):
        encoder.train()
        predictor.train()
        for i, (fov, motion, video, label, id) in enumerate(train_loader):  
            fov = fov.cuda()
            video = video.cuda()
            motion = motion.cuda()
            label = label.view(-1, 20, 40).cuda()
            inputs = [fov, motion, video]

            encoder_optimizer.zero_grad()
            predictor_optimizer.zero_grad()

            embedding = encoder(*inputs)
            outputs = predictor(embedding)

            accuracy, iou, precision, recall, f1 = cal_prediction_metric(outputs, label)
            loss = criterion(outputs, label)
            loss.backward()
        
            encoder_optimizer.step()
            predictor_optimizer.step()
            
            if i % 10 == 0:
                for j in range(0, constant.NUM_FOV_PREDICTION):
                    cv_toolkits.save_matrix_to_image(-label[j,:,:].cpu().detach().numpy(), 'visualize/epoch-%d-step-%d-%d-target.jpg'%(epoch+1,i,j))
                    cv_toolkits.save_matrix_to_image(-outputs[j,:,:].cpu().detach().numpy(), 'visualize/epoch-%d-step-%d-%d-result.jpg'%(epoch+1,i,j))

            if (i+1) % 1 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.6f' 
                    %(epoch+1, constant.FOV_NUM_EPOCHS, i+1, np.floor(len(train_dataset)/float(train_batch_size)), loss.data))
                for j in range(constant.NUM_FOV_PREDICTION):
                    print('Time window [%d/%d], Accuracy: %.6f, IOU: %.6f, Precision: %.6f, Recall: %.6f'
                        %(j+1, constant.NUM_FOV_PREDICTION, accuracy[j::4].mean().data, iou[j::4].mean().data, precision[j::4].mean().data, recall[j::4].mean().data))
            
            print()
                
        accuracies = [[] for i in range(constant.NUM_FOV_PREDICTION)]
        ious = [[] for i in range(constant.NUM_FOV_PREDICTION)]
        precisions = [[] for i in range(constant.NUM_FOV_PREDICTION)]
        recalls = [[] for i in range(constant.NUM_FOV_PREDICTION)]
        with torch.no_grad():
            encoder.eval()
            predictor.eval()

            for i, (fov, motion, video, label, id) in enumerate(tqdm(valid_loader, ascii = True)):  
                fov = fov.cuda()
                video = video.cuda()
                motion = motion.cuda()
                label = label.view(-1, 20, 40).cuda()
                inputs = [fov, motion, video]

                embedding = encoder(*inputs)
                outputs = predictor(embedding)

                accuracy, iou, precision, recall, f1 = cal_prediction_metric(outputs, label)
                
                for j in range(constant.NUM_FOV_PREDICTION):
                    accuracies[j].append(accuracy[j::4].cpu().mean().data.numpy())
                    ious[j].append(iou[j::4].cpu().mean().data.numpy())
                    precisions[j].append(precision[j::4].cpu().mean().data.numpy())
                    recalls[j].append(recall[j::4].cpu().mean().data.numpy())

        for t in range(constant.NUM_FOV_PREDICTION):
            print ('Time Window: [%d/%d], Accuracy: %.6f, IOU: %.6f, Precision: %.6f, Recall: %.6f'
                %(t+1, constant.NUM_FOV_PREDICTION, np.mean(accuracies[t]), np.mean(ious[t]), np.mean(precisions[t]), np.mean(recalls[t])))
        print()

        if args.multi_gpu:
            torch.save(encoder.module.cpu().state_dict(), output_model_path+'encoder_{}.pth'.format(epoch+1))
            torch.save(predictor.module.cpu().state_dict(), output_model_path+'predictor_{}.pth'.format(epoch+1))
        else:
            torch.save(encoder.cpu().state_dict(), output_model_path+'encoder_{}.pth'.format(epoch+1))
            torch.save(predictor.cpu().state_dict(), output_model_path+'predictor_{}.pth'.format(epoch+1))

        encoder.cuda()
        predictor.cuda()

        print('Model Saved\n')
        encoder_lr_scheduler.step()    
        predictor_lr_scheduler.step()

    print('Training finished')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--gpu-ids")
    parser.add_argument("--model")
    parser.add_argument("--batch")

    args = parser.parse_args()

    train(args)




            # if args.multi_gpu:
            #     print('fov weight', torch.max(torch.abs(encoder.module.motion_embed.weight)).cpu().detach().numpy())
            #     print('fov grad', torch.max(torch.abs(encoder.module.motion_embed.weight.grad)).cpu().detach().numpy())

            #     print('video weight', torch.max(torch.abs(encoder.module.video_embed.weight)).cpu().detach().numpy())
            #     print('video grad', torch.max(torch.abs(encoder.module.video_embed.weight.grad)).cpu().detach().numpy())

            #     print('motion weight', torch.max(torch.abs(encoder.module.motion_embed.weight)).cpu().detach().numpy())
            #     print('motion grad', torch.max(torch.abs(encoder.module.motion_embed.weight.grad)).cpu().detach().numpy())
