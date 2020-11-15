import torch
import torch.nn as nn
from s2_model import mdn, g_mdn, spherical_encoder, spherical_residual_encoder, neg_log_likely_hood, vmf_to_pdf, gmm_to_pdf
from s2_dataset import image_dataset, simple_image_dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import argparse
import torch.nn.functional as f
import time, constant
import cv_toolkits

np.set_printoptions(precision=2)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train(args):
    # my_dataset = image_dataset()
    my_dataset = simple_image_dataset()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = spherical_residual_encoder()
    predictor = g_mdn()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=constant.LEARNING_RATE)
    predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=constant.LEARNING_RATE)

    gpu_cnt = 1
    if args.multi_gpu: # and torch.cuda.device_count() > 1:
        if args.gpu_ids != None:
            gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        else:
            gpu_ids = [0, 1]

        gpu_cnt = len(gpu_ids)
        encoder = nn.DataParallel(encoder, device_ids=gpu_ids)
        predictor = nn.DataParallel(predictor, device_ids=gpu_ids)
        
    # encoder.to(device)
    # predictor.to(device)
    encoder.cuda()
    predictor.cuda()

    nll = neg_log_likely_hood()
    mse = torch.nn.MSELoss()

    batch_size = constant.BATCH_SIZE[gpu_cnt-1]
    start_time = time.ctime()
    for epoch in range(constant.NUM_EPOCHS):

        train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        # adjust_learning_rate(optimizer, epoch)

        for i, (inputs, labels) in enumerate(train_loader):  
            # inputs = inputs.to(device)
            inputs = inputs.cuda()
            # labels = labels.to(device)
            labels = labels.cuda()
            input_shape = inputs.shape

            encoder_optimizer.zero_grad()
            predictor_optimizer.zero_grad()

            assert len(input_shape) == 6
            
            inputs = inputs.view(input_shape[0]*input_shape[1], input_shape[2], input_shape[3], input_shape[4], input_shape[5])

            representations = encoder(inputs).view(input_shape[0], input_shape[1], -1)           
            gmm_params = predictor(representations)
            predictions = gmm_to_pdf(*gmm_params)

            targets = labels.view(labels.shape[0]*labels.shape[1], labels.shape[2], labels.shape[3])
            
            if (i+1) % 50 == 0:
                for j in range(0, constant.PREDICTOR_SEQ_LEN):
                    visualize.save_matrix_to_image(-targets[j,:,:].cpu().detach().numpy(), 'visua_res/epoch-%d-step-%d-%d-target.jpg'%(epoch+1,i,j))
                    visualize.save_matrix_to_image(-predictions[j,:,:].cpu().detach().numpy(), 'visua_res/epoch-%d-step-%d-%d-result.jpg'%(epoch+1,i,j))

            loss = nll(predictions, targets)
            loss = torch.mean(loss)
            loss.backward()

            nn.utils.clip_grad_norm_(encoder.parameters(), 1.)
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.)
            
            encoder_optimizer.step()
            predictor_optimizer.step()

            if (i+1) % 5 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.6f' 
                    %(epoch+1, constant.NUM_EPOCHS, i+1, len(my_dataset)//batch_size, loss.data))

        my_dataset.refresh_pool()

        if args.multi_gpu:
            torch.save(encoder.module.state_dict(), 'encoder.pth')
            torch.save(predictor.module.state_dict(), 'predictor.pth')
        else:
            torch.save(encoder.state_dict(), 'encoder.pth')
            torch.save(predictor.state_dict(), 'predictor.pth')
        print('Save model')
    print("finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids")
    parser.add_argument("--multi_gpu", type=str2bool, nargs='?',
                        const=True, default='True')
    args = parser.parse_args()
    train(args)
