import os
import os.path as osp

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data

import argparse
import numpy as np
from packaging import version
import wandb
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from model.refinenetlw import rf_lw101
from compute_iou import compute_mIoU
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.Foggy_Zurich_test import foggyzurichDataSet
from dataset.foggy_driving import foggydrivingDataSet
from dataset.paired_cityscapes import Pairedcityscapes
from dataset.foggy_cityscapes_dataset import foggy_cityscapesDataSet


import time 

def log(message, timestr): # 나의 편의를 위해 log 추가
    with open('./log/'+ str(timestr) +'_log.txt', 'a+') as logger:
        logger.write(f'{message}\n')

timestr = time.strftime("%m%d-%H%M")

#SAVED_MODEL_PATH = './saved_weights_final/Final_stage_4.pth'
SAVED_MODEL_PATH = './weights/weights_stage4/Stage4_Final.pth'
NUM_CLASSES = 19

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def eval():
    """Create the model and start the evaluation process."""

    model = rf_lw101(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    
    save_dir_fz = osp.join(f'./result_FZ', 'FIFO_model_Origin')
    save_dir_fd = osp.join(f'./result_FD', 'FIFO_model_Origin')
    save_dir_fdd = osp.join(f'./result_FDD', 'FIFO_model_Origin')
    save_dir_clindau = osp.join(f'./result_Clindau', 'FIFO_model_Origin')
    save_dir_foggy = osp.join(f'./result_foggycity', 'FIFO_model_Origin')  

    if not os.path.exists(save_dir_fz):
        os.makedirs(save_dir_fz)
    if not os.path.exists(save_dir_fd):
        os.makedirs(save_dir_fd)
    if not os.path.exists(save_dir_fdd):
        os.makedirs(save_dir_fdd)
    if not os.path.exists(save_dir_clindau):
        os.makedirs(save_dir_clindau)
    if not os.path.exists(save_dir_foggy):
        os.makedirs(save_dir_foggy)
    
    model.eval()
    device = torch.device('cuda:0')
    model.to(device)


    testloader1 = data.DataLoader(foggyzurichDataSet("./data/Foggy_Zurich", "./data/Foggy_Zurich/Foggy_Zurich/lists_file_names/RGB_testv2_filenames.txt", crop_size=(1152, 648), mean=IMG_MEAN),
                                    batch_size=1, shuffle=False, pin_memory=True)
    testloader2 = data.DataLoader(foggyzurichDataSet("./data/Foggy_Zurich", "./data/Foggy_Zurich/Foggy_Zurich/lists_file_names/RGB_testv2_filenames.txt", crop_size=(1536, 864), mean=IMG_MEAN),
                                    batch_size=1, shuffle=False, pin_memory=True)
    testloader3 = data.DataLoader(foggyzurichDataSet("./data/Foggy_Zurich", "./data/Foggy_Zurich/Foggy_Zurich/lists_file_names/RGB_testv2_filenames.txt", crop_size=(1920, 1080), mean=IMG_MEAN),
                                    batch_size=1, shuffle=False, pin_memory=True)

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp_eval = nn.Upsample(size=(1080,1920), mode='bilinear', align_corners=True)
    else:
        interp_eval = nn.Upsample(size=(1080,1920), mode='bilinear')

    testloader_iter2 = enumerate(testloader2)
    testloader_iter3 = enumerate(testloader3)


    for index, batch1 in enumerate(testloader1):
        image, label_test, _, name = batch1
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(0))
            output_1 = interp_eval(output2)

        _, batch2 = testloader_iter2.__next__()
        image, label_test, _, name = batch2
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(0))
            output_2 = interp_eval(output2)

        _, batch3 = testloader_iter3.__next__()    
        image, label_test, _, name = batch3
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(0))
            output_3 = interp_eval(output2)

        output = torch.cat([output_1,output_2,output_3])
        output = torch.mean(output, dim=0)
        output = output.cpu().numpy()
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (save_dir_fz, name))
        output_col.save('%s/%s_color.png' % (save_dir_fz, name[:-4]))
    miou_fz = compute_mIoU("./data/Foggy_Zurich/Foggy_Zurich", save_dir_fz, "./data/Foggy_Zurich/Foggy_Zurich/lists_file_names", 'FZ')
    log(f"Foggy Zurich MIOU {miou_fz}", timestr)


    testloader1 = data.DataLoader(foggydrivingDataSet("./data/Foggy_Driving/Foggy_Driving", './lists_file_names/leftImg8bit_testdense_filenames.txt' , scale=1),
                                    batch_size=1, shuffle=False, pin_memory=True)

    testloader2 = data.DataLoader(foggydrivingDataSet("./data/Foggy_Driving/Foggy_Driving", './lists_file_names/leftImg8bit_testdense_filenames.txt' , scale=0.8),
                                    batch_size=1, shuffle=False, pin_memory=True) 

    testloader3 = data.DataLoader(foggydrivingDataSet("./data/Foggy_Driving/Foggy_Driving", './lists_file_names/leftImg8bit_testdense_filenames.txt' , scale=0.6),
                                    batch_size=1, shuffle=False, pin_memory=True)
    testloader_iter2 = enumerate(testloader2)
    testloader_iter3 = enumerate(testloader3)

    for index, batch in enumerate(testloader1):
        image, size, name = batch
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(0))
            interp_eval = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')
            output_1 = interp_eval(output2)

        _, batch2 = testloader_iter2.__next__()
        image, _, name = batch2
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(0))
            output_2 = interp_eval(output2)

        _, batch3 = testloader_iter3.__next__()    
        image, _, name = batch3
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(0))
            output_3 = interp_eval(output2)

        output = torch.cat([output_1,output_2,output_3])
        output = torch.mean(output, dim=0)
        output = output.cpu().numpy()
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (save_dir_fdd, name))
        output_col.save('%s/%s_color.png' % (save_dir_fdd, name[:-4]))
    miou_fdd = compute_mIoU("./data/Foggy_Driving/Foggy_Driving", save_dir_fdd, './lists_file_names', 'FDD')
    log(f"Foggy Driving Dense MIOU {miou_fdd}", timestr)

    testloader1 = data.DataLoader(foggydrivingDataSet("./data/Foggy_Driving/Foggy_Driving", './lists_file_names/leftImg8bit_testall_filenames.txt', scale=1),
                                    batch_size=1, shuffle=False, pin_memory=True) 

    testloader2 = data.DataLoader(foggydrivingDataSet("./data/Foggy_Driving/Foggy_Driving", './lists_file_names/leftImg8bit_testall_filenames.txt', scale=0.8),
                                    batch_size=1, shuffle=False, pin_memory=True) 

    testloader3 = data.DataLoader(foggydrivingDataSet("./data/Foggy_Driving/Foggy_Driving", './lists_file_names/leftImg8bit_testall_filenames.txt', scale=0.6),
                                    batch_size=1, shuffle=False, pin_memory=True) 
    testloader_iter2 = enumerate(testloader2)
    testloader_iter3 = enumerate(testloader3)

    for index, batch in enumerate(testloader1):
        image, size, name = batch
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(0))
            interp_eval = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')

            output_1 = interp_eval(output2)

        _, batch2 = testloader_iter2.__next__()
        image, _, name = batch2
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(0))
            output_2 = interp_eval(output2)

        _, batch3 = testloader_iter3.__next__()    
        image, _, name = batch3
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(0))
            output_3 = interp_eval(output2)

        output = torch.cat([output_1,output_2,output_3])
        output = torch.mean(output, dim=0)
        output = output.cpu().numpy()
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (save_dir_fd, name))
        output_col.save('%s/%s_color.png' % (save_dir_fd, name[:-4]))
    miou_fd = compute_mIoU("./data/Foggy_Driving/Foggy_Driving", save_dir_fd, './lists_file_names', 'FD')
    log(f"Foggy Driving light MIOU {miou_fd}", timestr)

    testloader1 = data.DataLoader(cityscapesDataSet("./data/Cityscape", './dataset/cityscapes_list/val.txt', crop_size = (2048, 1024), mean=IMG_MEAN, scale=False, mirror=False, set='val'),
                            batch_size=1, shuffle=False, pin_memory=True)
    testloader2 = data.DataLoader(cityscapesDataSet("./data/Cityscape", './dataset/cityscapes_list/val.txt', crop_size = (2048*0.8, 1024*0.8), mean=IMG_MEAN, scale=False, mirror=False, set='val'),
                            batch_size=1, shuffle=False, pin_memory=True)
    testloader3 = data.DataLoader(cityscapesDataSet("./data/Cityscape", './dataset/cityscapes_list/val.txt', crop_size = (2048*0.6, 1024*0.6), mean=IMG_MEAN, scale=False, mirror=False, set='val'),
                            batch_size=1, shuffle=False, pin_memory=True)   
    testloader_iter2 = enumerate(testloader2)
    testloader_iter3 = enumerate(testloader3)


    for index, batch in enumerate(testloader1):
        image, size, name = batch
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(0))
            interp_eval = nn.Upsample(size=(1024, 2048), mode='bilinear')
            output_1 = interp_eval(output2)

        _, batch2 = testloader_iter2.__next__()
        image, _, name = batch2
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(0))
            output_2 = interp_eval(output2)

        _, batch3 = testloader_iter3.__next__()    
        image, _, name = batch3
        with torch.no_grad():
            output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(0))
            output_3 = interp_eval(output2)

        output = torch.cat([output_1,output_2,output_3])
        output = torch.mean(output, dim=0)
        output = output.cpu().numpy()
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (save_dir_clindau, name))
        output_col.save('%s/%s_color.png' % (save_dir_clindau, name.split('.')[0]))

    miou_clindau = compute_mIoU("./data/Cityscape/gtFine", save_dir_clindau, './dataset/cityscapes_list', 'Clindau')
    log(f"Cityscape Clean MIOU {miou_clindau}", timestr)




if __name__ == '__main__':
    eval()
