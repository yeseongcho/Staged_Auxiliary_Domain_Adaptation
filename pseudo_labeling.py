import os
import os.path as osp

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import grad 

import numpy as np
import random

from tqdm import tqdm
from PIL import Image
from packaging import version
from datetime import datetime

from model.refinenetlw import rf_lw101
from model.discriminator import FCDiscriminator
from model.discriminator import OutspaceDiscriminator
from utils.losses import CrossEntropy2d
from dataset.paired_cityscapes_CS_FS import Pairedcityscapes
from dataset.Foggy_Zurich_train import foggyzurichDataSet
from utils.optimisers import get_optimisers, get_lr_schedulers

from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer
from sklearn.metrics import pairwise_distances
import inspect
from torch.nn.parallel import DataParallel
from tqdm.notebook import tqdm
from utils.prototype_dist_estimator import prototype_dist_estimator
from utils.pcl_loss import PrototypeContrastiveLoss

import time
import shutil

def log(message, timestr): # 나의 편의를 위해 log 추가
    with open('./log/'+ str(timestr) +'_log.txt', 'a+') as logger:
        logger.write(f'{message}\n')

timestr = time.strftime("%m%d-%H%M")

def loss_calc(pred, label, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)
    return criterion(pred, label)

def setup_optimisers_and_schedulers(model):
    optimisers = get_optimisers(
        model=model,
        enc_optim_type="sgd",
        enc_lr=6e-4,
        #enc_lr=1e-3,
        enc_weight_decay=1e-5,
        enc_momentum=0.9,
        dec_optim_type="sgd",
        dec_lr=6e-3,
        dec_weight_decay=1e-5,
        dec_momentum=0.9,
    )
    schedulers = get_lr_schedulers(
        enc_optim=optimisers[0],
        dec_optim=optimisers[1],
        enc_lr_gamma=0.5,
        dec_lr_gamma=0.5,
        enc_scheduler_type="multistep",
        dec_scheduler_type="multistep",
        epochs_per_stage=(100, 100, 100),
    )
    return optimisers, schedulers


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
    new_mask.putpalette(palette)
    return new_mask

def evaluate(model, log, timestr) : 
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
    #import wandb
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

    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

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

    '''
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
    '''
    return

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(LEARNING_RATE, i_iter, NUM_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(LEARNING_RATE_D, i_iter, NUM_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_split_list(list_name):
    image_list = []
    image_name_list = []
    file_num = 0
    with open(list_name) as f:
        for item in f.readlines():
            fields = item.strip()
            image_name = fields.split('/')[-1]
            image_list.append(fields)
            image_name_list.append(image_name)
            file_num += 1
    return image_list, image_name_list, file_num

RANDOM_SEED = 1234
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
NUM_STEPS = 300000
POWER = 0.9
NUM_CLASSES = 19

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

now = datetime.now().strftime('%m-%d-%H-%M')

cudnn.enabled = True
gpu = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_iter = 0
### Segmentation model load (RefineNet-lw)
model = rf_lw101(num_classes=NUM_CLASSES)

#re = torch.load("./weights_stage3/Stage3_80000.pth")
re = torch.load("C:/Users/user/Desktop/personalstudy/Smooth_and_Clear/saved_weights_final/Final_stage_3.pth")
#re = torch.load("./saved_weights_final/Final_stage_3.pth")
log(f"Success Load",timestr)
model.load_state_dict(re)

from dataset.Foggy_Zurich_labeling import foggyzurichDataSet
tgt_root = "./data/Foggy_Zurich"

## Labeling for Clean Target
ct_list_path = "./data/Foggy_Zurich/Foggy_Zurich/lists_file_names/RGB_Clean_Target.txt"
rf_loader_val_cwz = data.DataLoader(foggyzurichDataSet(tgt_root, ct_list_path,crop_size=(1920, 1080),
                                            mean=IMG_MEAN),
                                            batch_size=1, shuffle=False, num_workers=0) ## Shuffle이 training에 영향을 줄 수 있다.

## Labeling for Fog Target
ft_list_path = "./data/Foggy_Zurich/Foggy_Zurich/lists_file_names/RGB_Fog_Target.txt"
rf_loader_val_fz = data.DataLoader(foggyzurichDataSet(tgt_root, ft_list_path,crop_size=(1920, 1080),
                                            mean=IMG_MEAN),
                                            batch_size=1, shuffle=False, num_workers=0) ## Shuffle이 training에 영향을 줄 수 있다. 


## Pseudo-Labelin Round, we only one round! (No any other additional round)
round_idx = 0


#### (1) Clean Target Pseudo-Labeling
SAVE_PATH_Clean = "./pseudo_label_clean"
save_path = SAVE_PATH_Clean
save_pseudo_label_path = osp.join(save_path, 'pseudo_label')  
save_stats_path = osp.join(save_path, 'stats') 
save_lst_path = osp.join(save_path, 'list')

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_pseudo_label_path):
    os.makedirs(save_pseudo_label_path)
if not os.path.exists(save_stats_path):
    os.makedirs(save_stats_path)
if not os.path.exists(save_lst_path):
    os.makedirs(save_lst_path)

save_round_eval_path = osp.join(SAVE_PATH_Clean,str(round_idx))
save_pseudo_label_color_path = osp.join(save_round_eval_path, 'pseudo_label_color')

if not os.path.exists(save_round_eval_path):
    os.makedirs(save_round_eval_path)
if not os.path.exists(save_pseudo_label_color_path):
    os.makedirs(save_pseudo_label_color_path)

## upsampling layer
interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)
## output of deeplab is logits, not probability
softmax2d = nn.Softmax2d()
## output folder
save_pred_vis_path = osp.join(save_round_eval_path, 'pred_vis')
save_prob_path = osp.join(save_round_eval_path, 'prob')
save_pred_path = osp.join(save_round_eval_path, 'pred')
if not os.path.exists(save_pred_vis_path):
    os.makedirs(save_pred_vis_path)
if not os.path.exists(save_prob_path):
    os.makedirs(save_prob_path)
if not os.path.exists(save_pred_path):
    os.makedirs(save_pred_path)

model.eval()
model.to(device)

## upsampling layer
interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)

## output of deeplab is logits, not probability
softmax2d = nn.Softmax2d()

## output folder
save_pred_vis_path = osp.join(save_round_eval_path, 'pred_vis')
save_prob_path = osp.join(save_round_eval_path, 'prob')
save_pred_path = osp.join(save_round_eval_path, 'pred')

if not os.path.exists(save_pred_vis_path):
    os.makedirs(save_pred_vis_path)
if not os.path.exists(save_prob_path):
    os.makedirs(save_prob_path)
if not os.path.exists(save_pred_path):
    os.makedirs(save_pred_path)
        
# saving output data, confidence score
conf_dict = {k: [] for k in range(19)}
pred_cls_num = np.zeros(19)
## evaluation process
with torch.no_grad():
    for index, batch in tqdm(enumerate(rf_loader_val_cwz)):
        image, size, name  = batch
        feature_rf0,feature_rf1,feature_rf2, feature_rf3,feature_rf4,feature_rf5 = model(image.to(device))
        output = softmax2d(interp(feature_rf5)).cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        amax_output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        conf = np.amax(output,axis=2)
        # score
        pred_label = amax_output.copy()

        # save visualized seg maps & predication prob map
        amax_output_col = colorize_mask(amax_output)
        name = name[0].split('/')[-1]
        image_name = name[:-4] 
        # prob
        np.save('%s/%s.npy' % (save_prob_path, image_name), output)
        # trainIDs/vis seg maps
        amax_output = Image.fromarray(amax_output)
        amax_output.save('%s/%s.png' % (save_pred_path, image_name))
        amax_output_col.save('%s/%s_color.png' % (save_pred_vis_path, image_name))

        # save class-wise confidence maps
        for idx_cls in range(19):
            idx_temp = pred_label == idx_cls
            pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
            if idx_temp.any():
                conf_cls_temp = conf[idx_temp].astype(np.float32)
                len_cls_temp = conf_cls_temp.size
                # downsampling by ds_rate
                conf_cls = conf_cls_temp[0:len_cls_temp:4]
                conf_dict[idx_cls].extend(conf_cls)

image_src_list, _, src_num = parse_split_list("./dataset/cityscapes_list/train_pseudolabel.txt")
image_tgt_list, image_name_tgt_list, tgt_num = parse_split_list("./data/Foggy_Zurich/Foggy_Zurich/lists_file_names/RGB_Clean_Target.txt")

cls_thresh = np.ones(19,dtype = np.float32)
cls_sel_size = np.zeros(19, dtype=np.float32)
cls_size = np.zeros(19, dtype=np.float32)

tgt_portion = 0.5 ## We use pseudo-labels, which are almost top 50% confidence scores by each classes

import math
for idx_cls in tqdm(np.arange(0, 19)):
    cls_size[idx_cls] = pred_cls_num[idx_cls]
    if conf_dict[idx_cls] != None:
        conf_dict[idx_cls].sort(reverse=True) # sort in descending order
        len_cls = len(conf_dict[idx_cls])
        cls_sel_size[idx_cls] = int(math.floor(len_cls * tgt_portion))
        len_cls_thresh = int(cls_sel_size[idx_cls])
        if len_cls_thresh != 0:
            if conf_dict[idx_cls][len_cls_thresh-1]<0.9:
                cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh-1]
            else:
                cls_thresh[idx_cls] = 0.9
        conf_dict[idx_cls] = None
        
np.save(save_stats_path + '/cls_thresh_round' + str(round_idx) + '.npy', cls_thresh)
np.save(save_stats_path + '/cls_sel_size_round' + str(round_idx) + '.npy', cls_sel_size)
cls_thresh = np.load(save_stats_path + '/cls_thresh_round' + str(round_idx) + '.npy')

## Final Labeling
for idx in tqdm(range(tgt_num)):
    sample_name = image_name_tgt_list[idx][:-4]
    probmap_path = osp.join(save_prob_path, '{}.npy'.format(sample_name))
    pred_path = osp.join(save_pred_path, '{}.png'.format(sample_name))
    pred_prob = np.load(probmap_path)
    pred_label_trainIDs = np.asarray(Image.open(pred_path))
    save_wpred_vis_path = osp.join(save_round_eval_path, 'weighted_pred_vis')
    if not os.path.exists(save_wpred_vis_path):
        os.makedirs(save_wpred_vis_path)
    weighted_prob = pred_prob/cls_thresh
    weighted_pred_trainIDs = np.asarray(np.argmax(weighted_prob, axis=2), dtype=np.uint8)
    # save weighted predication
    wpred_label_col = weighted_pred_trainIDs.copy()
    wpred_label_col = colorize_mask(wpred_label_col)
    wpred_label_col.save('%s/%s_color.png' % (save_wpred_vis_path, sample_name))
    weighted_conf = np.amax(weighted_prob, axis=2)
    pred_label_trainIDs = weighted_pred_trainIDs.copy()
    pred_label_trainIDs[weighted_conf < 1] = 255 # '255' in cityscapes indicates 'unlabaled' for trainIDs

    # pseudo-labels with labelID
    pseudo_label_trainIDs = pred_label_trainIDs.copy()
    # save colored pseudo-label map
    pseudo_label_col = colorize_mask(pseudo_label_trainIDs)
    pseudo_label_col.save('%s/%s_color.png' % (save_pseudo_label_color_path, sample_name))
    # save pseudo-label map with label IDs
    pseudo_label_save = Image.fromarray(pseudo_label_trainIDs.astype(np.uint8))
    pseudo_label_save.save('%s/%s.png' % (save_pseudo_label_path, sample_name))

src_train_lst = osp.join(save_lst_path,'src_train.txt')
tgt_train_lst = osp.join(save_lst_path, 'tgt_train.txt')

# generate src train list
with open(src_train_lst, 'w') as f:
    for idx in range(src_num):
        f.write("%s\n" % (image_src_list[idx]))
# generate tgt train list
with open(tgt_train_lst, 'w') as f:
    for idx in range(tgt_num):
        image_tgt_path = osp.join(save_pseudo_label_path,image_name_tgt_list[idx])
        f.write("%s\t%s\n" % (image_tgt_list[idx], image_tgt_path))
        
shutil.rmtree(save_prob_path)


#### (2) Fog Target Pseudo-Labeling
SAVE_PATH_Foggy = "./pseudo_label_foggy"
save_path = SAVE_PATH_Foggy
save_pseudo_label_path = osp.join(save_path, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
save_stats_path = osp.join(save_path, 'stats') # in 'save_path'
save_lst_path = osp.join(save_path, 'list')

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_pseudo_label_path):
    os.makedirs(save_pseudo_label_path)
if not os.path.exists(save_stats_path):
    os.makedirs(save_stats_path)
if not os.path.exists(save_lst_path):
    os.makedirs(save_lst_path)

save_round_eval_path = osp.join(SAVE_PATH_Foggy,str(round_idx))
save_pseudo_label_color_path = osp.join(save_round_eval_path, 'pseudo_label_color')

if not os.path.exists(save_round_eval_path):
    os.makedirs(save_round_eval_path)
if not os.path.exists(save_pseudo_label_color_path):
    os.makedirs(save_pseudo_label_color_path)

## upsampling layer
interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)
## output of deeplab is logits, not probability
softmax2d = nn.Softmax2d()
## output folder
save_pred_vis_path = osp.join(save_round_eval_path, 'pred_vis')
save_prob_path = osp.join(save_round_eval_path, 'prob')
save_pred_path = osp.join(save_round_eval_path, 'pred')

if not os.path.exists(save_pred_vis_path):
    os.makedirs(save_pred_vis_path)
if not os.path.exists(save_prob_path):
    os.makedirs(save_prob_path)
if not os.path.exists(save_pred_path):
    os.makedirs(save_pred_path)

model.eval()
model.to(device)

## upsampling layer
interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)

## output of deeplab is logits, not probability
softmax2d = nn.Softmax2d()

## output folder
save_pred_vis_path = osp.join(save_round_eval_path, 'pred_vis')
save_prob_path = osp.join(save_round_eval_path, 'prob')
save_pred_path = osp.join(save_round_eval_path, 'pred')

if not os.path.exists(save_pred_vis_path):
    os.makedirs(save_pred_vis_path)
if not os.path.exists(save_prob_path):
    os.makedirs(save_prob_path)
if not os.path.exists(save_pred_path):
    os.makedirs(save_pred_path)
        
# saving output data
conf_dict = {k: [] for k in range(19)}
pred_cls_num = np.zeros(19)
## evaluation process
with torch.no_grad():
    for index, batch in tqdm(enumerate(rf_loader_val_fz)):
        image, size, name = batch
        if image == [] :
            print(name)
            continue
        feature_rf0,feature_rf1,feature_rf2, feature_rf3,feature_rf4,feature_rf5 = model(image.to(device))
        output = softmax2d(interp(feature_rf5)).cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        amax_output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        conf = np.amax(output,axis=2)
        # score
        pred_label = amax_output.copy()

        # save visualized seg maps & predication prob map
        amax_output_col = colorize_mask(amax_output)
        name = name[0].split('/')[-1]
        image_name = name[:-4] ## .png만 빼기 위해
        # prob
        np.save('%s/%s.npy' % (save_prob_path, image_name), output)
        # trainIDs/vis seg maps
        amax_output = Image.fromarray(amax_output)
        amax_output.save('%s/%s.png' % (save_pred_path, image_name))
        amax_output_col.save('%s/%s_color.png' % (save_pred_vis_path, image_name))

        # save class-wise confidence maps
        for idx_cls in range(19):
            idx_temp = pred_label == idx_cls
            pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
            if idx_temp.any():
                conf_cls_temp = conf[idx_temp].astype(np.float32)
                len_cls_temp = conf_cls_temp.size
                # downsampling by ds_rate
                conf_cls = conf_cls_temp[0:len_cls_temp:4]
                conf_dict[idx_cls].extend(conf_cls)

image_src_list, _, src_num = parse_split_list("./dataset/cityscapes_list/train_pseudolabel.txt")
image_tgt_list, image_name_tgt_list, tgt_num = parse_split_list("./data/Foggy_Zurich/Foggy_Zurich/lists_file_names/RGB_Fog_Target.txt")

cls_thresh = np.ones(19,dtype = np.float32)
cls_sel_size = np.zeros(19, dtype=np.float32)
cls_size = np.zeros(19, dtype=np.float32)

tgt_portion = 0.5

import math
for idx_cls in tqdm(np.arange(0, 19)):
    cls_size[idx_cls] = pred_cls_num[idx_cls]
    if conf_dict[idx_cls] != None:
        conf_dict[idx_cls].sort(reverse=True) # sort in descending order
        len_cls = len(conf_dict[idx_cls])
        cls_sel_size[idx_cls] = int(math.floor(len_cls * tgt_portion))
        len_cls_thresh = int(cls_sel_size[idx_cls])
        if len_cls_thresh != 0:
            if conf_dict[idx_cls][len_cls_thresh-1]<0.9:
                cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh-1]
            else:
                cls_thresh[idx_cls] = 0.9
        conf_dict[idx_cls] = None
        
np.save(save_stats_path + '/cls_thresh_round' + str(round_idx) + '.npy', cls_thresh)
np.save(save_stats_path + '/cls_sel_size_round' + str(round_idx) + '.npy', cls_sel_size)
cls_thresh = np.load(save_stats_path + '/cls_thresh_round' + str(round_idx) + '.npy')

for idx in tqdm(range(tgt_num)):
    sample_name = image_name_tgt_list[idx][:-4]
    probmap_path = osp.join(save_prob_path, '{}.npy'.format(sample_name))
    pred_path = osp.join(save_pred_path, '{}.png'.format(sample_name))
    try : 
        pred_prob = np.load(probmap_path)
    except :
        continue
    pred_label_trainIDs = np.asarray(Image.open(pred_path))
    save_wpred_vis_path = osp.join(save_round_eval_path, 'weighted_pred_vis')
    if not os.path.exists(save_wpred_vis_path):
        os.makedirs(save_wpred_vis_path)
    weighted_prob = pred_prob/cls_thresh
    weighted_pred_trainIDs = np.asarray(np.argmax(weighted_prob, axis=2), dtype=np.uint8)
    # save weighted predication
    wpred_label_col = weighted_pred_trainIDs.copy()
    wpred_label_col = colorize_mask(wpred_label_col)
    wpred_label_col.save('%s/%s_color.png' % (save_wpred_vis_path, sample_name))
    weighted_conf = np.amax(weighted_prob, axis=2)
    pred_label_trainIDs = weighted_pred_trainIDs.copy()
    pred_label_trainIDs[weighted_conf < 1] = 255 # '255' in cityscapes indicates 'unlabaled' for trainIDs

    # pseudo-labels with labelID
    pseudo_label_trainIDs = pred_label_trainIDs.copy()
    # save colored pseudo-label map
    pseudo_label_col = colorize_mask(pseudo_label_trainIDs)
    pseudo_label_col.save('%s/%s_color.png' % (save_pseudo_label_color_path, sample_name))
    # save pseudo-label map with label IDs
    pseudo_label_save = Image.fromarray(pseudo_label_trainIDs.astype(np.uint8))
    pseudo_label_save.save('%s/%s.png' % (save_pseudo_label_path, sample_name))

src_train_lst = osp.join(save_lst_path,'src_train.txt')
tgt_train_lst = osp.join(save_lst_path, 'tgt_train.txt')

# generate src train list
with open(src_train_lst, 'w') as f:
    for idx in range(src_num):
        f.write("%s\n" % (image_src_list[idx]))
# generate tgt train list
with open(tgt_train_lst, 'w') as f:
    for idx in range(tgt_num):
        image_tgt_path = osp.join(save_pseudo_label_path,image_name_tgt_list[idx])
        f.write("%s\t%s\n" % (image_tgt_list[idx], image_tgt_path))


shutil.rmtree(save_prob_path)












