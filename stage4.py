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

SAVE_PRED_EVERY = 100
SNAPSHOT_DIR = "./weights/weights_stage4"
NUM_STEPS_STOP = 30000

now = datetime.now().strftime('%m-%d-%H-%M')

cudnn.enabled = True
gpu = 0

start_iter = 0
### Segmentation model load (RefineNet-lw)
model = rf_lw101(num_classes=NUM_CLASSES)

re = torch.load("./weights/weights_stage3/Stage3_Final.pth")
#re = torch.load("./saved_weights_final/Final_stage_3.pth")
log(f"Success Load",timestr)
model.load_state_dict(re)

# init D
num_class_list = [2048, 19] 
model_CS_CT_D = nn.ModuleList([FCDiscriminator(num_classes=num_class_list[i]).train().cuda(0) if i<1 else OutspaceDiscriminator(num_classes=num_class_list[i]).train().cuda(0) for i in range(2)])
model_CS_CT_D.load_state_dict(torch.load("./weights/weights_stage3/Stage3_Final_CS_CT_D.pth"))
#model_CS_CT_D.load_state_dict(torch.load("./saved_weights_final/Final_stage_3_CS_CT_D.pth"))
optimizer_CS_CT_D = optim.Adam(model_CS_CT_D.parameters(), lr=1e-4, betas=(0.9, 0.99))
optimizer_CS_CT_D.zero_grad()

model_FS_FT_D = nn.ModuleList([FCDiscriminator(num_classes=num_class_list[i]).train().cuda(0) if i<1 else OutspaceDiscriminator(num_classes=num_class_list[i]).train().cuda(0) for i in range(2)])
model_FS_FT_D.load_state_dict(torch.load("./weights/weights_stage3/Stage3_Final_FS_FT_D.pth"))
#model_FS_FT_D.load_state_dict(torch.load("./saved_weights_final/Final_stage_3_FS_FT_D.pth"))
optimizer_FS_FT_D = optim.Adam(model_FS_FT_D.parameters(), lr=1e-4, betas=(0.9, 0.99))
optimizer_FS_FT_D.zero_grad()

model_CT_FT_D = nn.ModuleList([FCDiscriminator(num_classes=num_class_list[i]).train().cuda(0) if i<1 else OutspaceDiscriminator(num_classes=num_class_list[i]).train().cuda(0) for i in range(2)])
model_FS_FT_D.load_state_dict(torch.load("./weights/weights_stage3/Stage3_Final_CT_FT_D.pth"))
#model_CT_FT_D.load_state_dict(torch.load("./saved_weights_final/Final_stage_3_CT_FT_D.pth"))
optimizer_CT_FT_D = optim.Adam(model_CT_FT_D.parameters(), lr=1e-4, betas=(0.9, 0.99))
optimizer_CT_FT_D.zero_grad()

bce_loss = torch.nn.MSELoss()
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cs_root = "./data/Cityscape"
fs_root = "./data/Foggy_Cityscape"
cs_list_path = './dataset/cityscapes_list/train_origin.txt'
fs_list_path = f'./dataset/cityscapes_list/train_foggy_{0.005}.txt'
max_iters = 100000 * 1 * 4

CS_FS_pair_loader = data.DataLoader(Pairedcityscapes(fs_root, cs_root, fs_list_path, cs_list_path,
                            max_iters=max_iters,
                            mean=IMG_MEAN, set='train'), batch_size=2, shuffle=True, num_workers=0,
                            pin_memory=True,drop_last=True)

fs_image, cs_image,label, size, _, _, fs_arry, cs_arry = next(iter(CS_FS_pair_loader))

from dataset.Foggy_Zurich_pseudo_label import foggyzurichDataSet

### Target Domains with Labels
save_pseudo_label_path_CT = "./pseudo_label_clean/pseudo_label"
save_pseudo_label_path_FT = "./pseudo_label_foggy/pseudo_label"

max_iters = 100000 * 1 * 2
tgt_root = "./data/Foggy_Zurich"
ct_list_path = "./data/Foggy_Zurich/Foggy_Zurich/lists_file_names/RGB_Clean_Target.txt"

CT_loader_ST = data.DataLoader(foggyzurichDataSet(tgt_root, ct_list_path,save_pseudo_label_path_CT,
                                            mean=IMG_MEAN),
                                            batch_size=2, shuffle=True, num_workers=0,
                                            pin_memory=True)
img_ct, label_tgt, size_ct, name_ct = next(iter(CT_loader_ST))

ft_list_path = "./data/Foggy_Zurich/Foggy_Zurich/lists_file_names/RGB_Fog_Target.txt"

FT_loader_ST = data.DataLoader(foggyzurichDataSet(tgt_root, ft_list_path,save_pseudo_label_path_FT,
                                            mean=IMG_MEAN),
                                            batch_size=2, shuffle=True, num_workers=0,
                                            pin_memory=True)
img, label_tgt, size, name = next(iter(FT_loader_ST))

model.train()
model.to(device)

optimisers, schedulers = setup_optimisers_and_schedulers( model=model)
opts = make_list(optimisers)

source_label = 0
target_label = 1

kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
m = nn.Softmax(dim=1)
log_m = nn.LogSoftmax(dim=1)

## Redefining Centroids (Prototypes)
feat_estimator_CS_CT = prototype_dist_estimator(feature_num=2048)
out_estimator_CS_CT = prototype_dist_estimator(feature_num=19)

feat_estimator_FS_FT = prototype_dist_estimator(feature_num=2048)
out_estimator_FS_FT = prototype_dist_estimator(feature_num=19)

feat_estimator_CT_FT = prototype_dist_estimator(feature_num=2048)
out_estimator_CT_FT = prototype_dist_estimator(feature_num=19)

pcl_criterion = PrototypeContrastiveLoss()


for i_iter in tqdm(range(NUM_STEPS)):

    if i_iter == 0 : 
        evaluate(model,log,timestr)
    
    model.train()

    for opt in opts:
        opt.zero_grad()
        adjust_learning_rate(opt, i_iter)

    optimizer_CS_CT_D.zero_grad()
    adjust_learning_rate_D(optimizer_CS_CT_D, i_iter)
    
    optimizer_FS_FT_D.zero_grad()
    adjust_learning_rate_D(optimizer_FS_FT_D, i_iter)

    optimizer_CT_FT_D.zero_grad()
    adjust_learning_rate_D(optimizer_CT_FT_D, i_iter)

    for param in model_CS_CT_D.parameters():
        param.requires_grad = False

    for param in model_FS_FT_D.parameters():
        param.requires_grad = False

    for param in model_CT_FT_D.parameters():
        param.requires_grad = False
        
    if i_iter % 3 == 0 : 

        batch = next(iter(CS_FS_pair_loader))
        fs_image, cs_image,label, size, _, _, fs_arry, cs_arry = batch
        interp = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')
        cs_image = cs_image.to(device)
        label = label.long().to(device)

        images = Variable(cs_image).cuda()
        feature_cs0,feature_cs1,feature_cs2, feature_cs3,feature_cs4,feature_cs5 = model(images)
        pred_cs5 = interp(feature_cs5)
        loss_seg = loss_calc(pred_cs5, label,0)
        #loss_seg.backward()

        images2 = Variable(fs_image).cuda()
        feature_fs0,feature_fs1,feature_fs2, feature_fs3,feature_fs4,feature_fs5 = model(images2)
        pred_fs5 = interp(feature_fs5)
        loss_seg2 = loss_calc(pred_fs5, label,0)

        feature_cs5_logsoftmax = log_m(feature_cs5)
        feature_fs5_softmax = m(feature_fs5)
        feature_fs5_logsoftmax = log_m(feature_fs5)
        feature_cs5_softmax = m(feature_cs5)

        loss_con = kl_loss(feature_fs5_logsoftmax, feature_cs5_softmax)
        
        try : 
            batch_ct = next(iter(CT_loader_ST))
            img_ct, label_tgt, size_ct, name_ct = batch_ct
            img_ct = img_ct.to(device)
            label_tgt = label_tgt.long().to(device)
        except : 
            continue

        feature_ct0,feature_ct1,feature_ct2, feature_ct3,feature_ct4,feature_ct5 = model(img_ct)
        pred_ct5 = interp(feature_ct5)
        loss_seg3 = loss_calc(pred_ct5, label_tgt, 0)

        loss_segs = loss_seg+loss_seg2+loss_seg3

        #### Contrastive Adaptation
        B, A, Hs, Ws = feature_cs4.size()
        src_mask = F.interpolate(label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
        src_mask = src_mask.contiguous().view(B * Hs * Ws, )
        
        _, _, Ht, Wt = feature_ct4.size()
        tgt_mask = F.interpolate(label_tgt.unsqueeze(0).float(), size=(Ht, Wt), mode='nearest').squeeze(0).long()
        tgt_mask = tgt_mask.contiguous().view(B * Ht * Wt, )

        src_feat = feature_cs4.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
        tgt_feat = feature_ct4.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)

        feat_estimator_CS_CT.update(features=tgt_feat.detach(), labels=tgt_mask)
        feat_estimator_CS_CT.update(features=src_feat.detach(), labels=src_mask)

        loss_feat1 = pcl_criterion(Proto=feat_estimator_CS_CT.Proto.detach(),
                                      feat=src_feat,
                                      labels=src_mask) \
                        + pcl_criterion(Proto=feat_estimator_CS_CT.Proto.detach(),
                                      feat=tgt_feat,
                                      labels=tgt_mask)
        
        B, A, Hs, Ws = feature_cs5.size()
        src_mask2 = F.interpolate(label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
        src_mask2 = src_mask2.contiguous().view(B * Hs * Ws, )
        
        _, _, Ht, Wt = feature_ct5.size()
        tgt_mask2 = F.interpolate(label_tgt.unsqueeze(0).float(), size=(Ht, Wt), mode='nearest').squeeze(0).long()
        tgt_mask2 = tgt_mask2.contiguous().view(B * Ht * Wt, )
        
        src_out = feature_cs5.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, 19)
        tgt_out = feature_ct5.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, 19)
        
        out_estimator_CS_CT.update(features=tgt_out.detach(), labels=tgt_mask2)
        out_estimator_CS_CT.update(features=src_out.detach(), labels=src_mask2)
        
        loss_feat2 = pcl_criterion(Proto=out_estimator_CS_CT.Proto.detach(),
                                      feat=src_out,
                                      labels=src_mask2) \
                        + pcl_criterion(Proto=out_estimator_CS_CT.Proto.detach(),
                                      feat=tgt_out,
                                      labels=tgt_mask2)

        loss_adv = 0
        D_out = model_CS_CT_D[0](feature_ct4)
        loss_adv += bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
        D_out = model_CS_CT_D[1](prob_2_entropy(F.softmax(pred_ct5, dim=1)))
        loss_adv += bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
        loss_adv = loss_adv*0.01
        #loss_adv.backward()
        loss_seg_feat_adv = loss_segs+loss_feat1+loss_feat2+loss_adv+0.0001*loss_con
        loss_seg_feat_adv.backward()

        for opt in opts:
            opt.step()

        # train D
        # bring back requires_grad
        for param in model_CS_CT_D.parameters():
            param.requires_grad = True

        # train with source
        loss_D_source = 0
        D_out_source = model_CS_CT_D[0](feature_cs4.detach())
        loss_D_source += bce_loss(D_out_source, torch.FloatTensor(D_out_source.data.size()).fill_(source_label).to(device))
        D_out_source = model_CS_CT_D[1](prob_2_entropy(F.softmax(pred_cs5.detach(),dim=1)))
        loss_D_source += bce_loss(D_out_source, torch.FloatTensor(D_out_source.data.size()).fill_(source_label).to(device))
        loss_D_source.backward()

        # train with target
        loss_D_target = 0
        D_out_target = model_CS_CT_D[0](feature_ct4.detach())
        loss_D_target += bce_loss(D_out_target, torch.FloatTensor(D_out_target.data.size()).fill_(target_label).to(device))
        D_out_target = model_CS_CT_D[1](prob_2_entropy(F.softmax(pred_ct5.detach(),dim=1)))
        loss_D_target += bce_loss(D_out_target, torch.FloatTensor(D_out_target.data.size()).fill_(target_label).to(device))
        loss_D_target.backward()
            
        optimizer_CS_CT_D.step()

    if i_iter % 3 == 1 : 

        batch = next(iter(CS_FS_pair_loader))
        fs_image, cs_image,label, size, _, _, fs_arry, cs_arry = batch
        interp = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')
        cs_image = cs_image.to(device)
        label = label.long().to(device)

        images = Variable(cs_image).cuda()
        feature_cs0,feature_cs1,feature_cs2, feature_cs3,feature_cs4,feature_cs5 = model(images)
        pred_cs5 = interp(feature_cs5)
        loss_seg = loss_calc(pred_cs5, label,0)
        #loss_seg.backward()

        images2 = Variable(fs_image).cuda()
        feature_fs0,feature_fs1,feature_fs2, feature_fs3,feature_fs4,feature_fs5 = model(images2)
        pred_fs5 = interp(feature_fs5)
        loss_seg2 = loss_calc(pred_fs5, label,0)

        feature_cs5_logsoftmax = log_m(feature_cs5)
        feature_fs5_softmax = m(feature_fs5)
        feature_fs5_logsoftmax = log_m(feature_fs5)
        feature_cs5_softmax = m(feature_cs5)

        loss_con = kl_loss(feature_fs5_logsoftmax, feature_cs5_softmax)
        
        # train with target
        try : 
            batch = next(iter(FT_loader_ST))
            img, label_tgt, size, name = batch
            img = img.to(device)
            label_tgt = label_tgt.long().to(device)
        except : 
            continue

        feature_ft0,feature_ft1,feature_ft2, feature_ft3,feature_ft4,feature_ft5 = model(img)
        pred_ft5 = interp(feature_ft5)
        loss_seg3 = loss_calc(pred_ft5, label_tgt, 0)

        loss_segs = loss_seg+loss_seg2+loss_seg3

        #### Contrastive Adaptation
        B, A, Hs, Ws = feature_fs4.size()
        src_mask = F.interpolate(label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
        src_mask = src_mask.contiguous().view(B * Hs * Ws, )
        
        _, _, Ht, Wt = feature_ft4.size()
        tgt_mask = F.interpolate(label_tgt.unsqueeze(0).float(), size=(Ht, Wt), mode='nearest').squeeze(0).long()
        tgt_mask = tgt_mask.contiguous().view(B * Ht * Wt, )

        src_feat = feature_fs4.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
        tgt_feat = feature_ft4.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)

        feat_estimator_FS_FT.update(features=tgt_feat.detach(), labels=tgt_mask)
        feat_estimator_FS_FT.update(features=src_feat.detach(), labels=src_mask)

        loss_feat1 = pcl_criterion(Proto=feat_estimator_FS_FT.Proto.detach(),
                                      feat=src_feat,
                                      labels=src_mask) \
                        + pcl_criterion(Proto=feat_estimator_FS_FT.Proto.detach(),
                                      feat=tgt_feat,
                                      labels=tgt_mask)
        
        B, A, Hs, Ws = feature_fs5.size()
        src_mask2 = F.interpolate(label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
        src_mask2 = src_mask2.contiguous().view(B * Hs * Ws, )
        
        _, _, Ht, Wt = feature_ft5.size()
        tgt_mask2 = F.interpolate(label_tgt.unsqueeze(0).float(), size=(Ht, Wt), mode='nearest').squeeze(0).long()
        tgt_mask2 = tgt_mask2.contiguous().view(B * Ht * Wt, )
        
        src_out = feature_fs5.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, 19)
        tgt_out = feature_ft5.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, 19)
        
        out_estimator_FS_FT.update(features=tgt_out.detach(), labels=tgt_mask2)
        out_estimator_FS_FT.update(features=src_out.detach(), labels=src_mask2)
        
        loss_feat2 = pcl_criterion(Proto=out_estimator_FS_FT.Proto.detach(),
                                      feat=src_out,
                                      labels=src_mask2) \
                        + pcl_criterion(Proto=out_estimator_FS_FT.Proto.detach(),
                                      feat=tgt_out,
                                      labels=tgt_mask2)
        
        loss_adv = 0
        D_out = model_FS_FT_D[0](feature_ft4)
        loss_adv += bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
        D_out = model_FS_FT_D[1](prob_2_entropy(F.softmax(pred_ft5, dim=1)))
        loss_adv += bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
        loss_adv = loss_adv*0.01
        #loss_adv.backward()
        loss_seg_feat_adv = loss_segs+loss_feat1+loss_feat2+loss_adv+0.0001*loss_con
        loss_seg_feat_adv.backward()

        for opt in opts:
            opt.step()

        # train D
        # bring back requires_grad
        for param in model_FS_FT_D.parameters():
            param.requires_grad = True

        # train with source
        loss_D_source = 0
        D_out_source = model_FS_FT_D[0](feature_fs4.detach())
        loss_D_source += bce_loss(D_out_source, torch.FloatTensor(D_out_source.data.size()).fill_(source_label).to(device))
        D_out_source = model_FS_FT_D[1](prob_2_entropy(F.softmax(pred_fs5.detach(),dim=1)))
        loss_D_source += bce_loss(D_out_source, torch.FloatTensor(D_out_source.data.size()).fill_(source_label).to(device))
        loss_D_source.backward()

        # train with target
        loss_D_target = 0
        D_out_target = model_FS_FT_D[0](feature_ft4.detach())
        loss_D_target += bce_loss(D_out_target, torch.FloatTensor(D_out_target.data.size()).fill_(target_label).to(device))
        D_out_target = model_FS_FT_D[1](prob_2_entropy(F.softmax(pred_ft5.detach(),dim=1)))
        loss_D_target += bce_loss(D_out_target, torch.FloatTensor(D_out_target.data.size()).fill_(target_label).to(device))
        loss_D_target.backward()
            
        optimizer_FS_FT_D.step()

    if i_iter % 3 == 2 : 

        loss_seg = torch.zeros((1,)).to(device)
        loss_seg2 = torch.zeros((1,)).to(device)
        loss_con = torch.zeros((1,)).to(device)
        
        try : 
            batch_ct = next(iter(CT_loader_ST))
            img_ct, label_tgt_ct, size_ct, name_ct = batch_ct
            img_ct = img_ct.to(device)
            label_tgt_ct = label_tgt_ct.long().to(device)
        except : 
            continue
        
        # train with target
        try : 
            batch = next(iter(FT_loader_ST))
            img, label_tgt_ft, size, name = batch
            img = img.to(device)
            label_tgt_ft = label_tgt_ft.long().to(device)
        except : 
            continue

        feature_ct0,feature_ct1,feature_ct2, feature_ct3,feature_ct4,feature_ct5 = model(img_ct)
        pred_ct5 = interp(feature_ct5)

        feature_ft0,feature_ft1,feature_ft2, feature_ft3,feature_ft4,feature_ft5 = model(img)
        pred_ft5 = interp(feature_ft5)

        loss_seg3 = loss_calc(pred_ft5, label_tgt_ft, 0)
        loss_seg3 += loss_calc(pred_ct5, label_tgt_ct, 0)

        loss_segs = loss_seg+loss_seg2+loss_seg3

 
        #### Contrastive Adaptation
        B, A, Hs, Ws = feature_ct4.size()
        src_mask = F.interpolate(label_tgt_ct.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
        src_mask = src_mask.contiguous().view(B * Hs * Ws, )
        
        _, _, Ht, Wt = feature_ft4.size()
        tgt_mask = F.interpolate(label_tgt_ft.unsqueeze(0).float(), size=(Ht, Wt), mode='nearest').squeeze(0).long()
        tgt_mask = tgt_mask.contiguous().view(B * Ht * Wt, )

        src_feat = feature_ct4.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
        tgt_feat = feature_ft4.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)

        feat_estimator_CT_FT.update(features=tgt_feat.detach(), labels=tgt_mask)
        feat_estimator_CT_FT.update(features=src_feat.detach(), labels=src_mask)

        loss_feat1 = pcl_criterion(Proto=feat_estimator_CT_FT.Proto.detach(),
                                      feat=src_feat,
                                      labels=src_mask) \
                        + pcl_criterion(Proto=feat_estimator_CT_FT.Proto.detach(),
                                      feat=tgt_feat,
                                      labels=tgt_mask)
        
        B, A, Hs, Ws = feature_ct5.size()
        src_mask2 = F.interpolate(label_tgt_ct.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
        src_mask2 = src_mask2.contiguous().view(B * Hs * Ws, )
        
        _, _, Ht, Wt = feature_ft5.size()
        tgt_mask2 = F.interpolate(label_tgt_ft.unsqueeze(0).float(), size=(Ht, Wt), mode='nearest').squeeze(0).long()
        tgt_mask2 = tgt_mask2.contiguous().view(B * Ht * Wt, )
        
        src_out = feature_ct5.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, 19)
        tgt_out = feature_ft5.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, 19)
        
        out_estimator_CT_FT.update(features=tgt_out.detach(), labels=tgt_mask2)
        out_estimator_CT_FT.update(features=src_out.detach(), labels=src_mask2)
        
        loss_feat2 = pcl_criterion(Proto=out_estimator_CT_FT.Proto.detach(),
                                      feat=src_out,
                                      labels=src_mask2) \
                        + pcl_criterion(Proto=out_estimator_CT_FT.Proto.detach(),
                                      feat=tgt_out,
                                      labels=tgt_mask2) 

        loss_adv = 0
        D_out = model_CT_FT_D[0](feature_ft4)
        loss_adv += bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
        D_out = model_CT_FT_D[1](prob_2_entropy(F.softmax(pred_ft5, dim=1)))
        loss_adv += bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
        loss_adv = loss_adv*0.01
        #loss_adv.backward()
        loss_seg_feat_adv = loss_segs+loss_adv+0.0001*loss_con+loss_feat1+loss_feat2
        loss_seg_feat_adv.backward()

        for opt in opts:
            opt.step()

        # train D
        # bring back requires_grad
        for param in model_CT_FT_D.parameters():
            param.requires_grad = True

        # train with source
        loss_D_source = 0
        D_out_source = model_CT_FT_D[0](feature_ct4.detach())
        loss_D_source += bce_loss(D_out_source, torch.FloatTensor(D_out_source.data.size()).fill_(source_label).to(device))
        D_out_source = model_CT_FT_D[1](prob_2_entropy(F.softmax(pred_ct5.detach(),dim=1)))
        loss_D_source += bce_loss(D_out_source, torch.FloatTensor(D_out_source.data.size()).fill_(source_label).to(device))
        loss_D_source.backward()

        # train with target
        loss_D_target = 0
        D_out_target = model_CT_FT_D[0](feature_ft4.detach())
        loss_D_target += bce_loss(D_out_target, torch.FloatTensor(D_out_target.data.size()).fill_(target_label).to(device))
        D_out_target = model_CT_FT_D[1](prob_2_entropy(F.softmax(pred_ft5.detach(),dim=1)))
        loss_D_target += bce_loss(D_out_target, torch.FloatTensor(D_out_target.data.size()).fill_(target_label).to(device))
        loss_D_target.backward()
            
        optimizer_CT_FT_D.step()

    if i_iter % 10 == 0:
        print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f} loss_adv = {3:.3f} loss_D_s = {4:.3f}, loss_D_t = {5:.3f}'.format(
        i_iter, NUM_STEPS, loss_seg.item(), loss_adv.item(), loss_D_source.item(), loss_D_target.item()))
        log(f"iter = {i_iter}/{NUM_STEPS},loss_seg = {loss_seg.item()},loss_seg2 = {loss_seg2.item()},loss_seg3 = {loss_seg3.item()},loss_adv = {loss_adv.item()}, loss_D_s = {loss_D_source.item()}, loss_D_t = {loss_D_target.item()}", timestr)

    if i_iter >= NUM_STEPS_STOP - 1:
        torch.save(model.state_dict(), osp.join(SNAPSHOT_DIR, 'Stage4_' + str(NUM_STEPS_STOP) + '.pth'))
        torch.save(model_CS_CT_D.state_dict(), osp.join(SNAPSHOT_DIR, 'Stage4_Final_CS_CT_D.pth'))
        torch.save(model_FS_FT_D.state_dict(), osp.join(SNAPSHOT_DIR, 'Stage4_Final_FS_FT_D.pth'))
        torch.save(model_CT_FT_D.state_dict(), osp.join(SNAPSHOT_DIR, 'Stage4_Final_CT_FT_D.pth'))
        break
    
    if i_iter % SAVE_PRED_EVERY == 0 and i_iter != 0:
        print('taking snapshot ...')
        evaluate(model,log,timestr)
        torch.save(model.state_dict(), osp.join(SNAPSHOT_DIR, 'Stage4_' + str(i_iter) + '.pth'))
        torch.save(model_CS_CT_D.state_dict(), osp.join(SNAPSHOT_DIR, 'Stage4_' + str(i_iter) + 'CS_CT_D.pth'))
        torch.save(model_FS_FT_D.state_dict(), osp.join(SNAPSHOT_DIR, 'Stage4_' + str(i_iter) + 'FS_FT_D.pth'))
        torch.save(model_CT_FT_D.state_dict(), osp.join(SNAPSHOT_DIR, 'Stage4_' + str(i_iter) + 'CT_FT_D.pth'))
