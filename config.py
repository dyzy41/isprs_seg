import os
batch_size = 32  # 设置批量大小
use_gpu = True  # 是否使用use_gpu
img_size = 512  # 输入图片大小
overlap = 64
epoches = 800
base_lr = 0.001  # 学习率
weight_decay = 2e-5
momentum = 0.9
power = 0.99
gpu_id = '0, 1'

loss_type = 'ce'
save_iter = 20
num_workers = 1
val_visual = True
image_driver = 'pillow'   #pillow, gdal

num_class = 6  # some parameters
model_name = 'DeepLabV3Plus'  # ' unet, res_unet_psp, res_unet'
input_bands = 3
if input_bands == 4:
    mean = (0.472455, 0.320782, 0.318403, 0.357)
    std = (0.144, 0.151, 0.211, 0.195)
else:
    mean = (0.472455, 0.320782, 0.318403)
    std = (0.215084, 0.408135, 0.409993)  # 标准化参数

#data path
root_data = '../vai_data/cut_data'
dataset = 'massroad'
exp_name = '0915'
save_dir = '../{}_files'.format(exp_name)
model_experision = '3'
save_dir_model = os.path.join(save_dir, model_name+'_'+model_experision)
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
if os.path.exists(save_dir_model) is False:
    os.mkdir(save_dir_model)
    # shutil.copy('config.py', os.path.join(save_dir_model, 'config.py'))
data_dir = os.path.join(save_dir, 'data_slice_{}'.format(dataset))
train_path = os.path.join(root_data, 'train')
train_gt = os.path.join(root_data, 'train_labels')
val_path = '../vai_data/train_img/val'
val_gt = '../vai_data/val_gt'
test_path = '../vai_data/train_img/val'
test_gt = os.path.join(root_data, 'test_labels')
save_path = '../resultsucnet_val'

# save path
output = os.path.join(save_dir_model, './result_{}/'.format(model_name))
output_gray = os.path.join(save_dir_model, './result_gray_{}/'.format(model_name))
model_dir = os.path.join(save_dir_model, './pth_{}/'.format(model_name))
