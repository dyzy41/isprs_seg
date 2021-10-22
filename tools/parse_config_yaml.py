import yaml
import os


def get_base_param(yaml_file):
    f = open(yaml_file, 'r', encoding='utf-8')
    params = yaml.load(f, Loader=yaml.FullLoader)
    return params


def add_param(param_dict):
    if param_dict['input_bands'] == 3:
        param_dict['mean'] = (0.472455, 0.320782, 0.318403)
        param_dict['std'] = (0.215084, 0.408135, 0.409993)
    else:
        param_dict['mean'] = (0.472455, 0.320782, 0.318403, 0.357)
        param_dict['std'] = (0.215084, 0.408135, 0.409993, 0.195)
    param_dict['save_dir'] = '../{}_files'.format(param_dict['exp_name'])
    param_dict['save_dir_model'] = os.path.join(param_dict['save_dir'], param_dict['model_name'] + '_' + param_dict['model_experision'])
    if os.path.exists(param_dict['save_dir']) is False:
        os.mkdir(param_dict['save_dir'])
    if os.path.exists(param_dict['save_dir_model']) is False:
        os.mkdir(param_dict['save_dir_model'])
    param_dict['data_dir'] = os.path.join(param_dict['save_dir'], 'data_slice_{}'.format(param_dict['dataset']))
    param_dict['train_path'] = os.path.join(param_dict['root_data'], 'train')
    param_dict['train_gt'] = os.path.join(param_dict['root_data'], 'train_labels')
    param_dict['test_gt'] = os.path.join(param_dict['root_data'], 'test_labels')
    param_dict['model_dir'] = os.path.join(param_dict['save_dir_model'], './pth_{}/'.format(param_dict['model_name']))
    return param_dict


def parse_yaml(yaml_file):
    params = get_base_param(yaml_file)
    params = add_param(params)
    return params

if __name__ == '__main__':
    f = '../config.yaml'
    params = parse_yaml(f)
    print('ok')
