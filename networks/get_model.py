from networks import *

def get_net(model_name, in_c, num_class, img_size=512):
    if model_name == 'DeepLabV3Plus':
        model = DeepLabV3Plus(in_c=in_c, num_class=num_class)
    elif model_name == 'UCTransNet':
        model = UCTransNet(in_c=in_c, num_class=num_class)
    elif model_name == 'PSPNet':
        model = PSPNet(in_c=in_c, num_class=num_class)
    elif model_name == 'SegNet':
        model = SegNet(in_c=in_c, num_class=num_class)
    elif model_name == 'U_Net':
        model = U_Net(in_c=in_c, num_class=num_class)
    elif model_name == 'Res_UNet_50':
        model = Res_UNet_50(in_c=in_c, num_class=num_class)
    else:
        raise (
            'this model is not exist!!!!, existing mode is pspnet ,densenet_aspp, segnet, refinenet, unet1, UNet, UNet_2Plus, UNet_3Plus')
    return model
