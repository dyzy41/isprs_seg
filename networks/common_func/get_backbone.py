import timm
import torch


def get_model(model_name, in_c=3, pretrained=True,
              checkpoint_path=None):
    if pretrained:
        model = timm.create_model(model_name, pretrained=pretrained,
                                  features_only=True, in_chans=in_c)
        return model
    if checkpoint_path is not None:
        model = timm.create_model(model_name, pretrained=False,
                                  features_only=True, in_chans=in_c)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        del state_dict['fc' + '.weight']
        del state_dict['fc' + '.bias']
        model.load_state_dict(state_dict, strict=False)
        return model
