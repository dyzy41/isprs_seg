import timm

def get_model(model_name, in_c=3, pretrained=False):
    model = timm.create_model(model_name, pretrained=pretrained, features_only=True, in_chans=in_c)
    return model