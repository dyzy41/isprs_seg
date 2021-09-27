import timm

def get_model(model_name, pretrained=False):
    model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
    return model