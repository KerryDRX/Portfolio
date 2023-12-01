from config import cfg
import monai


def build_model(model_name):
    if model_name == 'autoencoder':
        model = monai.networks.nets.AutoEncoder(
            spatial_dims=3, 
            in_channels=1, 
            out_channels=1, 
            channels=(128, 256, 512, 1024), 
            strides=(2, 2, 2, 2),
        )
    else:
        raise NotImplementedError(f'Model not implemented: {model_name}')
    return model
