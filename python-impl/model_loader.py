from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
import torch
import model_converter

def preload_models_from_standard_weights(device):
    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(torch.load("models/encoder.pth"))

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(torch.load("models/decoder.pth"))

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(torch.load("models/diffusion.pth"))

    clip = CLIP().to(device)
    clip.load_state_dict(torch.load("models/clip.pth"))

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }