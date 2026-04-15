import torch
from model import Autoencoder

def get_score(model, img):
    model.eval()
    with torch.no_grad():
        recon = model(img)
        diff = torch.abs(img - recon)
        score = diff.mean().item()
    return score, recon