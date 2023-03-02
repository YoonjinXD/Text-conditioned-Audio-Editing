import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import datetime

from specvqgan.modules.losses.vggishish.transforms import Crop
from generate_samples_batch import Diffsound

class CropImage(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training script')
    parser.add_argument('-m', '--model_path', type=str, default='../../diffsound/diffsound_audioset_audiocaps.pth')
    parser.add_argument('-a', '--audio_path', type=str)
    parser.add_argument('-s', '--save_root', type=str, default=None)
    args = parser.parse_args()

    # 0. Settings
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(curr_dir, 'caps_text.yaml')
    ckpt_vocoder = os.path.join('/'.join(curr_dir.split('/')[:-1]), 'vocoder/logs/vggsound/')
    
    # 1. Get the target audio and the model
    diffsound = Diffsound(config=config_path, path=args.model_path, ckpt_vocoder=ckpt_vocoder)
    spec = np.load(args.audio_path)
    
    # 2. Preprocess spec
    item = {}
    item['input'] = spec
    
    mel_num=80
    spec_crop_len=848
    transforms = CropImage([mel_num, spec_crop_len], False)
    
    item = transforms(item)
    image = 2 * item['input'] - 1 # why --> it also expects inputs in [-1, 1] but specs are in [0, 1]
    # image = image[None,:,:]
    spec = image.astype(np.float32)
    
    # 2. Recon with the VQ-VAE
    save_root = os.path.dirname(args.audio_path) if args.save_root is None else args.save_root
    save_name = os.path.basename(args.audio_path).strip('_mel.npy')
    diffsound.reconstruct_sample(spec, save_root, save_name)