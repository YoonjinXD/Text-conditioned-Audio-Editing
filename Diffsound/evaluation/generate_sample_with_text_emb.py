import os
import argparse

import torch
import numpy as np

from generate_samples_batch import Diffsound

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training script')
    parser.add_argument('-m', '--model_path', type=str, default='/home/yoonjin/ongoing/Text-to-sound-Synthesis/diffsound/diffsound_audiocaps.pth')
    parser.add_argument('-p', '--prompt', type=str, default='Cat is whistling')
    parser.add_argument('-r', '--replication', type=int, default=2)
    parser.add_argument('-s', '--save_root', type=str, default='./save', help='Only absolute path available')
    args = parser.parse_args()

    # 0. Settings
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(curr_dir, 'caps_text.yaml')
    ckpt_vocoder = os.path.join('/'.join(curr_dir.split('/')[:-1]), 'vocoder/logs/vggsound/')
    
    replication = args.replication
    model_path = args.model_path
    target_text = args.prompt
    
    if args.save_root is None:
        save_path = os.path.join(curr_dir, 'save', '-'.join([target_text.replace(' ', '_'), os.path.basename(model_path)]))
    else:
        save_path = os.path.join(args.save_root, '-'.join([target_text.replace(' ', '_'), os.path.basename(model_path)]))
    os.makedirs(save_path, exist_ok=True)
    
    # 1.1 Get the original target prompt embedding.
    diffsound = Diffsound(config=config_path, path=model_path, ckpt_vocoder=ckpt_vocoder)
    target_emb = diffsound.save_text_emb(target_text, save_path, 'target')
    emb = target_emb.clone()
    
    # 1.2 Sample from the original target prompt
    diffsound.inference_generate_sample_with_text_emb(text_emb=target_emb, truncation_rate=0.85, 
                                                      replicate=replication, save_root=save_path, 
                                                      inference_name='target', fast=False)
    
    # 2. Optimize the embedding
    emb.requires_grad = True
    lr = 0.001
    it = 500
    opt = torch.optim.Adam([emb], lr=lr)
    
    # 3. Get and save the Optimized prompt
    finetuned_diffsound = Diffsound(config=config_path, path=finetuned_model_path, ckpt_vocoder=ckpt_vocoder)
    optimized_emb = finetuned_diffsound.save_text_emb(target_text, save_path, 'optimized')
    
    # 4. Interpolate and save the emb
    intervals = np.linspace(0,1,11) # 0, 0.1, 0.2 ... 1.0
    interpolated_embs = [eta*target_emb + (1-eta)*optimized_emb for eta in intervals]
    
    # 5. Generate sample with the interpolated text feature
    replication = args.replication
    finetuned_diffsound.inference_generate_sample_with_text_emb(text_emb=target_emb, truncation_rate=0.85, 
                                                                replicate=replication, save_root=save_path, 
                                                                inference_name='target', fast=False)
    finetuned_diffsound.inference_generate_sample_with_text_emb(text_emb=optimized_emb, truncation_rate=0.85, 
                                                                replicate=replication, save_root=save_path, 
                                                                inference_name='optimized', fast=False)
    for i, interval in enumerate(intervals):
        finetuned_diffsound.inference_generate_sample_with_text_emb(text_emb=interpolated_embs[i], truncation_rate=0.85, 
                                                                    replicate=replication, save_root=save_path, 
                                                                    inference_name='interpolates', fast=False,
                                                                    name='{:.1f}'.format(interval))