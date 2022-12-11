import os
import argparse
import shutil
# import pandas as pd
import pickle
import csv
from tqdm import tqdm

def get_mel_paths(root, phase, y_id):
    os.makedirs(os.path.join(root, phase), exist_ok=True)
    return os.path.join(root, phase, f'{y_id}_mel.npy')

def make_caption_txt(root, phase, y_id, caption):
    os.makedirs(os.path.join(root, 'text', phase), exist_ok=True)
    with open(os.path.join(root, 'text', phase, f'{y_id}.txt'), 'w') as f:
        f.write(caption.replace('_', ' '))
    f.close()

def make_name_list(root, phase, y_id):
    with open(os.path.join(root, phase, 'name_list.pkl'), 'wb') as f:
        pickle.dump([y_id], f)
    f.close()

if __name__ == '__main__':
    # argparse
    # parser = argparse.ArgumentParser(description='PyTorch Training script')
    # parser.add_argument('-i', '--y_id', type=str)
    # parser.add_argument('-p', '--prompt', type=str)
    # parser.add_argument('--target_phase', type=str, default='train')
    # args = parser.parse_args()
    
    
    data_path = 'caps_full'
    # df = pd.read_csv(f'./{args.target_phase}.csv')
    
    # csv
    preproc_csv_path = './preproc_list.csv'
    target_phase = 'train' # args.target_phase

    with open(preproc_csv_path, 'r') as f_read:
        csv_reader = csv.DictReader(f_read)
        
        for row in tqdm(csv_reader): # row.keys(): youtube_id,caption
            y_id = row['youtube_id'] # args.y_id
            prompt = row['prompt'] # args.prompt
            prompt = prompt.replace(' ', '_')
            
            os.makedirs(prompt, exist_ok=True)
    
            for phase in ['train', 'val', 'test']:
                shutil.copyfile(get_mel_paths(data_path, target_phase, y_id),
                                get_mel_paths(prompt, phase, y_id))
                make_caption_txt(prompt, phase, y_id, caption=prompt)
                make_name_list(prompt, phase, y_id)