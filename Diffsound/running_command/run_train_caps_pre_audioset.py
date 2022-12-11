import os

# string = "python /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/train_spec2.py --name caps_train_audioset_pre --config_file /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/configs/caps_pre_audioset.yaml --load_path /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/OUTPUT/pretrain/000099e_132799iter.pth"

string = "python ../train_spec2.py --gpu 0 --name person_plays_the_bell --config_file ../configs/caps_pre_audioset.yaml --load_path ../../diffsound/audioset_pretrain_diffsound.pth"

os.system(string)

# string = "python train_spec2.py --name caps_train_audioset_pre --config_file configs/caps_pre_audioset.yaml --load_path /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/OUTPUT/pretrain/000099e_132799iter.pth"