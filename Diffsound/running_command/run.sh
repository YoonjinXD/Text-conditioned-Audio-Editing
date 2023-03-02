#!/bin/bash

while IFS=',' read -r y_id prompt
do
    if [ "$y_id" = "youtube_id" ]
    then
        continue
    fi
    
    prompt_hypen=$(echo "$prompt" | tr -s ' ' '_')

    echo "[[[[-----------FINETUNE CLIP on $y_id, $prompt---------------]]]"
    python ../train_spec2.py --gpu=1 --name="$prompt_hypen" --output="/media/daftpunk2/home/yoonjin/exp/clip_finetune_newpretrained"

    echo "[[[[-----------FINETUNE Decoders on $y_id, $prompt---------------]]]"
    python ../train_spec2.py --gpu=1 --name="$prompt_hypen" --config_file="../configs/caps_2stage.yaml" --load_path="/media/daftpunk2/home/yoonjin/exp/clip_finetune_newpretrained/$prompt_hypen/checkpoint/000199e_199iter.pth" --output="/media/daftpunk2/home/yoonjin/exp/decoder_finetune_newpretrained"

    echo "[[[[-----------INFERENCE on $y_id, $prompt---------------]]]"
    model_path="/media/daftpunk2/home/yoonjin/exp/decoder_finetune_newpretrained/$prompt_hypen/checkpoint/000399e_399iter.pth"
    python ../evaluation/generate_sample_with_text_emb.py -m="$model_path" -p="$prompt" -s="/media/daftpunk2/home/yoonjin/2nd_inference_save_newpretrained"
done < ../../Data/preproc_list_0.csv