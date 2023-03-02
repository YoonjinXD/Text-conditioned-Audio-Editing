SPEC_DIR_PATH="vggsound.VGGSound"
cls_token_dir_path="vggsound.VGGSound"
NOW=`date +"%Y-%m-%dT%H-%M-%S"`

python generate_samples_caps.py \
        sampler.config_sampler=configs/sampler.yaml \
        data.params.spec_dir_path=$SPEC_DIR_PATH \
        data.params.cls_token_dir_path=$cls_token_dir_path \
        sampler.now=$NOW