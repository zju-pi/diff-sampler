################# Training #################
# We provide the recommended settings below. We use 4 A100 GPUs for all experiments. 
# You may change the batch size based on your devices.

# Train a 2-NFE SFD (for EDM models trained on cifar10, ffhq, afvqv2 and imagenet64)
# "num_steps" is the number of timestamps (sampling steps + 1). The use of AFS saves 1 sampling step,
# hence (num_steps=4 with afs=False) equals 3 sampling steps and (num_steps=4 with afs=True）equals 2 sampling steps.
torchrun --standalone --nproc_per_node=4 --master_port=12345 train.py \
--dataset_name="cifar10" --total_kimg=200 --batch=128 --lr=5e-5 \
--num_steps=4 --M=3 --afs=True --sampler_tea="dpmpp" --max_order=3 --predict_x0=True --lower_order_final=True \
--schedule_type="polynomial" --schedule_rho=7 --use_step_condition=False --is_second_stage=False

# Train SFD-v (NFE-variable version, allow sampling for num_steps within 4 to 7, a.k.a. NFE within 2 to 5, using one model)
torchrun --standalone --nproc_per_node=4 --master_port=12345 train.py \
--dataset_name="cifar10" --total_kimg=800 --batch=128 --lr=5e-5 \
--num_steps=4 --M=3 --afs=True --sampler_tea="dpmpp" --max_order=3 --predict_x0=True --lower_order_final=True \
--schedule_type="polynomial" --schedule_rho=7 --use_step_condition=True --is_second_stage=False

# Second-stage distillation for 1-NFE model 
# (for M=2, the teacher should be a SFD model trained with num_steps=7 or a SFD-v model)
torchrun --standalone --nproc_per_node=4 --master_port=12345 train.py \
--model_path="path/to/the/first-stage/model" \
--dataset_name="cifar10" --total_kimg=2000 --batch=128 --lr=5e-4 \
--num_steps=3 --M=2 --afs=True --sampler_tea="euler" --is_second_stage=True

################# Sampling #################
# After training, the distilled SFD model will be saved at "./exps" with a five digit experiment number (e.g. 00000). 
# The settings for sampling are stored in the model file. You can perform accelerated sampling with SFD by giving 
# the file path or the experiment digit number (e.g. 0) to `--model_path`.

# Sample 50k images using SFD for FID evaluation
torchrun --standalone --nproc_per_node=4 --master_port=12345 sample.py \
--dataset_name='cifar10' --model_path=0 --seeds='0-49999' --batch=256

# Sample 50k images using SFD-v for FID evaluation
# When use_step_condition=True is used for distillation, set a specific num_steps during sampling
torchrun --standalone --nproc_per_node=4 --master_port=12345 sample.py \
--dataset_name='cifar10' --model_path=0 --seeds='0-49999' --batch=256 --num_steps=4

################# Evaluation #################
# FID
python fid.py calc --images="path/to/generated/images" --ref="path/to/fid/stat"

# Calculate precision, recall, density and coverage
# The reference images for CIFAR-10 (cifar10-32x32.zip) can be downloaded here: 
# https://drive.google.com/file/d/196tB1pdpFzZ4cAuHxF_p46P1Aw37bUHz/view?usp=drive_link
python prdc.py calc --images="path/to/generated/images" --images_ref="path/to/reference/images"
