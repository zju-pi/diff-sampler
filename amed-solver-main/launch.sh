################# A. Train the AMED predictor #################
# We provide the recommended settings below. We use 4 A100 GPUs for all experiments. 
# You may change the batch size based on your devices.

# A.1. CIFAR10
# "num_steps" is the number of original timestamps. Our method inserts a new timestamp between every two adjacent timestamps,
# hence num_steps=4 finally gives a total of 7 timestamps (6 sampling steps). So NFE=(5 if afs==True else 6).
SOLVER_FLAGS="--sampler_stu=amed --sampler_tea=heun --num_steps=4 --M=1 --afs=True --scale_dir=0.01 --scale_time=0"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"
torchrun --standalone --nproc_per_node=4 --master_port=11111 \
train.py --dataset_name="cifar10" --batch=128 --total_kimg=10 $SOLVER_FLAGS $SCHEDULE_FLAGS

SOLVER_FLAGS="--sampler_stu=ipndm --sampler_tea=ipndm --num_steps=4 --M=2 --afs=True --scale_dir=0.01 --scale_time=0.2"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
torchrun --standalone --nproc_per_node=4 --master_port=11111 \
train.py --dataset_name="cifar10" --batch=128 --total_kimg=10 $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS


# A.2. FFHQ
SOLVER_FLAGS="--sampler_stu=amed --sampler_tea=heun --num_steps=4 --M=1 --afs=True --scale_dir=0.01 --scale_time=0"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"
torchrun --standalone --nproc_per_node=4 --master_port=11111 \
train.py --dataset_name="ffhq" --batch=64 --total_kimg=10 $SOLVER_FLAGS $SCHEDULE_FLAGS

SOLVER_FLAGS="--sampler_stu=ipndm --sampler_tea=ipndm --num_steps=4 --M=2 --afs=True --scale_dir=0.01 --scale_time=0"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
torchrun --standalone --nproc_per_node=4 --master_port=11111 \
train.py --dataset_name="ffhq" --batch=64 --total_kimg=10 $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS


# A.3. ImageNet64
SOLVER_FLAGS="--sampler_stu=amed --sampler_tea=heun --num_steps=4 --M=1 --afs=True --scale_dir=0.01 --scale_time=0"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"
torchrun --standalone --nproc_per_node=4 --master_port=11111 \
train.py --dataset_name="imagenet64" --batch=64 --total_kimg=10 $SOLVER_FLAGS $SCHEDULE_FLAGS

SOLVER_FLAGS="--sampler_stu=ipndm --sampler_tea=ipndm --num_steps=4 --M=2 --afs=True --scale_dir=0.01 --scale_time=0"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
torchrun --standalone --nproc_per_node=4 --master_port=11111 \
train.py --dataset_name="imagenet64" --batch=64 --total_kimg=10 $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS


# A.4. LSUN_Bedroom_ldm
SOLVER_FLAGS="--sampler_stu=dpmpp --sampler_tea=dpmpp --num_steps=4 --M=2 --afs=True --scale_dir=0.01 --scale_time=0"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=3 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=uncond --guidance_rate=1"
torchrun --standalone --nproc_per_node=4 --master_port=11111 \
train.py --dataset_name="lsun_bedroom_ldm" --batch=64 --total_kimg=10 $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS $GUIDANCE_FLAGS


# A.5. Stable Diffusion
# The NFE doubles due to the classifier-free guidance
SOLVER_FLAGS="--sampler_stu=dpmpp --sampler_tea=dpmpp --num_steps=4 --M=2 --afs=True --scale_dir=0 --scale_time=0.2"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=2 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"
torchrun --standalone --nproc_per_node=4 --master_port=11111 \
train.py --dataset_name="ms_coco" --batch=32 --total_kimg=5 $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS $GUIDANCE_FLAGS




################# B. Generate 50k samples for FID evaluation #################
# After finishing the training, the AMED predictor will be saved at "./exps" with a five digit experiment number (e.g. 00000). 
# The settings for sampling are stored in the predictor. You can sample with the AMED predictor by giving the file path 
# or the exp number (e.g. 0) of the AMED predictor in ```--predictor_path```
# B.1. Usually used
torchrun --standalone --nproc_per_node=4 --master_port=22222 \
sample.py --predictor_path=0 --batch=128 --seeds="0-49999"

# B.1. For Stable Diffusion
torchrun --standalone --nproc_per_node=4 --master_port=22222 \
sample.py --predictor_path=0 --batch=8 --seeds="0-29999"




################# C. Evaluation #################
# C.1. FID
python fid.py calc --images="path/to/images" --ref="path/to/fid/stat"

# C.2. CLIP score
python clip_score.py calc --images="path/to/images"
