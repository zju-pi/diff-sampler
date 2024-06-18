################# A. GITS sampling #################
# Supported datasets are "cifar10", "ffhq", "afhqv2", "imagenet64", "imagenet256", "lsun_bedroom", 
# "lsun_cat", "lsun_bedroom_ldm", "ffhq_ldm", "ms_coco"(Stable Diffusion)
# "num_steps" is the number of timestamps, hence num_steps=7 equals 6 steps

# A.1. Generate a grid of 64 samples
SOLVER_FLAGS="--solver=euler --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
GITS_FLAGS="--dp=True --metric=dev --coeff=1.15 --num_steps_tea=61"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $GITS_FLAGS

SOLVER_FLAGS="--solver=ipndm --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
GITS_FLAGS="--dp=True --metric=dev --coeff=1.15 --num_steps_tea=61"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS $GITS_FLAGS


# A.2. Generate samples for FID evaluation
# We provide the recommended settings below. We use 4 A100 GPUs for sampling. 
# You can change the batch size based on your devices.
SOLVER_FLAGS="--solver=euler --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
GITS_FLAGS="--dp=True --metric=bdev --coeff=1.15 --num_steps_tea=61"
torchrun --standalone --nproc_per_node=4 --master_port=22222 \
sample.py --dataset_name="cifar10" --batch=256 --seeds="0-49999" $SOLVER_FLAGS $SCHEDULE_FLAGS $GITS_FLAGS

SOLVER_FLAGS="--solver=ipndm --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
GITS_FLAGS="--dp=True --metric=dev --coeff=1.15 --num_steps_tea=61"
torchrun --standalone --nproc_per_node=4 --master_port=22222 \
sample.py --dataset_name="cifar10" --batch=256 --seeds="0-49999" $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS $GITS_FLAGS

SOLVER_FLAGS="--solver=dpmpp --num_steps=11 --afs=False"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=2 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"
GITS_FLAGS="--dp=True --metric=dev --coeff=1.20 --num_steps_tea=61 --solver_tea=dpmpp"
torchrun --standalone --nproc_per_node=4 --master_port=22222 \
sample.py --dataset_name="ms_coco" --batch=16 --seeds="0-29999" $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS $GUIDANCE_FLAGS $GITS_FLAGS




################# B. Baseline sampling #################
# (The same as the commands in "diff-solvers-main")
SOLVER_FLAGS="--solver=euler --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS

SOLVER_FLAGS="--solver=ipndm --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS

# You can directly specify the time schedule with a list of timestamps (delete space!)
SOLVER_FLAGS="--solver=ipndm --afs=False"
SCHEDULE_FLAGS="--t_steps=[80,10.9836,3.8811,1.584,0.5666,0.1698,0.002]"
ADDITIONAL_FLAGS="--max_order=4"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS




################# C. Evaluation #################
# C.1. FID
python fid.py calc --images="path/to/images" --ref="path/to/fid/stat"

# C.2. CLIP score
python clip_score.py calc --images="path/to/images"
