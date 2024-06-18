################# A. Generate a grid of samples #################
# Supported datasets are "cifar10", "ffhq", "afhqv2", "imagenet64", "imagenet256", "lsun_bedroom", 
# "lsun_cat", "lsun_bedroom_ldm", "ffhq_ldm", "ms_coco"(Stable Diffusion)

# A.1. Commands for generating samples on CIFAR-10 (and other EDM models) with different solvers
# Below we use recommended settings for each solver as example, but you can adjust them if needed
# DDIM ("num_steps" is the number of timestamps, hence num_steps=7 equals 6 steps)
SOLVER_FLAGS="--solver=euler --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS

# Heun (1 step = 2 NFE)
SOLVER_FLAGS="--solver=heun --num_steps=4 --afs=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS

# DPM-Solver(2S) (1 step = 2 NFE)
SOLVER_FLAGS="--solver=dpm --num_steps=4 --afs=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS 

# iPNDM
SOLVER_FLAGS="--solver=ipndm --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS

# iPNDM_v
SOLVER_FLAGS="--solver=ipndm_v --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS

# DEIS-tAB3
SOLVER_FLAGS="--solver=deis --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=2"
ADDITIONAL_FLAGS="--max_order=4 --deis_mode=tab"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS

# DPM-Solver++(3M)
SOLVER_FLAGS="--solver=dpmpp --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=logsnr"
ADDITIONAL_FLAGS="--max_order=3 --predict_x0=True --lower_order_final=True"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS

# UniPC-3
SOLVER_FLAGS="--solver=unipc --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=logsnr"
ADDITIONAL_FLAGS="--max_order=3 --predict_x0=True --lower_order_final=True --variant=bh2"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS

# A.2. Commands for generating samples on CM models
SOLVER_FLAGS="--solver=dpmpp --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=logsnr"
ADDITIONAL_FLAGS="--max_order=3 --predict_x0=True --lower_order_final=True"
python sample.py --dataset_name="lsun_bedroom" --batch=4 --seeds="0-3" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS

# A.3. Commands for generating samples on ADM models
SOLVER_FLAGS="--solver=dpmpp --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=3 --predict_x0=True --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=cg --guidance_rate=4"
python sample.py --dataset_name="imagenet256" --batch=4 --seeds="0-3" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS $GUIDANCE_FLAGS

# A.4. Commands for generating samples on LDM models
# Stable Diffusion (1 step = 2 NFE due to the classifier-free guidance)
SOLVER_FLAGS="--solver=dpmpp --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=2 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"
python sample.py --dataset_name="ms_coco" --batch=4 --seeds="0-3" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS $GUIDANCE_FLAGS

# Others
SOLVER_FLAGS="--solver=dpmpp --num_steps=7 --afs=False"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=3 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=uncond --guidance_rate=1"
python sample.py --dataset_name="lsun_bedroom_ldm" --batch=4 --seeds="0-3" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS $GUIDANCE_FLAGS




################# B. Generate samples for FID evaluation #################
# These commands can be parallelized across multiple GPUs by setting --nproc_per_node
# num_steps is the number of timestamps, hence num_steps=6 equals 5 steps

# B.1. on CIFAR10 (always 50k images)
SOLVER_FLAGS="--solver=ipndm --num_steps=6 --afs=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
torchrun --standalone --nproc_per_node=1 --master_port=11111 \
sample.py --dataset_name="cifar10" --batch=128 --seeds="0-49999" $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS 

# B.2. on Stable Diffusion (usually 30k or 5k images are required)
# For Stable Diffusion 1 step = 2 NFE due to the classifier-free guidance
SOLVER_FLAGS="--solver=dpmpp --num_steps=5 --afs=False"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=2 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"
torchrun --standalone --nproc_per_node=1 --master_port=11111 \
sample.py --dataset_name="ms_coco" --batch=4 --seeds="0-29999" $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS $GUIDANCE_FLAGS




################# C. Use your own prompt for text-to-image generation #################
SOLVER_FLAGS="--solver=dpmpp --num_steps=6 --afs=False"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=2 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"
torchrun --standalone --nproc_per_node=1 --master_port=11111 \
sample.py --dataset_name="ms_coco" --batch=4 --seeds="0-3" --grid=True \
--prompt="a photograph of an astronaut riding a horse" \
$SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS $GUIDANCE_FLAGS




################# D. Use your own time schedule with a list of timestamps #################
# Remember to delete the space of the list!
SOLVER_FLAGS="--solver=ipndm --afs=False"
SCHEDULE_FLAGS="--t_steps=[80,10.9836,3.8811,1.584,0.5666,0.1698,0.002]"
ADDITIONAL_FLAGS="--max_order=4"
python sample.py --dataset_name="cifar10" --batch=64 --seeds="0-63" --grid=True $SOLVER_FLAGS $SCHEDULE_FLAGS $ADDITIONAL_FLAGS




################# E. Evaluation #################
# D.1. FID
python fid.py calc --images="path/to/images" --ref="path/to/fid/stat"

# D.2. CLIP score
python clip_score.py calc --images="path/to/images"
