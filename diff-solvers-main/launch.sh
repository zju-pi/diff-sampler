############## Generate 50k samples for FID evaluation ##############
SOLVER_FLAGS="--solver=ipndm --num_steps=6 --afs=False --denoise_to_zero=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
GUIDANCE_FLAGS=""
torchrun --standalone --nproc_per_node=1 sample.py \
--dataset_name="name of the dataset" \
--model_path="/path/to/your/model" \
--batch=64 \
--seeds="0-49999" \
$SOLVER_FLAGS \
$SCHEDULE_FLAGS \
$ADDITIONAL_FLAGS \
$GUIDANCE_FLAGS


################# Generate 16 samples in grid form ##################
SOLVER_FLAGS="--solver=ipndm --num_steps=6 --afs=False --denoise_to_zero=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
GUIDANCE_FLAGS=""
torchrun --standalone --nproc_per_node=1 sample.py \
--dataset_name="name of the dataset" \
--model_path="/path/to/your/model" \
--batch=16 \
--seeds="0-15" \
--grid=True \
$SOLVER_FLAGS \
$SCHEDULE_FLAGS \
$ADDITIONAL_FLAGS \
$GUIDANCE_FLAGS


################# Example: Generate samples on Stable-Diffusion with DPM-Solver++(2M) ##################
SOLVER_FLAGS="--solver=dpmpp --num_steps=6 --afs=False --denoise_to_zero=False"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=2 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"
torchrun --standalone --nproc_per_node=1 sample.py \
--dataset_name="ms_coco" \
--model_path="/path/to/stable-diffusion-v1/model.ckpt" \
--batch=4 \
--seeds="0-3" \
--grid=True \
--prompt="a photograph of an astronaut riding a horse" \
$SOLVER_FLAGS \
$SCHEDULE_FLAGS \
$ADDITIONAL_FLAGS \
$GUIDANCE_FLAGS
# --prompt_path="/path/to/MS-COCO_val2014_30k_captions.csv" \  
# add --prompt_path, set --seeds="0-29999", delete --prompt and --grid to generating 30k samples for FID evaluation


########################## FID evaluation ##########################
python fid.py calc --images="path/to/images" --ref="path/to/fid/stat"
