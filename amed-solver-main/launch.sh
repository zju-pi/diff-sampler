##################### Train the AMED predictor ######################
SOLVER_FLAGS="--sampler_stu=ipndm --sampler_tea=ipndm --num_steps=4 --M=2 --afs=True --max_order=4 --scale_time=True"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
GUIDANCE_FLAGS=""
torchrun --standalone --nproc_per_node=1 train.py \
--dataset_name="name of the dataset" \
--model_path="/path/to/your/model" \
--batch=64 \
--total_kimg=10 \
$SOLVER_FLAGS \
$SCHEDULE_FLAGS \
$ADDITIONAL_FLAGS \
$GUIDANCE_FLAGS

############## Generate 50k samples for FID evaluation ##############
torchrun --standalone --nproc_per_node=1 sample.py \
--predictor_path="/path/or/exp/order/of/AMED/predictor" \
--model_path="/path/to/your/model" \
--dataset_name="name of the dataset" \
--batch=64 \
--seeds="0-49999"

########################## FID evaluation ###########################
python fid.py calc --images="path/to/images" --ref="path/to/fid/stat"

