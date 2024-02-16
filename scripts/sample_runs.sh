# For testing on full comp dataset on cambridge cluster
accelerate launch --dynamo_backend no --gpu_ids all --num_processes 1  --num_machines 1 --use_deepspeed trainer/scripts/train.py +experiment=compositionality_example output_dir=/rds/project/rds-lSmP1cwRttU/vu214/reward-models-compositionality-resources/pickscore_trained_model_outputs_test

# To run PickScore on full comp dataset (~1.4M samples) on cambridge cluster
accelerate launch --dynamo_backend no --gpu_ids all --num_processes 1  --num_machines 1 --use_deepspeed trainer/scripts/train.py +experiment=compositionality_full output_dir=/rds/project/rds-lSmP1cwRttU/vu214/reward-models-compositionality-resources/pickscore_trained_model_outputs_full_dataset_no_filtering

# To run PickScore on Pick-a-pic data subset (~960K samples) on cambridge cluster
accelerate launch --dynamo_backend no --gpu_ids all --num_processes 1  --num_machines 1 --use_deepspeed trainer/scripts/train.py +experiment=compositionality_pickapic output_dir=/rds/project/rds-lSmP1cwRttU/vu214/reward-models-compositionality-resources/pickscore_trained_model_outputs_pickapic_dataset_filtering

# To run PickScore on DiffusionDB data subset (~195K samples) on cambridge cluster
accelerate launch --dynamo_backend no --gpu_ids all --num_processes 1  --num_machines 1 --use_deepspeed trainer/scripts/train.py +experiment=compositionality_DiffusionDB output_dir=/rds/project/rds-lSmP1cwRttU/vu214/reward-models-compositionality-resources/pickscore_trained_model_outputs_DiffusionDB_dataset_filtering