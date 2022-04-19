#export CUDA_VISIBLE_DEVICES=1
batch_sizes=(16)
learning_rates=(1e-6 2e-6 3e-6 5e-6)
epochs=(2 3 4)

for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for n_epoch in "${epochs[@]}"; do
            python -m torch.distributed.launch --nproc_per_node 1 run_qa.py \
            --model_name_or_path CenIA/albert_xlarge_spanish \
            --train_file /home/fbravo/ft-spanish-models/datasets/QA/TAR/tar-train.json \
            --validation_file /home/fbravo/ft-spanish-models/datasets/QA/TAR/tar-dev.json \
            --max_seq_length 512 \
            --pad_to_max_length False \
            --output_dir /home/fbravo/data/all_results/tar/albeto_xlarge/epochs_"$n_epoch"_bs_16_lr_"$lr" \
            --do_train \
            --do_eval \
            --per_device_eval_batch_size "$bs" \
            --per_device_train_batch_size "$bs" \
            --learning_rate "$lr" \
            --num_train_epochs "$n_epoch" \
            --logging_dir /home/fbravo/data/all_results/tar/albeto_xlarge/epochs_"$n_epoch"_bs_16_lr_"$lr" \
            --seed 42 \
            --cache_dir /home/fbravo/data/cache \
            --use_auth_token True \
            --evaluation_strategy steps \
            --save_steps 300 \
            --eval_steps 300 \
            --load_best_model_at_end True \
            --metric_for_best_model f1 \
            --greater_is_better True \
            --fp16 \
            --gradient_accumulation_steps 1 \
            ;
        done
    done
done