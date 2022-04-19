export CUDA_VISIBLE_DEVICES=0,1
batch_sizes=(32)
learning_rates=(1e-5 2e-5 3e-5 5e-5)
epochs=(2 3 4)

for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for n_epoch in "${epochs[@]}"; do
            python run_qa.py \
            --model_name_or_path CenIA/albert_base_spanish \
            --train_file /home/jcanete/datasets/QA/MLQA/tar-train.json \
            --validation_file /home/jcanete/datasets/QA/MLQA/mlqa-dev.json \
            --max_seq_length 512 \
            --pad_to_max_length False \
            --output_dir /data/jcanete/all_results/tar_mlqa/albeto_base/epochs_"$n_epoch"_bs_64_lr_"$lr" \
            --do_train \
            --do_eval \
            --per_device_eval_batch_size "$bs" \
            --per_device_train_batch_size "$bs" \
            --learning_rate "$lr" \
            --num_train_epochs "$n_epoch" \
            --logging_dir /data/jcanete/all_results/tar_mlqa/albeto_base/epochs_"$n_epoch"_bs_64_lr_"$lr" \
            --seed 42 \
            --cache_dir /data/jcanete/cache \
            --use_auth_token True \
            --evaluation_strategy steps \
            --save_steps 300 \
            --eval_steps 300 \
            --load_best_model_at_end True \
            --metric_for_best_model f1 \
            --greater_is_better True \
            --fp16 \
            ;
        done
    done
done