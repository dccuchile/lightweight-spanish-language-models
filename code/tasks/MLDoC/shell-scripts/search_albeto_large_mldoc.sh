export CUDA_VISIBLE_DEVICES=0
batch_sizes=(8)
learning_rates=(1e-6 2e-6 3e-6 5e-6)
epochs=(2 3 4)

for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for n_epoch in "${epochs[@]}"; do
            python run_glue.py \
            --model_name_or_path CenIA/albert_large_spanish \
            --train_file /home/jcanete/datasets/MLDoc/spanish.train.10000.json \
            --validation_file /home/jcanete/datasets/MLDoc/spanish.dev.json \
            --max_seq_length 512 \
            --pad_to_max_length False \
            --output_dir /data/jcanete/all_results/mldoc/albeto_large/epochs_"$n_epoch"_bs_16_lr_"$lr" \
            --use_fast_tokenizer True \
            --do_train \
            --do_eval \
            --per_device_eval_batch_size "$bs" \
            --per_device_train_batch_size "$bs" \
            --learning_rate "$lr" \
            --num_train_epochs "$n_epoch" \
            --logging_dir /data/jcanete/all_results/mldoc/albeto_large/epochs_"$n_epoch"_bs_16_lr_"$lr" \
            --seed 42 \
            --cache_dir /data/jcanete/cache \
            --use_auth_token True \
            --evaluation_strategy steps \
            --save_steps 100 \
            --eval_steps 100 \
            --load_best_model_at_end True \
            --fp16 \
            ;
        done
    done
done