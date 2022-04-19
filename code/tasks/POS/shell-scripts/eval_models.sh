export CUDA_VISIBLE_DEVICES=0

python run_pos.py \
--model_name_or_path CenIA/albert-xxlarge-spanish-finetuned-pos \
--max_seq_length 512 \
--pad_to_max_length False \
--do_lower_case True \
--output_dir /data/jcanete/evaluation/pos/ \
--use_fast_tokenizer True \
--language es \
--train_language es \
--do_eval \
--do_predict \
--per_device_eval_batch_size 64 \
--logging_dir /data/jcanete/evaluation/pos/ \
--seed 42 \
--cache_dir /data/jcanete/cache \
--use_auth_token True \
--fp16 \