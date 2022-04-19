export CUDA_VISIBLE_DEVICES=0

python run_glue.py \
--model_name_or_path CenIA/albert-xxlarge-spanish-finetuned-pawsx \
--max_seq_length 512 \
--pad_to_max_length False \
--train_file /home/jcanete/datasets/PAWS-X/es/translated_train.json \
--validation_file /home/jcanete/datasets/PAWS-X/es/dev_2k.json \
--test_file /home/jcanete/datasets/PAWS-X/es/test_2k.json \
--output_dir /data/jcanete/evaluation/pawsx \
--use_fast_tokenizer True \
--do_predict \
--per_device_eval_batch_size 64 \
--logging_dir /data/jcanete/evaluation/pawsx \
--seed 42 \
--cache_dir /data/jcanete/cache \
--use_auth_token True \
--fp16 