export CUDA_VISIBLE_DEVICES=0

python run_glue.py \
--model_name_or_path CenIA/distillbert-base-spanish-uncased-finetuned-mldoc \
--max_seq_length 512 \
--pad_to_max_length False \
--train_file /home/jcanete/datasets/MLDoc/spanish.train.10000.json \
--validation_file /home/jcanete/datasets/MLDoc/spanish.dev.json \
--test_file /home/jcanete/datasets/MLDoc/spanish.test.json \
--output_dir /data/jcanete/evaluation/pawsx \
--use_fast_tokenizer True \
--do_predict \
--per_device_eval_batch_size 64 \
--logging_dir /data/jcanete/evaluation/pawsx \
--seed 42 \
--cache_dir /data/jcanete/cache \
--use_auth_token True \
--fp16 