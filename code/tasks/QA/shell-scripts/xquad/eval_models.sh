export CUDA_VISIBLE_DEVICES=0
model_name="CenIA/distillbert-base-spanish-uncased-finetuned-qa-tar"

python run_qa.py \
--model_name_or_path $model_name \
--train_file /home/jcanete/ft-spanish-models/datasets/QA/TAR/tar-train.json \
--validation_file /home/jcanete/ft-spanish-models/datasets/QA/TAR/tar-dev.json \
--test_file /home/jcanete/ft-spanish-models/datasets/QA/XQuAD/xquad-test.json \
--max_seq_length 512 \
--pad_to_max_length False \
--output_dir /data/jcanete/evaluation/xquad \
--do_eval \
--do_predict \
--per_device_eval_batch_size 64 \
--logging_dir /data/jcanete/evaluation/xquad \
--seed 42 \
--cache_dir /data/jcanete/cache \
--use_auth_token True \
--fp16 \