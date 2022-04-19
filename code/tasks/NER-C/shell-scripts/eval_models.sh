export CUDA_VISIBLE_DEVICES=0

python run_ner.py \
  --model_name_or_path CenIA/bert-base-spanish-wwm-cased-finetuned-ner \
  --max_seq_length 512 \
  --pad_to_max_length False \
  --do_lower_case False \
  --output_dir /data/jcanete/evaluation/ner \
  --use_fast_tokenizer  \
  --do_predict \
  --per_device_eval_batch_size 64 \
  --logging_dir /data/jcanete/evaluation/ner \
  --seed 42 \
  --fp16 \
  --cache_dir /data/jcanete/cache \
  --use_auth_token True \
;   