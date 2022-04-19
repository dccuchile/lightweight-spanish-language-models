from transformers import AutoModelForPreTraining

model_name = "dccuchile/bert-base-spanish-wwm-uncased"
model = AutoModelForPreTraining.from_pretrained(model_name, use_auth_token=True)

total_params = sum(p.numel() for p in model.parameters())
print(total_params)