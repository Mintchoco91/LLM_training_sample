from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 학습한 모델 경로
model_path = "./gpt2-gen-model/final"  # 또는 checkpoint 경로

# 모델과 토크나이저 불러오기
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model.eval()

# 입력 프롬프트
prompt = "Q: Who is kiwon?:"

# 토크나이즈
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]  # ✅ 경고 없애기 위해 이 줄 추가

# 텍스트 생성
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        max_length=30,
        do_sample=False,  # → 무작위성 제거: 항상 가장 확률 높은 토큰 선택
        pad_token_id=tokenizer.eos_token_id,
    )

# 결과 디코딩
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)