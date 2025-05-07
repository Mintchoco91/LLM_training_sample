from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 1. 학습된 모델과 토크나이저 로딩 (학습 시 저장된 output_dir 경로 사용)
model_path = "./gpt2-gen-model"  # <- 학습된 모델이 저장된 경로
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# 2. 프롬프트 설정
prompt = "Q: Who is kiwon?\nA:"

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask  # ✅ 이걸 추가해줘!


# 3. 텍스트 생성
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,       # ✅ 명시적으로 추가!
        max_length=100,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id  # ✅ 안정적으로 설정
    )

# 4. 출력 결과
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)