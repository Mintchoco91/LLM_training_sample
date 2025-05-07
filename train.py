from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from transformers.trainer_utils import IntervalStrategy

# 1. 토크나이저와 모델 불러오기
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 2. 데이터셋 로딩 (단순한 txt 파일 사용)
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=64
    )

train_dataset = load_dataset("train.txt", tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 3. 학습 설정
training_args = TrainingArguments(
    output_dir="./gpt2-gen-model",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir="./logs",
    logging_strategy=IntervalStrategy.STEPS,
    logging_steps=1,         # 🔴 로그 자주 찍기
    save_steps=10,            # 🔴 체크포인트 자주 저장
)

# 4. Trainer 객체로 학습 시작
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model("./GPT2-GEN-MODEL/final")  # ✅ 이걸 추가하면 final/ 폴더에 최종본 저장
tokenizer.save_pretrained("./GPT2-GEN-MODEL/final")

print("✅ 학습 완료")
exit()