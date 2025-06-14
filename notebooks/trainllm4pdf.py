import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
import pymupdf
fitz=pymupdf
# Paths
books_dir = "/content/drive/MyDrive/cybersecurity_books/"
model_path = "/content/drive/MyDrive/LEWIS-TinyLlama-merged"
output_base_path = "/content/drive/MyDrive/LEWIS-Checkpoint"

# Load tokenizer and model once
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# Tokenize function
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Function to create dataset from PDF text
def create_dataset_from_pdf(pdf_path, max_samples=200):
    all_text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        all_text += page.get_text() + "\n"
    doc.close()

    paragraphs = [p.strip() for p in all_text.split("\n\n") if len(p.strip()) > 60][:max_samples]
    qa_pairs = [{"text": f"<s>[INST] Explain this: [/INST] {p} </s>"} for p in paragraphs]
    return Dataset.from_list(qa_pairs)

# Loop through each PDF and fine-tune one by one
for i, filename in enumerate(sorted(os.listdir(books_dir))):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(books_dir, filename)
        print(f"\n📘 Training on: {filename}")

        # Step 1: Create dataset from this PDF
        dataset = create_dataset_from_pdf(pdf_path)
        tokenized_dataset = dataset.map(tokenize)

        # Step 2: Define training args
        training_args = TrainingArguments(
            output_dir=f"{output_base_path}_{i}",
            num_train_epochs=1,  # 1 epoch per PDF is usually enough
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            save_strategy="no",  # Save manually below
            logging_dir="./logs",
            logging_steps=10
        )

        # Step 3: Fine-tune model on current PDF
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=tokenized_dataset
        )
        # Explicitly cast model parameters to float32 before training to avoid FP16 gradient issues
        for param in model.parameters():
            if param.dtype == torch.float16:
                param.data = param.data.float()
                if param.grad is not None:
                    param.grad.data = param.grad.data.float()

        trainer.train()

        # Step 4: Save intermediate checkpoint
        model.save_pretrained(f"{output_base_path}_step{i}")
        tokenizer.save_pretrained(f"{output_base_path}_step{i}")
        print(f"✅ Saved checkpoint after training on: {filename}")

print("\n🎉 All PDFs processed and model saved after each.")
