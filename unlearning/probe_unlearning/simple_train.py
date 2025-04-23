# train_model.py
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

print(f"imported")
# Step 1: Load dataset
dataset = load_dataset("imdb")

# Optional: rename column if needed
dataset = dataset.rename_column("text", "input_text")
print(f"dataset: {dataset}")

# Create a validation split from the training set
dataset = dataset["train"].train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Step 2: Load tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print( f"tokenizer: {tokenizer}")
# Step 3: Preprocess function
def preprocess(example):
    return tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=128)

# Apply preprocessing
train_dataset = train_dataset.map(preprocess, batched=True)
eval_dataset = eval_dataset.map(preprocess, batched=True)

# Remove columns that are not needed for the model
train_dataset = train_dataset.remove_columns(["input_text"])
eval_dataset = eval_dataset.remove_columns(["input_text"])

print(f"train_dataset: {train_dataset}")
# Set the dataset format to PyTorch
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

# Step 4: Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 5: Define metric and compute_metrics
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Step 6: Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_steps=100,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

# Step 7: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Step 8: Train
trainer.train()

# Step 9: Evaluate
test_result = trainer.evaluate(eval_dataset=eval_dataset)
print("Test accuracy: ", test_result["eval_accuracy"])

# Step 10: Save model
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
