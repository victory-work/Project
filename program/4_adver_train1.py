from func_def import *
import pandas as pd
import textattack

bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 128
batch_size = 64

model = BERTClassifier(bert_model_name, num_classes)
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model_wrapper = BERTClassifierWrapper(model, tokenizer)

# We only use DeepWordBugGao2018 to demonstration purposes.
attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
print(attack)
csv_path = "double_quotes_train.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_path)
print(df.shape)

# Split the dataset into training and evaluation sets
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = textattack.datasets.HuggingFaceDataset(df, split="train")
eval_dataset = textattack.datasets.HuggingFaceDataset(df, split="test")

train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)


# Train for 3 epochs with 1 initial clean epochs, 1000 adversarial examples per epoch, learning rate of 5e-5, and effective batch size of 32 (8x4).
training_args = textattack.TrainingArgs(
    num_epochs=3,
    num_clean_epochs=1,
    num_train_adv_examples=100,
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    log_to_tb=True,
)

trainer = textattack.Trainer(
    model_wrapper,
    "classification",
    attack,
    train_dataloader,
    eval_dataloader,
    training_args
)
trainer.train()