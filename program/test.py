from func_def import *
import pandas as pd
import textattack

csv_path = "double_quotes_train.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_path)
print(df.shape)

##### 問題...

# Split the dataset into training and evaluation sets
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
# train_dataset = textattack.datasets.HuggingFaceDataset(df, split="train")
# eval_dataset = textattack.datasets.HuggingFaceDataset(df, split="test")

# # Create TextAttack datasets
train_texts, train_labels = train_df['text'].tolist(), train_df['label'].tolist()
eval_texts, eval_labels = eval_df['text'].tolist(), eval_df['label'].tolist()

train_dataset = textattack.datasets.Dataset(train_texts, train_labels)
eval_dataset = textattack.datasets.Dataset(eval_texts, eval_labels)
print(train_dataset)
