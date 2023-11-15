from func_def import *
import pandas as pd
import textattack

csv_path = "double_quotes_train.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_path)
print(df.shape)

train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)


train_list = [(text, label) for text, label in zip(train_df["text"], train_df["label"])]
              
data = [("A man inspects the uniform of a figure in some East Asian country.", "The man is sleeping", 1)]
dataset = textattack.datasets.Dataset(data, input_columns=("premise", "hypothesis"))

# Example for seq2seq
data = [("J'aime le film.", "I love the movie.")]
dataset = textattack.datasets.Dataset(data)