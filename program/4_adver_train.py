from func_def import *

import pandas as pd
from textattack.models.wrappers import BertModelWrapper
from textattack.attack_recipes import BAE
from textattack.datasets import TextCSVDataset

def load_csv_dataset(csv_path):
    # Load CSV dataset using pandas
    df = pd.read_csv(csv_path)
    
    # Assuming your CSV has 'text' and 'label' columns
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    return texts, labels

def main():
    # Step 1: Load BERT model
    model_wrapper = BertModelWrapper(model_name="bert-base-uncased")

    # Step 2: Load the CSV dataset
    csv_path = "double_quotes_train.csv"  # Replace with your CSV file path
    texts, labels = load_csv_dataset(csv_path)
    
    # Step 3: Initialize the attack recipe
    recipe = BAE.build(model_wrapper)

    # Step 4: Configure the attack
    recipe = BAE.build(model_wrapper, transformations_per_example=4, max_candidates=10, max_attempts=3)

    # Step 5: Train the adversarial model
    model_wrapper.train((texts, labels), recipe, num_epochs=3)

    # Step 6: Evaluate the adversarial model
    accuracy = model_wrapper.eval((texts, labels))
    print(f"Adversarial Model Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
