from func_def import *

# pay attention to where the model trained (GPU or CPU)

bert_model_name = 'bert-base-cased'
num_classes = 2
max_length = 128
batch_size = 64

if __name__ == "__main__":
    mali_df = pd.read_csv("adversary_dataset.csv")
    mali_texts = list(mali_df["text"])
    mali_labels = list(mali_df["label"])

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    mali_dataset = TextClassificationDataset(
        mali_texts, mali_labels, tokenizer, max_length)

    mali_dataloader = DataLoader(mali_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = BERTClassifier(bert_model_name, num_classes)
    model.load_state_dict(torch.load(
        "BERT_model_state_uncased.pt"), map_location=device)

    accuracy, report, mse = evaluate(model, mali_dataloader, device)
    print(f"Accuracy: {accuracy}")
    print(f"mean_squared_error: {mse}")
    print(report)
