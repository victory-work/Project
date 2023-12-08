from func_def import *

# pay attention to where the model trained (GPU or CPU)

bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 128
batch_size = 64

if __name__ == "__main__":
    csv_file = "double_quotes_mali.csv"
    mali_df = pd.read_csv(f"Project/{csv_file}")
    mali_texts = list(mali_df["text"])
    mali_labels = list(mali_df["label"])

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = BERTClassifier(bert_model_name, num_classes)
    model.load_state_dict(torch.load("BERT_model_state_uncased.pt"))
    model.to(device)
    model.eval()

    successful_predictions = []

    for i in range(len(mali_texts)):
        text = mali_texts[i]
        label = mali_labels[i]

        inputs = tokenizer(text, padding='max_length', truncation=True,
                           max_length=max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        label = torch.tensor([label]).to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(output, dim=1)

            if preds.item() == label.item():
                successful_predictions.append(
                    {"text": text, "label": label.item()})
    lenOfTotal = len(mali_df)
    lenOfSuccessfulPrediction = len(successful_predictions)
    pred_list = [d["label"] for d in successful_predictions] + \
        [1] * (lenOfTotal-lenOfSuccessfulPrediction)
    accuracy = accuracy_score(pred_list, [0] * lenOfTotal)
    mse = mean_squared_error(pred_list, [0] * lenOfTotal)
    report = classification_report(pred_list, [0] * lenOfTotal)

    print(f"Accuracy: {accuracy}")
    print(f"mean_squared_error: {mse}")
    print(report)

    # Create a DataFrame from the successful predictions
    successful_predictions_df = pd.DataFrame(successful_predictions)
    successful_predictions_df.to_csv(target_file)
    print(successful_predictions_df)
