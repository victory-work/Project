from func_def import *


bert_model_name = 'bert-base-cased'
num_classes = 2
max_length = 128
batch_size = 64
num_epochs = 20
learning_rate = 2e-5

setup_seed(20)

if __name__ == "__main__":
    df = pd.read_csv("train_dataset.csv")
    cert_texts = list(df["text"])
    cert_labels = list(df["label"])

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(cert_texts, cert_labels, shuffle=True,
                                                                          test_size=0.4, random_state=42)
    print(len(train_texts))
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length)
    eval_dataset = TextClassificationDataset(
        eval_texts, eval_labels, tokenizer, max_length)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier(bert_model_name, num_classes).to(device)

    # AdamW is a class from the huggingface library (as opposed to pytorch)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * num_epochs
    # Create the learning rate scheduler. This changes the learning rate as the training loop progresses
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    accuracy, report, mse = evaluate(model, eval_dataloader, device)
