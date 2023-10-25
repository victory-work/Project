from func_def import *
from textattack.attack_recipes import BERTAttackLi2020, PWWSRen2019
from textattack import Attacker
from textattack import AttackArgs

bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 128
batch_size = 64


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # target dataset

    csv_file = "double_quotes_attack.csv"
    attack_df = pd.read_csv(f"Project/{csv_file}")
    attack_texts = list(attack_df["text"])
    attack_labels = list(attack_df["label"])
    target_dataset = CustomTextClassificationDataset(
        attack_texts, attack_labels)

    # load model
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = BERTClassifier(bert_model_name, num_classes)
    model.load_state_dict(torch.load("BERT_model_state_uncased.pt"))

    # PWWSRen2019
    # attack method
    attack_recipe = PWWSRen2019.build(
        model_wrapper=BERTClassifierWrapper(model, tokenizer))
    attack_args = AttackArgs(
        num_examples=-1,
        log_to_csv="PWWSREN2019_result.csv",
        disable_stdout=True
    )
    attacker = Attacker(attack_recipe, target_dataset, attack_args=attack_args)

    # Run the attack on the target dataset
    try:
        attacker.attack_dataset()
    except Exception as error:
        print('\n Exception:', error)
