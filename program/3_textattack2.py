from func_def import *
from textattack import Attack
from textattack.search_methods import GreedySearch
from textattack.goal_functions import UntargetedClassification
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.pre_transformation import RepeatModification
from textattack.constraints.pre_transformation import StopwordModification

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

    model_wrapper = BERTClassifierWrapper(model, tokenizer)
    # We'll use untargeted classification as the goal function.
    goal_function = UntargetedClassification(model_wrapper)
    # We'll to use our WordSwapEmbedding as the attack transformation.
    transformation = WordSwapEmbedding()
    # We'll constrain modification of already modified indices and stopwords
    constraints = [RepeatModification(), StopwordModification()]
    # We'll use the Greedy search method
    search_method = GreedySearch()
    # Now, let's make the attack from the 4 components:
    attack = Attack(goal_function, constraints, transformation, search_method)

    num = len(attack_texts)
    for i in range(num):
        print(f'{i+1}/{num}')
        result = attack.attack(attack_texts[i], attack_labels[i])
        print(result.__str__(color_method="ansi"))
