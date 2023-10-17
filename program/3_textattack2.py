from textattack import Attack
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset, Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.pre_transformation import RepeatModification
from textattack.transformations import WordSwapMaskedLM, WordSwapEmbedding

# recipe = TextFoolerJin2019.build(model="bert-base-uncased", use_cache=True)
# dataset = TextDataset(dataframe=my_dataframe, text_column="text", label_column="label")
# results = recipe.attack(dataset)
