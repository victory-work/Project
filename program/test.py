from textattack import Attacker
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.transformations import WordSwap
from textattack.search_methods import greedy_word_swap_wir

model_wrapper = HuggingFaceModelWrapper("bert-base-uncased")

# Choose a transformation (e.g., WordSwap)
transformation = WordSwap()

# Choose a search method (e.g., GreedyWordSwap)
search_method = greedy_word_swap_wir()

# Create an attacker
attacker = Attacker(model_wrapper, transformation, search_method)

# Attack a text input
attack_result = attacker.attack("This is a sample sentence.")

# Analyze the results
print(attack_result)
