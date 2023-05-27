from .no_permutations import NoPermutations
from .random_permutations import RandomPermutations
from .learn_permutations import LearnPermutations

PermutationsMapper = {
    "NoPermutations": NoPermutations,
    "RandomPermutations": RandomPermutations,
    "LearnPermutations": LearnPermutations,
}