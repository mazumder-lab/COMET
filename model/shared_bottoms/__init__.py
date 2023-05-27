from .empty_bottom import EmptyBottom
from .dense_bottom import DenseBottom
from .dense_bottom_with_concatenated_embeddings import DenseBottomWithConcatenatedEmbeddings
from .dense_bottom_with_concatenated_embeddings_no_dense import DenseBottomWithConcatenatedEmbeddingsNoDense
# from .bert_base_uncased_bottom import BERTBaseUncasedBottom


BottomsMapper = {
    "DenseBottom": DenseBottom,
    "DenseBottomWithConcatenatedEmbeddings": DenseBottomWithConcatenatedEmbeddings,
    "DenseBottomWithConcatenatedEmbeddingsNoDense": DenseBottomWithConcatenatedEmbeddingsNoDense,
    "EmptyBottom": EmptyBottom,
#     "BERTBaseUncasedBottom": BERTBaseUncasedBottom,
}