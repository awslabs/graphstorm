"""Embedding Cache Layer to hold all intermediate embeddings"""

from dgl.distributed import DistTensor

class EmbedCache():
    def __init__(self, embeds):
        """ We may need to put the cache into kvstore when run with distributed training

        TODO(xiangsx): extend cache to support distributed training
        """
        if isinstance(embeds, DistTensor) is False:
            # multigpu
            embeds.share_memory_()
        self._embeds = embeds

    def __getitem__(self, keys):
        return self._embeds[keys]

    def __setitem__(self, keys, items):
        self._embeds[keys] = items.detach().cpu()

    @property
    def embeds(self):
        return self._embeds
