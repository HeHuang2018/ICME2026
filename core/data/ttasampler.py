import numpy as np
from torch.utils.data.sampler import Sampler
from .datasets.base_dataset import DatumBase
from typing import List, Optional
from collections import defaultdict
from numpy.random import dirichlet
import random
from loguru import logger as log
from collections import defaultdict, deque

class MixedDomainDirichletSampler(Sampler):
    def __init__(self, data_source, gamma, batch_size, slots=None):
        """
        Initialize a sampler that can mix samples across domains according to Dirichlet distribution.
        
        Parameters:
            data_source (list): List of DatumBase instances.
            gamma (float): Parameter for the Dirichlet distribution controlling label distribution within each domain.
            batch_size (int): Number of samples per batch.
            domain_mixture_alpha (float): Dirichlet parameter controlling domain mixture level. Higher means more mixed.
            slots (int): Number of slots in the batch for each class. Defaults to the number of classes.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.gamma = gamma

        # Organize data indices by domain and shuffle them
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
        
        # Convert lists of indices to deques and shuffle them
        self.domain_queues = {domain: deque(np.random.permutation(indices)) 
                              for domain, indices in self.domain_dict.items()}
        
        self.domains = sorted(list(self.domain_queues.keys()))

    def __iter__(self):
        final_indices = []
        # Generate domain mixture probabilities
        domain_mixture_dist = dirichlet([self.gamma] * len(self.domains))

        for _ in range(len(self.data_source) // self.batch_size):
            batch_indices = []
            # Sample a domain according to the mixture distribution
            domain_choices = np.random.choice(self.domains, size=self.batch_size, p=domain_mixture_dist)

            for domain in domain_choices:
                # Retrieve an index from the chosen domain without repetition
                if not self.domain_queues[domain]:  # If empty, refill and shuffle
                    self.domain_queues[domain] = deque(np.random.permutation(self.domain_dict[domain]))
                
                # Pop from the left to avoid repetition
                index = self.domain_queues[domain].popleft()
                batch_indices.append(index)

            # Add the indices for this batch to the final list
            final_indices.extend(batch_indices)

        return iter(final_indices)

    def __len__(self):
        return len(self.data_source)



# # New Sampler for simple global mixing
# class MixedDomainDirichletSampler(Sampler):
#     def __init__(self, data_source: List[DatumBase], gamma, batch_size, slots=None):
#         """
#         Sampler that gets all sample indices and shuffles them globally.
#         The gamma and slots parameters are kept for compatibility with build_sampler
#         but are not used in the sampling logic itself, as the goal is simple global shuffling.

#         Args:
#             data_source: List of DatumBase objects.
#             gamma: Parameter (unused in this sampler).
#             batch_size: Batch size (unused directly in __iter__ but used by DataLoader).
#             slots: Parameter (unused in this sampler).
#         """
#         self.data_source = data_source
#         self.batch_size = batch_size # Stored, but not used to yield batches

#     def __len__(self):
#         """Returns the total number of samples."""
#         return len(self.data_source)

#     def __iter__(self):
#         """
#         Generates a sequence of indices by shuffling all indices globally.
#         """
#         # 1. Get all possible indices from 0 to N-1, where N is the total number of samples
#         all_indices = list(range(len(self.data_source)))

#         # 2. Shuffle the list of all indices randomly
#         random.shuffle(all_indices)

#         # 3. Return an iterator over the shuffled indices
#         return iter(all_indices)


class LabelDirichletDomainSequence(Sampler):
    def __init__(self, data_source: List[DatumBase], gamma, batch_size, slots=None):

        self.domain_dict = defaultdict(list)
        self.classes = set()
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
            self.classes.add(item.label)
        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        self.data_source = data_source

        self.batch_size = batch_size
        self.num_class = len(self.classes)
        if slots is not None:
            self.num_slots = slots
        else:
            self.num_slots = self.num_class if self.num_class <= 100 else 100
        
        if gamma > 0:
            self.gamma = gamma
        else:
            self.gamma = random.uniform(0, self.num_slots)
            
    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        final_indices = []
        for domain in self.domains:
            indices = np.array(self.domain_dict[domain])
            labels = np.array([self.data_source[i].label for i in indices])

            class_indices = [np.argwhere(labels == y).flatten() for y in range(self.num_class)]
            slot_indices = [[] for _ in range(self.num_slots)]

            label_distribution = dirichlet([self.gamma] * self.num_slots, self.num_class)

            for c_ids, partition in zip(class_indices, label_distribution):
                for s, ids in enumerate(np.split(c_ids, (np.cumsum(partition)[:-1] * len(c_ids)).astype(int))):
                    slot_indices[s].append(ids)

            for s_ids in slot_indices:
                permutation = np.random.permutation(range(len(s_ids)))
                ids = []
                for i in permutation:
                    ids.extend(s_ids[i])
                final_indices.extend(indices[ids])

        return iter(final_indices)


def build_sampler(
        cfg,
        data_source: List[DatumBase],
        **kwargs
):
    if cfg.LOADER.SAMPLER.TYPE == "temporal":
        return LabelDirichletDomainSequence(data_source, cfg.LOADER.SAMPLER.GAMMA, cfg.TEST.BATCH_SIZE, **kwargs)
    elif cfg.LOADER.SAMPLER.TYPE == "mix":
        return MixedDomainDirichletSampler(data_source, cfg.LOADER.SAMPLER.GAMMA, cfg.TEST.BATCH_SIZE, **kwargs)
    else:
        raise NotImplementedError()
