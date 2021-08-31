# -*- coding: utf-8 -*-

"""Implementation of factory that create instances containing of triples and numeric literals.tsv."""

import logging
import pathlib
from typing import Dict, Optional, TextIO, Tuple, Union

import numpy as np
import torch

from .instances import MultimodalLCWAInstances, MultimodalSLCWAInstances
from .triples_factory import TriplesFactory
from .utils import load_external_embeddings
from ..typing import EntityMapping, LabeledTriples, MappedExternalEmbeddings

__all__ = [
    'TriplesExternalEmbeddingFactory',
]

logger = logging.getLogger(__name__)


def create_matrix_of_literals(
    external_embeddings: np.array,
    entity_to_id: EntityMapping,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Create matrix of literals where each row corresponds to an entity and each column to a literal."""
    
    
    
    data_relations = np.array([f"feat_{x}" for x in range(0, external_embeddings.shape[1]-1)])
    data_rel_to_id: Dict[str, int] = {
        value: key
        for key, value in enumerate(data_relations)
    }
    
    # Prepare literal matrix, set every literal to zero, and afterwards fill in the corresponding value if available
    num_literals = np.zeros([len(entity_to_id), len(data_rel_to_id)], dtype=np.float32)


    # TODO vectorize code
    for ent, *emb in external_embeddings:
        try:
            # row define entity, and column the literal. Set the corresponding literal for the entity
            num_literals[entity_to_id[ent]] = emb
        except KeyError:
            logger.info("Entity doesn't exist.")
            continue

    return num_literals, data_rel_to_id


class TriplesExternalEmbeddingFactory(TriplesFactory):
    """Create multi-modal instances given the path to triples."""

    def __init__(
        self,
        *,
        path: Union[None, str, pathlib.Path, TextIO] = None,
        triples: Optional[LabeledTriples] = None,
        path_to_external_embeddings: Union[None, str, pathlib.Path, TextIO] = None,
        external_embeddings: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Initialize the multi-modal triples factory.

        :param path: The path to a 3-column TSV file with triples in it. If not specified,
         you should specify ``triples``.
        :param triples:  A 3-column numpy array with triples in it. If not specified,
         you should specify ``path``
        :param path_to_numeric_triples: The path to a 3-column TSV file with triples and
         numeric. If not specified, you should specify ``numeric_triples``.
        :param numeric_triples:  A 3-column numpy array with numeric triples in it. If not
         specified, you should specify ``path_to_numeric_triples``.
        """
        if path is None:
            base = TriplesFactory.from_labeled_triples(triples=triples, **kwargs)
        else:
            base = TriplesFactory.from_path(path=path, **kwargs)
        super().__init__(
            entity_to_id=base.entity_to_id,
            relation_to_id=base.relation_to_id,
            mapped_triples=base.mapped_triples,
            create_inverse_triples=base.create_inverse_triples,
        )

        if path_to_external_embeddings is None and external_embeddings is None:
            raise ValueError('Must specify one of path_to_numeric_triples or numeric_triples')
        elif path_to_external_embeddings is not None and external_embeddings is not None:
            raise ValueError('Must not specify both path_to_numeric_triples and numeric_triples')
        elif path_to_external_embeddings is not None:
            external_embeddings = load_external_embeddings(path_to_external_embeddings)

        assert self.entity_to_id is not None
        self.external_embeddings, self.literals_to_id = create_matrix_of_literals(
            external_embeddings=external_embeddings,
            entity_to_id=self.entity_to_id,
        )

    def get_numeric_literals_tensor(self) -> torch.FloatTensor:
        """Return the numeric literals as a tensor."""
        return torch.as_tensor(self.external_embeddings, dtype=torch.float32)

    def extra_repr(self) -> str:  # noqa: D102
        return super().extra_repr() + (
            f"num_literals={len(self.literals_to_id)}"
        )

    def create_slcwa_instances(self) -> MultimodalSLCWAInstances:
        """Create multi-modal sLCWA instances for this factory's triples."""
        slcwa_instances = super().create_slcwa_instances()
        return MultimodalSLCWAInstances(
            mapped_triples=slcwa_instances.mapped_triples,
            numeric_literals=self.external_embeddings,
            literals_to_id=self.literals_to_id,
        )

    def create_lcwa_instances(self, use_tqdm: Optional[bool] = None) -> MultimodalLCWAInstances:
        """Create multi-modal LCWA instances for this factory's triples."""
        lcwa_instances = super().create_lcwa_instances(use_tqdm=use_tqdm)
        return MultimodalLCWAInstances(
            pairs=lcwa_instances.pairs,
            compressed=lcwa_instances.compressed,
            numeric_literals=self.external_embeddings,
            literals_to_id=self.literals_to_id,
        )
