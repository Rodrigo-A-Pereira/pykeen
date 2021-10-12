# -*- coding: utf-8 -*-

"""Implementation of combinations for the :class:`pykeen.models.LiteralModel`."""

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

import torch
from torch import nn

from class_resolver import HintOrType
from ..utils import activation_resolver, combine_complex, split_complex

__all__ = [
    'Combination',
    'ParameterizedRealCombination',
    'ComplexCombination',
    'ParameterizedComplexCombination',
    # Concrete classes
    'LinearDropout',
    'DistMultCombination',
    'ComplExLiteralCombination',
    'GatedCombination',
]


class Combination(nn.Module, ABC):
    """Base class for combinations."""

    def forward(self, x: torch.FloatTensor, literal: torch.FloatTensor) -> torch.FloatTensor:
        """Combine the representation and literal then score."""
        raise NotImplementedError


class ParametrizedCombination(Combination):
    """A combination based on wrapping a module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.FloatTensor, literal: torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined entity representation and literals baes on the given gating mechanism."""
        return self.module(x, literal)


class ParameterizedRealCombination(Combination):
    """A real combination parametrized by a scoring module."""

    def __init__(self, module: nn.Module):
        """Initialize the parameterized real combination.

        :param module: The module used to score the combination of the entity representation and literals.
        """
        super().__init__()
        self.module = module

    def forward(self, x: torch.FloatTensor,  literal: torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined entity representation and literals with the parameterized module."""
        return self.module(torch.cat([x, literal], dim=-1))


class ComplexCombination(Combination, ABC):
    """A mid-level base class for combinations of complex-valued vectors."""

    def forward(self, x: torch.FloatTensor, literal: torch.FloatTensor) -> torch.FloatTensor:
        """Split the complex vector, combine the representation parts and literal, score, then recombine."""
        x_re, x_im = split_complex(x)
        x_re = self.score_real(torch.cat([x_re, literal], dim=-1))
        x_im = self.score_imag(torch.cat([x_im, literal], dim=-1))
        return combine_complex(x_re=x_re, x_im=x_im)

    @abstractmethod
    def score_real(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined real part of the entity representation and literals."""
        raise NotImplementedError

    @abstractmethod
    def score_imag(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined imaginary part of the entity representation and literals."""
        raise NotImplementedError


class ParameterizedComplexCombination(ComplexCombination):
    """A complex combination parametrized by the real scoring module and imaginary soring module."""

    def __init__(self, real_module: nn.Module, imag_module: nn.Module):
        """Initialize the parameterized complex combination.

        :param real_module: The module used to score the combination of the real part of the entity representation
            and literals.
        :param imag_module: The module used to score the combination of the imaginary part of the entity
            representation and literals.
        """
        super().__init__()
        self.real_mod = real_module
        self.imag_mod = imag_module

    def score_real(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined real part of the entity representation and literals with the parameterized module."""
        return self.real_mod(x)

    def score_imag(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined imaginary part of the entity representation and literals with the parameterized module."""
        return self.imag_mod(x)


class LinearDropout(nn.Sequential):
    """A sequential module that has a linear layer, dropout later, and optional activation layer."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
        activation: HintOrType[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Instantiate the :class:`torch.nn.Sequential`.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.
        :param activation: An optional, pre-instantiated activation module, like :class:`torch.nn.Tanh`.
        """
        linear = nn.Linear(entity_embedding_dim + literal_embedding_dim, entity_embedding_dim)
        dropout = nn.Dropout(input_dropout)
        if activation:
            activation_instance = activation_resolver.make(activation, activation_kwargs)
            super().__init__(linear, dropout, activation_instance)
        else:
            super().__init__(linear, dropout)


class DistMultCombination(ParameterizedRealCombination):
    """The linear/dropout combination used in :class:`pykeen.models.DistMultLiteral`."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
    ) -> None:
        """Instantiate the :class:`ParameterizedRealCombination` with a :class:`LinearDropout`.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.

        This class does not use an activation in the :class:`LinearDropout` as
        described by [kristiadi2018]_.
        """
        super().__init__(LinearDropout(
            entity_embedding_dim=entity_embedding_dim,
            literal_embedding_dim=literal_embedding_dim,
            input_dropout=input_dropout,
        ))


class ComplExLiteralCombination(ParameterizedComplexCombination):
    """The linear/dropout/tanh combination used in :class:`pykeen.models.ComplExLiteral`."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
        activation: HintOrType[nn.Module] = 'tanh',
    ) -> None:
        """Instantiate the :class:`ParameterizedComplexCombination` with a :class:`LinearDropout` for real and complex.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.
        :param activation: The activation function, resolved by :data:`pykeen.utils.activation_resolver`.

        This class uses a :class:`torch.nn.Tanh` by default for the activation to the :class:`LinearDropout` as
        described by [kristiadi2018]_.
        """
        super().__init__(
            real_module=LinearDropout(
                entity_embedding_dim=entity_embedding_dim,
                literal_embedding_dim=literal_embedding_dim,
                input_dropout=input_dropout,
                activation=activation,
            ),
            imag_module=LinearDropout(
                entity_embedding_dim=entity_embedding_dim,
                literal_embedding_dim=literal_embedding_dim,
                input_dropout=input_dropout,
                activation=activation,
            ),
        )


class GatedCombination(Combination):
    """A module that implements a gated linear transformation for the combination of entities and literals.

    Implementation based on https://github.com/SmartDataAnalytics/LiteralE/blob/master/model.py Gate class.
    """

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        activation: HintOrType[nn.Module] = 'sigmoid',
        activation_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Instantiate the :class:`torch.nn.Module`.

        :param entity_embedding_dim: The dimension of the entity representations.
        :param literal_embedding_dim: The dimension of the literals.
        :param activation: An optional, pre-instantiated activation module, like :class:`torch.nn.sigmoid`.
        """
        super().__init__()
        self.gate_activation = activation_resolver.make(activation, activation_kwargs)
        self.combination_linear_layer = nn.Linear(
            entity_embedding_dim + literal_embedding_dim, entity_embedding_dim,
        )
        self.gate_entity_layer = nn.Linear(
            entity_embedding_dim, entity_embedding_dim,
        )
        self.gate_literal_layer = nn.Linear(
            literal_embedding_dim, entity_embedding_dim,
        )
        self.bias = nn.Parameter(torch.zeros(entity_embedding_dim))
        self.hyperbolic_tangent = torch.nn.Tanh()

    def forward(self, x: torch.FloatTensor, literal: torch.FloatTensor) -> torch.FloatTensor:
        """Given the entity and literal representations calucaltes a new embedding that contains information of both."""
        combination = torch.cat([x, literal], -1)
        z = self.gate_activation(self.gate_entity_layer(x) + self.gate_literal_layer(literal) + self.bias)
        h = self.hyperbolic_tangent(self.combination_linear_layer(combination))
        return z * h + (1 - z) * x
