from dataclasses import dataclass
import warnings

from beartype.typing import Optional
import jax
import jax.numpy as jnp
from jaxtyping import Num

from gpjax.typing import Array


@dataclass
@jax.tree_util.register_pytree_node_class
class Dataset_3D:
    r"""Base class for datasets.

    Args:
        X: input data.
        y: output data.
    """

    X: Optional[Num[Array, "N D"]] = None
    y: Optional[Num[Array, "N Q"]] = None

    def __repr__(self) -> str:
        r"""Returns a string representation of the dataset."""
        repr = f"Dataset(Number of observations: {self.n:=} - Input dimension: {self.in_dim})"
        return repr

    def is_supervised(self) -> bool:
        r"""Returns `True` if the dataset is supervised."""
        return self.X is not None and self.y is not None

    def is_unsupervised(self) -> bool:
        r"""Returns `True` if the dataset is unsupervised."""
        return self.X is None and self.y is not None

    def __add__(self, other: "Dataset") -> "Dataset":
        r"""Combine two datasets. Right hand dataset is stacked beneath the left."""
        X = None
        y = None

        if self.X is not None and other.X is not None:
            X = jnp.concatenate((self.X, other.X))

        if self.y is not None and other.y is not None:
            y = jnp.concatenate((self.y, other.y))

        return Dataset(X=X, y=y)

    @property
    def n(self) -> int:
        r"""Number of observations."""
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        r"""Dimension of the inputs, $X$."""
        return self.X.shape[1]

    def tree_flatten(self):
        return (self.X, self.y), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)



