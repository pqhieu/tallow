from __future__ import annotations

from typing import Any, Callable, Generic, ParamSpec, TypeVar

import attrs

P = ParamSpec("P")
T = TypeVar("T")

_REGISTRY: dict[str, Callable[..., Any]] = {}


@attrs.define(frozen=True, kw_only=True)
class Config(Generic[T, P]):
    callable: Callable[P, T]
    args: tuple
    kwargs: dict

    def build(self) -> T:
        args = tuple(_build_if_config(a) for a in self.args)
        kwargs = {k: _build_if_config(v) for k, v in self.kwargs.items()}
        return self.callable(*args, **kwargs)


def configurable(cls: Callable[..., Any]) -> Callable[..., Any]:
    key = f"{cls.__module__}.{cls.__qualname__}"
    _REGISTRY[key] = cls
    return cls


def make_config(callable: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Config[T, P]:
    return Config(callable=callable, args=tuple(args), kwargs=dict(kwargs))


def _build_if_config(value: Any) -> Any:
    return value.build() if isinstance(value, Config) else value


if __name__ == "__main__":
    import torch.nn as nn

    @configurable
    class MyModel(nn.Module):
        def __init__(self, encoder: nn.Module, hidden_size: int):
            super().__init__()
            self.encoder = encoder
            self.proj = nn.Linear(hidden_size, hidden_size)

        def forward(self, x):
            return self.proj(self.encoder(x))

    configurable(nn.TransformerEncoderLayer)

    config = make_config(
        MyModel,
        encoder=make_config(nn.TransformerEncoderLayer, d_model=512, nhead=8),
        hidden_size=512,
    )

    # Build
    model = config.build()
    print(model)
