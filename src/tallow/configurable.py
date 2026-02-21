import dataclasses
import inspect
from typing import Any, TypeVar

import torch.nn as nn

T = TypeVar("T")


def _field_from_signature_param(
    name: str, param: inspect.Parameter
) -> tuple[str, Any] | tuple[str, Any, Any]:
    annotation = Any if param.annotation is inspect.Parameter.empty else param.annotation
    if param.default is not inspect.Parameter.empty:
        return (name, annotation, param.default)
    return (name, annotation)


def config_class(cls: type[T]) -> type[T]:
    signature = inspect.signature(cls.__init__)
    fields: list[tuple[str, Any] | tuple[str, Any, Any]] = [
        _field_from_signature_param(name, param)
        for name, param in signature.parameters.items()
        if name != "self"
    ]
    cls_name = f"config_class({cls.__module__}.{cls.__qualname__}"
    return dataclasses.make_dataclass(cls_name, fields)


if __name__ == "__main__":

    class SmallNet(nn.Module):
        def __init__(self, a: int, b, c: list[int] | None = None, d: int = 4):
            super().__init__()

        def forward(self, x):
            return x

    print(config_class(SmallNet))
