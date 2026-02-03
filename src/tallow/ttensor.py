from typing import Any

import torch


class TTensor(torch.Tensor):
    def __init_subclass__(cls, /, channel_names: list[str], **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.channel_names = channel_names
        cls.num_channels = len(channel_names)

    def __new__(
        cls,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool = False,
    ):
        tensor = torch.as_tensor(data, dtype=dtype, device=device).requires_grad_(requires_grad)
        if tensor.shape[-1] != len(cls.channel_names):
            raise ValueError(f"Tensor shape {tensor.shape} does not match number of channels {cls.num_channels}")
        return tensor.as_subclass(cls)

    def __getattr__(self, name: str, /) -> Any:
        if name in self.channel_names:
            return self[..., self.channel_names.index(name)]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class Boxes3D(TTensor, channel_names=["x", "y", "z", "l", "w", "h", "yaw"]):
    pass


if __name__ == "__main__":
    data = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    boxes = Boxes3D(data)
    print(boxes[..., 0], boxes[..., 1], boxes[..., 2])
    print(boxes.x)
    boxes.d
