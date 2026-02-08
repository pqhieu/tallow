from copy import deepcopy
from typing import Any, Callable


class DictTransform:
    def __init__(
        self,
        fn: Callable,
        in_keys: list[str] | dict[str, str] | str,
        out_keys: list[str],
        inplace: bool = True,
    ):
        self.fn = fn
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.inplace = inplace

    def __call__(self, input: dict[str, Any]) -> dict[str, Any]:
        args: tuple[Any, ...] = tuple()
        kwargs: dict[str, Any] = dict()
        if isinstance(self.in_keys, list):
            args = tuple(input[k] for k in self.in_keys)
        elif isinstance(self.in_keys, dict):
            kwargs = {v: input[k] for k, v in self.in_keys.items()}
        elif isinstance(self.in_keys, str):
            args = (input[self.in_keys],)

        output = self.fn(*args, **kwargs)

        if isinstance(output, dict):
            output = tuple(output[key] for key in self.out_keys)
        if not isinstance(output, tuple):
            output = (output,)

        result = input if self.inplace else deepcopy(input)
        result.update(zip(self.out_keys, output))
        return result


if __name__ == "__main__":

    def test_fn(x, y):
        return x + y

    transform = DictTransform(test_fn, {"a": "x", "b": "y"}, ["a"])
    input = {"a": 1, "b": 2}
    output = transform(input)
    print(output)
