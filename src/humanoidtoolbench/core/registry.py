"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from typing import ClassVar, Generic, Type, TypeVar

T = TypeVar("T")
_S = TypeVar("_S")


class RegistryMixin(Generic[T]):
    _registry: ClassVar[dict[str, Type[T]]] = {}  # type: ignore

    def __init_subclass__(cls, **kwargs) -> None:
        """Give every concrete registry its own registration table."""
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, uid: str):
        def wrapper(subclass: Type[_S]) -> Type[_S]:
            # if not issubclass(subclass, T):
            #     raise TypeError(f"{subclass.__name__} must inherit from {cls._base_type().__name__}")
            cls._registry[uid] = subclass  # type: ignore
            return subclass

        return wrapper

    @classmethod
    def make(cls, uid: str, *args, **kwargs) -> T:
        if uid not in cls._registry:
            raise ValueError(f"No class registered under uid '{uid}'")

        return cls._registry[uid](*args, **kwargs)
