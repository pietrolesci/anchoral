import inspect
from types import ModuleType
from typing import Generator, List, Optional, Tuple, Type

import torch
import torch_optimizer
import transformers
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION


class _Registry(dict):
    def __call__(self, cls: Type, key: Optional[str] = None, override: bool = False) -> Type:
        """Registers a class mapped to a name.

        Args:
            cls: the class to be mapped.
            key: the name that identifies the provided class.
            override: Whether to override an existing key.
        """
        if key is None:
            key = cls.__name__
        elif not isinstance(key, str):
            raise TypeError(f"`key` must be a str, found {key}")

        if key not in self or override:
            self[key.lower()] = cls

        return cls

    def register_classes(
        self,
        module: ModuleType,
        base_cls: Type,
        override: bool = False,
    ) -> None:
        """This function is an utility to register all classes from a module."""
        for cls in self.get_members(module, base_cls):
            self(cls=cls, override=override)

    @staticmethod
    def get_members(module: ModuleType, base_cls: Type) -> Generator[Type, None, None]:
        return (
            cls
            for _, cls in inspect.getmembers(module, predicate=inspect.isclass)
            if issubclass(cls, base_cls) and cls != base_cls
        )

    @property
    def names(self) -> List[str]:
        """Returns the registered names."""
        return list(self.keys())

    @property
    def classes(self) -> Tuple[Type, ...]:
        """Returns the registered classes."""
        return tuple(self.values())

    def __str__(self) -> str:
        return f"Registered objects: {self.names}"

    def __getitem__(self, __key):
        return super().__getitem__(__key.lower())


# redefine this to have torch and transformers overwrite torch_optimizer
OPTIMIZER_REGISTRY = _Registry()
OPTIMIZER_REGISTRY.register_classes(torch_optimizer, torch.optim.Optimizer)
OPTIMIZER_REGISTRY.register_classes(transformers.optimization, torch.optim.Optimizer, override=True)
OPTIMIZER_REGISTRY.register_classes(torch.optim, torch.optim.Optimizer, override=True)


# add trasformers convenience functions
SCHEDULER_REGISTRY = _Registry()
SCHEDULER_REGISTRY.register_classes(torch.optim.lr_scheduler, torch.optim.lr_scheduler._LRScheduler)
SCHEDULER_REGISTRY.update({v.__name__[4:]: v for v in TYPE_TO_SCHEDULER_FUNCTION.values()})
