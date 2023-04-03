from collections import abc
from typing import Any


class Config:
    """ object providing read-only access to configurations """

    def __new__(cls, arg: abc.MutableSequence | Any) -> 'Config' | list['Config'] | Any:
        if isinstance(arg, abc.Mapping):
            return super().__new__(cls)
        elif isinstance(arg, abc.MutableSequence):
            return [cls(item) for item in arg]
        else:
            return arg

    @staticmethod
    def convert_type(value: str):
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        return value

    def convert(self, d: dict) -> dict:
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = self.convert(value)
            elif isinstance(value, str):
                d[key] = self.convert_type(value)
        return d

    def __init__(self, mapping) -> None:
        self.__data = self.convert(dict(mapping))

    def __getattr__(self, name):
        if hasattr(self.__data, name):
            return getattr(self.__data, name)
        else:
            return Config(self.__data[name])
