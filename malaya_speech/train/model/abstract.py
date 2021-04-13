# https://github.com/pytorch/fairseq/blob/master/fairseq/dataclass/configs.py

from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional


@dataclass
class Dataclass:
    """base dataclass that supported fetching attributes and metas"""

    _name: Optional[str] = None

    @staticmethod
    def name():
        return None

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(
        self, attribute_name: str, meta: str, default: Optional[Any] = None
    ) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(
            meta, default
        )

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith('${'):
                return str(getattr(self, attribute_name))
            elif str(
                self.__dataclass_fields__[attribute_name].default
            ).startswith('${'):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif (
                getattr(self, attribute_name)
                != self.__dataclass_fields__[attribute_name].default
            ):
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, 'help')

    def _get_argparse_const(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, 'argparse_const')

    def _get_argparse_alias(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, 'argparse_alias')

    def _get_choices(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, 'choices')
