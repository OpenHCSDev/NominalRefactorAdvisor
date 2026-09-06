"""Shared source fixtures for registry discovery and native identity controls."""


def keyed_registry_source() -> str:
    return """

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import ClassVar, Generic, TypeVar


KeyT = TypeVar("KeyT")


class AutoRegisterByClassVar:
    registry_key_attr: ClassVar[str]
    _registry: ClassVar[dict[object, object]]


class KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]):
    pass


class Mode(Enum):
    ALPHA = auto()
    BETA = auto()


class ModeRunner(KeyedNominalFamily[Mode], ABC):
    registry_key_attr = "mode"
    _registry = {}
    mode: ClassVar[Mode]

    @classmethod
    def for_mode(cls, mode: Mode):
        return cls._registry[mode]

    @abstractmethod
    def run(self):
        raise NotImplementedError


class AlphaModeRunner(ModeRunner):
    mode = Mode.ALPHA

    def run(self):
        return "alpha"


class BetaModeRunner(ModeRunner):
    mode = Mode.BETA

    def run(self):
        return "beta"


def run_alpha():
    return ModeRunner.for_mode(Mode.ALPHA).run()


def run_beta():
    return ModeRunner.for_mode(Mode.BETA).run()

ModeRunner._registry[Mode.ALPHA] = AlphaModeRunner()
ModeRunner._registry[Mode.BETA] = BetaModeRunner()
"""


def _type_keyed_behavior_projection_source() -> str:
    return """
from abc import ABC, abstractmethod
from typing import ClassVar
from metaclass_registry import AutoRegisterMeta
from nominal_refactor_advisor.registry_identity import mro_registry_value


class Event:
    value: str


class NamedEvent(Event):
    name: str


class CountedEvent(Event):
    count: int


class EventProjection(ABC, metaclass=AutoRegisterMeta):
    __registry__: ClassVar[dict[type[Event], type["EventProjection"]]] = {}
    __registry_key__ = "event_type"
    __skip_if_no_key__ = True
    event_type: ClassVar[type[Event]]

    @abstractmethod
    def render(self, event: Event) -> str:
        raise NotImplementedError

    @classmethod
    def projection_for(cls, event: Event):
        projection_type = mro_registry_value(cls.__registry__, type(event))
        return projection_type() if projection_type is not None else None

    @classmethod
    def render_for(cls, event: Event) -> str:
        projection = cls.projection_for(event)
        if projection is None:
            return ""
        return projection.render(event)


class NamedEventProjection(EventProjection):
    event_type = NamedEvent

    def render(self, event: Event) -> str:
        return event.name


class CountedEventProjection(EventProjection):
    event_type = CountedEvent

    def render(self, event: Event) -> str:
        return str(event.count)


class FallbackEventProjection(EventProjection):
    event_type = Event

    def render(self, event: Event) -> str:
        return event.value


def render_event(event: Event) -> str:
    return EventProjection.render_for(event)


def render_event_locally(event: Event) -> str:
    projection = EventProjection.projection_for(event)
    if projection is None:
        return ""
    return projection.render(event)
"""
