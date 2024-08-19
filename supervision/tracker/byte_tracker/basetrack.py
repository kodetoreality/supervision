from collections import OrderedDict
from enum import Enum
from typing import NoReturn

import numpy as np


class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0

    def __init__(self) -> None:
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New

        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0

        # multi-camera
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_counter() -> None:
        BaseTrack._count = 0
        BaseTrack.track_id = 0
        BaseTrack.start_frame = 0
        BaseTrack.frame_id = 0
        BaseTrack.time_since_update = 0

    def activate(self, *args) -> NoReturn:
        raise NotImplementedError

    def predict(self) -> NoReturn:
        raise NotImplementedError

    def update(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed
