from enum import IntEnum


class ActionEnum(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    INTERACT = 5
    INIT_DIALOGUE = 6
    MOVE_TO_POS = 7
    NOOP = 8

class SubActionEnum(IntEnum):
    NONE = 0
    DEFAULT = 1
    PUSH = 2
    PICKUP_DROPOFF = 3