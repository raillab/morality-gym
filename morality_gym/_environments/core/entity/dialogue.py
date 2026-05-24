from enum import IntEnum

from morality_gym._environments.core.entity.base import BaseEntity


class DialogueEntity(BaseEntity):
    def __init__(
            self,
            name: str,
            group: str,
            question: str,
            responses,  # TODO
    ):
        super().__init__(name, group)

        self._is_active: bool = False

        self._question = question
        self._responses = responses


#################
# DIALOGUE TYPE #
#################
class DialogueType(IntEnum):
    TRUTH = 0
    LIE = 1
    KIND = 2
    NEUTRAL = 3
    MEAN = 4
#################