from enum import IntEnum


class DangerState(IntEnum):
    # Human Harmed
    HumanHarm = 0
    MajorHumanHarm = 1
    MajorHumanHarmByRobot = 2
    MajorHumanHarmByEnv = 3
    MinorHumanHarm = 4
    MinorHumanHarmByRobot = 5
    MinorHumanHarmByEnv = 6

    # Property Harmed
    PropertyHarm = 7
    MajorPropertyHarm = 8
    MinorPropertyHarm = 9

    # Robot Harmed
    RobotHarm = 10
    MajorRobotHarm = 11
    MinorRobotHarm = 12

    # def str_to_enum(self, val: str) -> IntEnum:
    #     raise NotImplementedError
