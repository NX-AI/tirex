# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from dataclasses import fields


def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
    return int(((x + multiple_of - 1) // multiple_of) * multiple_of)


def dataclass_from_dict(cls, dict: dict):
    class_fields = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in dict.items() if k in class_fields})
