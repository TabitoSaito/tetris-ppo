import random
from enum import Enum


class Identifier(Enum):
    NOTHING = 0
    OBSTACLE = -1
    PAINTED = 1


def generate_unique_coordinates(
    n, upper_bound_x, upper_bound_y, lower_bound_x=0, lower_bound_y=0, except_=[], rng=None
):
    cords = []
    gen = random.randint if rng is None else rng

    for _ in range(n):
        while True:
            x = gen(lower_bound_x, upper_bound_x)
            y = gen(lower_bound_y, upper_bound_y)

            cord = [x, y]
            if cord in cords:
                continue
            elif cord[0] == except_[0] and cord[1] == except_[1]:
                continue
            else:
                cords.append(cord)
                break
    return list(map(list, zip(*cords)))