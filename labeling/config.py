from dataclasses import dataclass

@dataclass
class Config:
    file: str = 'car_original'
    OUTER_STEP          = 0.25
    OUTER_CANDIDATE_POOL = 100
    MIN_LABEL_DIST = 1.0
    MAX_LABEL_DIST      = 3.0
    ITERATIVE_HUNGARIAN_MAX_ITERS = 50
    ITER_PENALTY_MULTIPLIER       = 2.0
    GRID_STEP = 0.5