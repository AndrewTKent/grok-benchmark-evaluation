def compute_composite_score(
    success: bool,
    steps_taken: int,
    time_taken_sec: float,
    recovery_score: float,
    safety_score: float,
    loop_score: float,
    max_steps: int = 20,
    max_time_sec: int = 300,
    w_success=0.4, w_eff=0.2, w_rec=0.2, w_safety=0.1, w_loop=0.1
) -> float:
    eff_steps = max(0.0, 1.0 - (steps_taken / max_steps))
    eff_time  = max(0.0, 1.0 - (time_taken_sec / max_time_sec))
    efficiency = (eff_steps + eff_time) / 2.0
    base_success = 1.0 if success else 0.0
    composite = (
        w_success*base_success +
        w_eff*efficiency +
        w_rec*recovery_score +
        w_safety*safety_score +
        w_loop*loop_score
    )
    return max(0.0, min(1.0, composite))
