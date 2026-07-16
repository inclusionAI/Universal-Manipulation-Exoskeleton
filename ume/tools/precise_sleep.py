import time


def precise_sleep(dt: float, slack_time: float = 0.001, time_func=time.monotonic):
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return


class FrequencyRegulator:
    def __init__(self, frequency: float, time_func=time.time):
        self.frequency = frequency
        self.time_func = time_func
        self.iter_idx = 0
        self.start_time = None
        self.last_sleep_time = None
        self.dt = 1 / frequency

    def sleep(self, verbose=False, verbose_interval=1, verbose_prefix=""):
        if self.start_time is None:
            self.start_time = self.time_func()
        else:
            self.iter_idx += 1
            t_target = self.start_time + self.dt * self.iter_idx
            precise_wait(t_target, time_func=self.time_func)
            if self.last_sleep_time is not None:
                actual_dt = self.time_func() - self.last_sleep_time
                if verbose and self.iter_idx % (self.frequency * verbose_interval) == 0:
                    print(f"{verbose_prefix} loop frequency: {1/actual_dt:.2f} Hz (target: {self.frequency} Hz)")
            self.last_sleep_time = self.time_func()
