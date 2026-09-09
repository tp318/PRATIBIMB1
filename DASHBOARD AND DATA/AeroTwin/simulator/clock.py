"""
AeroTwin-4 Simulation Clock.

Manages independent simulation time progression separated from wall-clock time.
"""


class SimulationClock:
    """
    Independent simulation clock tracking simulation time, step count, and step size.
    """

    def __init__(self, dt: float = 0.01, initial_time: float = 0.0):
        self._dt = float(dt)
        self._initial_time = float(initial_time)
        self._simulation_time = self._initial_time
        self._step_count = 0

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, val: float):
        if val <= 0:
            raise ValueError("Time step dt must be positive.")
        self._dt = float(val)

    @property
    def simulation_time(self) -> float:
        return self._simulation_time

    @property
    def step_count(self) -> int:
        return self._step_count

    def step(self) -> float:
        """
        Advance simulation clock by one time step dt.

        Returns
        -------
        float
            New simulation time in seconds.
        """
        self._step_count += 1
        self._simulation_time = self._initial_time + self._step_count * self._dt
        return self._simulation_time

    def reset(self, initial_time: float = None):
        """
        Reset simulation clock to initial state.
        """
        if initial_time is not None:
            self._initial_time = float(initial_time)
        self._simulation_time = self._initial_time
        self._step_count = 0

    def set_time(self, t: float):
        """
        Manually set simulation time.
        """
        self._simulation_time = float(t)
        self._step_count = int(round((self._simulation_time - self._initial_time) / self._dt))
