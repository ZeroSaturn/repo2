import numpy as np

class DoublePendulumEnv:
    """A simple double pendulum environment with torque control at the base."""
    def __init__(self, dt=0.05):
        self.dt = dt
        self.max_torque = 2.0
        self.reset()

    def reset(self):
        # Start near upright with small random tilt
        self.state = np.array([
            np.pi + np.random.uniform(-0.1, 0.1),
            0.0 + np.random.uniform(-0.1, 0.1),
            0.0,
            0.0
        ], dtype=np.float32)
        return self._get_obs()

    def step(self, action):
        action = float(np.clip(action, -self.max_torque, self.max_torque))
        th1, th2, dth1, dth2 = self.state
        m1 = m2 = 1.0
        l1 = l2 = 1.0
        g = 9.81

        d2th1 = (-g * (2 * m1 + m2) * np.sin(th1)
                 - m2 * g * np.sin(th1 - 2 * th2)
                 - 2 * np.sin(th1 - th2) * m2 * (dth2 ** 2 * l2 + dth1 ** 2 * l1 * np.cos(th1 - th2))
                 + action) / (l1 * (2 * m1 + m2 - m2 * np.cos(2 * th1 - 2 * th2)))

        d2th2 = (2 * np.sin(th1 - th2) * (dth1 ** 2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(th1) + dth2 ** 2 * l2 * m2 * np.cos(th1 - th2))) / (l2 * (2 * m1 + m2 - m2 * np.cos(2 * th1 - 2 * th2)))

        dth1 += d2th1 * self.dt
        dth2 += d2th2 * self.dt
        th1 += dth1 * self.dt
        th2 += dth2 * self.dt

        self.state = np.array([th1, th2, dth1, dth2], dtype=np.float32)
        reward = - (np.cos(th1) + np.cos(th1 + th2) - 2)
        done = False
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        th1, th2, dth1, dth2 = self.state
        return np.array([
            np.cos(th1), np.sin(th1),
            np.cos(th1 + th2), np.sin(th1 + th2),
            dth1, dth2
        ], dtype=np.float32)
