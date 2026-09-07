"""Absolute-action RGB adapter around the pinned SWM simulator.

Physical poses/contacts stay in evaluation labels. The learned planner receives
only encoded RGB and its private recurrent activations, never this wrapper.
"""

import math
import os

import numpy as np

ENVIRONMENT_SCHEMA_VERSION = 'pusht-swm-6f1e499e-absolute10hz-rgb96-to64-block-pose-v1'


def _pose(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (5,) or not np.isfinite(value).all():
        raise ValueError('pose must have five finite world coordinates')
    return value.copy()


class PushTEnv:
    def __init__(self):
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
        from third_party.swm.pusht import PushT
        self.simulator = PushT(resolution=96, relative=False, with_target=True)
        self.primitive_steps = 0
        self._goal_pose = None
        self._hold_action = None

    @property
    def pose(self):
        """Privileged actual pose5; a copy solely for harness labels/metrics."""
        return np.asarray(self.simulator._get_obs()[:5], dtype=np.float64).copy()

    @property
    def hold_action(self):
        """Privileged baseline targets the actual initial pusher; zero is origin."""
        if self._hold_action is None:
            raise RuntimeError('reset before requesting hold action')
        return self._hold_action.copy()

    def reset(self, pose5_world=None, goal_pose5_world=None, seed=0):
        options = {}
        if pose5_world is not None:
            options['state'] = np.r_[_pose(pose5_world), [0., 0.]]
        goal = _pose(goal_pose5_world) if goal_pose5_world is not None else None
        _, native_info = self.simulator.reset(seed=int(seed), options=options)
        self.primitive_steps = 0
        self._goal_pose = _pose(self.simulator.goal_state[:5])
        if goal is not None:
            # Native reset physically places bodies at its goal before restoring
            # the initial pose. Goal contact can leave collision corrections that
            # alter that initial world. A desired goal must affect labels only.
            self.set_goal(goal)
        # A declared baseline only. Rare out-of-world physical reset positions
        # are clipped here to executable targets; source actions are never clipped.
        self._hold_action = np.clip(self.pose[:2]/512, 0, 1)
        return self.render(), self._info(native_info, native_terminated=False)

    def set_goal(self, goal_pose5_world):
        """Change numeric evaluation goal, preserving source green-overlay drawing."""
        self._goal_pose = _pose(goal_pose5_world)
        self.simulator._set_goal_state(np.r_[self._goal_pose, [0., 0.]])

    def render(self):
        """Always obtain a fresh actual RGB96 render before canonicalizing."""
        from .data import canonical_frame
        return canonical_frame(self.simulator.render())

    def _info(self, native_info, *, native_terminated):
        actual = self.pose
        delta = actual-self._goal_pose
        angle_error = abs(float(np.arctan2(np.sin(delta[4]), np.cos(delta[4]))))
        block_distance = float(np.linalg.norm(delta[2:4]))
        pusher_distance = float(np.linalg.norm(delta[:2]))
        return {'pose_world': actual, 'goal_pose_world': self._goal_pose.copy(),
                'block_distance': block_distance, 'angle_error': angle_error,
                'pusher_distance': pusher_distance,
                'success': bool(block_distance < 20 and angle_error < math.pi/9),
                'native_terminated': bool(native_terminated),
                'primitive_steps': self.primitive_steps, 'elapsed_seconds': .1*self.primitive_steps,
                'contacts': int(getattr(self.simulator, 'n_contact_points', 0)),
                'hold_action_source': 'privileged actual reset pusher position clipped to [0,1]',
                'reset_velocity_convention': 'two zero pusher defaults; not source velocity labels',
                'goal_reset_convention': 'numeric goal set after physical reset; drawing unchanged',
                'environment_schema': ENVIRONMENT_SCHEMA_VERSION}

    def step(self, normalized_action):
        if self._goal_pose is None:
            raise RuntimeError('reset before taking an action')
        action = np.asarray(normalized_action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all() or np.any((action < 0) | (action > 1)):
            raise ValueError('action must be one finite normalized absolute XY target in [0,1]')
        # Exactly one 0.1-second PD-control primitive. Ignore SWM's full-pose
        # termination for prefix replay and apply this task's block-only success.
        _, _, native_terminated, truncated, native_info = self.simulator.step(512*action)
        self.primitive_steps += 1
        info = self._info(native_info, native_terminated=native_terminated)
        reward = -(info['block_distance']/20)**2-(info['angle_error']/(math.pi/9))**2
        return self.render(), reward, info['success'], bool(truncated), info

    def close(self):
        self.simulator.close()
