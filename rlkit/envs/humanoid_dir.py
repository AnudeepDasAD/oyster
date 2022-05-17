import numpy as np
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv
# from dm_control.suite.humanoid import Humanoid, Physics
from dm_control import mujoco
from dm_control.suite import common

from . import register_env
from .reward_utils import tolerance
from rlkit.core import logger


# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.4

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))

def head_height(data):
    return data.get_body_xpos('head')

def torso_upright(data):
    """Returns projection from z-axes of torso to the z-axes of world."""
    return data.get_body_xmat('torso') # Should be a projection

def control(data):
  """Returns MuJoCo actuation vector as defined in the model."""
  return data.ctrl[:]

def center_of_mass_velocity(data):
  """Returns the velocity of the center-of-mass."""
  return data.subtree_linvel['torso']

def center_of_mass_position(data):
  """Returns position of the center-of-mass."""
  return data.subtree_com['torso']

def torso_vertical_orientation(data):
  """Returns the z-projection of the torso orientation matrix."""
  return data.xmat['torso', ['zx', 'zy', 'zz']]

def joint_angles(data):
  """Returns the state without global orientation or position."""
  return data.qpos[7:]  # Skip the 7 DoFs of the free root joint.

def extremities(data):
  """Returns end effector positions in egocentric frame."""
  torso_frame = data.xmat['torso'].reshape(3, 3)
  torso_pos = data.xpos['torso']
  positions = []
  for side in ('left_', 'right_'):
    for limb in ('hand', 'foot'):
      torso_to_limb = data.xpos[side + limb] - torso_pos
      positions.append(torso_to_limb.dot(torso_frame))
  return np.hstack(positions)


@register_env('humanoid-dir')
class HumanoidDirEnv(HumanoidEnv):

    def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)
        self._move_speed = _WALK_SPEED
        self._pure_state = True
        super(HumanoidDirEnv, self).__init__()

    def step(self, action):
        # pos_before = np.copy(mass_center(self.model, self.sim)[:2])
        # self.do_simulation(action, self.frame_skip)
        # pos_after = mass_center(self.model, self.sim)[:2]

        # alive_bonus = 5.0
        # data = self.sim.data
        # goal_direction = (np.cos(self._goal), np.sin(self._goal))
        # lin_vel_cost = 0.25 * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
        # quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        # quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        # quad_impact_cost = min(quad_impact_cost, 10)
        # reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        # qpos = self.sim.data.qpos
        # done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        self.do_simulation(action, self.frame_skip)
        data = self.sim.data
        
        logger.log(f'Anudeep--- {dir(self.sim.data.get_body_xpos)}')
        
        """Returns a reward to the agent."""
        standing = tolerance(head_height(data),
                                    bounds=(_STAND_HEIGHT, float('inf')),
                                    margin=_STAND_HEIGHT/4)
        upright = tolerance(torso_upright(data),
                                    bounds=(0.9, float('inf')), sigmoid='linear',
                                    margin=1.9, value_at_margin=0)
        stand_reward = standing * upright
        small_control = tolerance(control(data), margin=1,
                                        value_at_margin=0,
                                        sigmoid='quadratic').mean()
        small_control = (4 + small_control) / 5
        if self._move_speed == 0:
            horizontal_velocity = center_of_mass_velocity(data)[[0, 1]]
            dont_move = tolerance(horizontal_velocity, margin=2).mean()
            reward = small_control * stand_reward * dont_move
        else:
            com_velocity = np.linalg.norm(center_of_mass_velocity(data)[[0, 1]])
            move = tolerance(com_velocity,
                                    bounds=(self._move_speed, float('inf')),
                                    margin=self._move_speed, value_at_margin=0,
                                    sigmoid='linear')
            move = (5*move + 1) / 6
            reward = small_control * stand_reward * move
        qpos = data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0)) # This was the PEARL definition of done

        return self._get_obs(), reward, done, dict(reward=reward)
        ''', dict(reward_linvel=lin_vel_cost,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost)'''

    def _get_obs(self):
    #   return self.get_observation(self.physics)
        # data = self.sim.data
        # # return np.concatenate([data.qpos.flat[2:],
        # #                        data.qvel.flat,
        # #                        data.cinert.flat,
        # #                        data.cvel.flat,
        # #                        data.qfrc_actuator.flat,
        # #                        data.cfrc_ext.flat])
        # """Returns either the pure state or a set of egocentric features."""
        obs = collections.OrderedDict()
        if self._pure_state:
            obs['position'] = data.qpos.flat
            obs['velocity'] = data.qvel.flat
        else:
            obs['joint_angles'] = joint_angles(data)
            obs['head_height'] = head_height(data)
            obs['extremities'] = extremities(data)
            obs['torso_vertical'] = torso_vertical_orientation(data)
            obs['com_velocity'] = center_of_mass_velocity(data)
            obs['velocity'] = velocity(data)
        return obs

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector

    def sample_tasks(self, num_tasks):
        # velocities = np.random.uniform(0., 1.0 * np.pi, size=(num_tasks,))
        directions = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        tasks = [{'goal': d} for d in directions]
        return tasks
