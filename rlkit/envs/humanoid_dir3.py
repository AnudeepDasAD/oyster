import numpy as np
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv
from gym import error, spaces

# from dm_control.suite.humanoid import Humanoid, Physics
# from dm_control import mujoco
# from dm_control.suite import common
from dm_control import suite


from . import register_env
from .reward_utils import tolerance
from rlkit.core import logger


# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.4

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('humanoid.xml'), common.ASSETS

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


@register_env('humanoid-dir')
class HumanoidDirEnv(HumanoidEnv):

    def __init__(self, task={}, n_tasks=1, randomize_tasks=True):
        self.tasks = self.sample_tasks(1)
        self.reset_task(0)
        # self._move_speed = _WALK_SPEED
        # self._pure_state = False
        # self.physics = Physics.from_xml_string(*get_model_and_assets())
        # self.humanoid = Humanoid(move_speed=_WALK_SPEED, pure_state=False)
        self.env = suite.load(domain_name="humanoid", task_name="walk")
        
        HumanoidEnv.__init__(self)


    def step(self, action):
        # pos_before = np.copy(mass_center(self.model, self.sim)[:2])
        # self.do_simulation(action, self.frame_skip)
        # pos_after = mass_center(self.model, self.sim)[:2]

        # alive_bonus = 5.0
        # self.do_simulation(action, self.frame_skip)
        # data = self.sim.data
        # goal_direction = (np.cos(self._goal), np.sin(self._goal))
        # lin_vel_cost = 0.25 * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
        # quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        # quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        # quad_impact_cost = min(quad_impact_cost, 10)
        # reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        # qpos = self.sim.data.qpos
        # done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        
        
        time_step = self.env.step(action)
        # qpos = data.qpos
        # done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0)) # This was the PEARL definition of done
        done=False

        return time_step.observation, time_step.reward, done, dict(reward=time_step.reward)
        ''', dict(reward_linvel=lin_vel_cost,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost)'''

    def _get_obs(self, observation):    
        return    
        # flattened_vals = [item.flatten() for item in list(observation.values())]
        # print(flattened_vals)
        # for val in flattened_vals:
        #     print(val.shape)
        # return np.concatenate(flattened_vals)
        
        # data = self.sim.data
        # return np.concatenate([data.qpos.flat[2:],
        #                        data.qvel.flat,
        #                        data.cinert.flat,
        #                        data.cvel.flat,
        #                        data.qfrc_actuator.flat,
        #                        data.cfrc_ext.flat])
        
        # """Returns either the pure state or a set of egocentric features."""
        # obs = collections.OrderedDict()
        # if self._pure_state:
        #     obs['position'] = data.qpos.flat
        #     obs['velocity'] = data.qvel.flat
        # else:
        #     obs['joint_angles'] = joint_angles(data)
        #     obs['head_height'] = head_height(data)
        #     obs['extremities'] = extremities(data)
        #     obs['torso_vertical'] = torso_vertical_orientation(data)
        #     obs['com_velocity'] = center_of_mass_velocity(data)
        #     obs['velocity'] = velocity(data)
        # return obs

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
