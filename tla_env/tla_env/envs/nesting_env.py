import math
import os
import tempfile
import xml.etree.ElementTree as ET
import inspect
from collections import deque
from enum import Enum, auto

import glfw
import numpy as np

import mujoco

from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE, MujocoEnv
from gymnasium import utils
from scipy.spatial.transform import Rotation

BIG = 1e6
DEFAULT_CAMERA_CONFIG = {}


def euler2mat(euler):
    r = Rotation.from_euler('xyz', euler, degrees=False)
    return r.as_matrix()


class ObjectClass(Enum):
    FOOD = auto()


class InteroClass(Enum):
    ENERGY = auto()
    TEMPERATURE = auto()


def qtoeuler(q):
    """ quaternion to Euler angle

    :param q: quaternion
    :return:
    """
    phi = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    theta = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    psi = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([phi, theta, psi])


def eulertoq(euler):
    phi, theta, psi = euler
    qx = np.cos(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qy = np.sin(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) - np.cos(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qz = np.cos(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2)
    qw = np.cos(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2) - np.sin(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2)
    return np.array([qx, qy, qz, qw])


class NestingEnv(MujocoEnv, utils.EzPickle):
    MODEL_CLASS = None
    ORI_IND = None
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps"  : 20,
    }
    
    def __init__(self,
                 coef_inner_rew=0.,
                 coef_main_rew=100.,
                 coef_ctrl_cost=0.001,
                 coef_head_angle=0.005,
                 dying_cost=-10,
                 max_episode_steps=1_000,
                 show_move_line=False,
                 domain_randomization=False,
                 position_homeostasis=True,
                 *args, **kwargs):
        self.coef_inner_rew = coef_inner_rew
        self.coef_main_rew = coef_main_rew
        self.coef_ctrl_cost = coef_ctrl_cost
        self.coef_head_angle = coef_head_angle
        self.dying_cost = dying_cost
        self._max_episode_steps = max_episode_steps
        self.show_move_line = show_move_line
        self.domain_randomization = domain_randomization
        self.position_homeostasis = position_homeostasis
        
        utils.EzPickle.__init__(**locals())
        
        # for openai baseline
        self.reward_range = (-float('inf'), float('inf'))
        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise Exception("MODEL_CLASS unspecified!")
        
        import pathlib
        p = pathlib.Path(inspect.getfile(self.__class__))
        MODEL_DIR = os.path.join(p.parent, "models", model_cls.FILE)
        
        # build mujoco
        self.wrapped_env = model_cls(
            MODEL_DIR,
            **kwargs
        )
        
        self.prev_robot_xy = self.wrapped_env.get_body_com("torso")[:2].copy()
        
        # optimization, caching obs spaces
        ub = BIG * np.ones((self.wrapped_env.observation_space.shape[0] + 4), dtype=np.float32)
        self.obs_space = spaces.Box(ub * -1, ub)
        ub = BIG * np.ones(self.wrapped_env.observation_space.shape, dtype=np.float32)
        self.robot_obs_space = spaces.Box(ub * -1, ub)
        
        # Augment the action space
        ub = np.ones(len(self.wrapped_env.action_space.high), dtype=np.float32)
        self.act_space = spaces.Box(ub * -1, ub)
        
        self.max_episode_length = self._max_episode_steps
        
        self._step = 0
        
        self.target_xy = None
        
        # visualization
        self.agent_positions = deque(maxlen=300)
    
    def set_new_target(self, xy):
        self.target_xy = xy
    
    def set_random_position(self):
        self.wrapped_env.init_qpos[:2] = self.np_random.uniform(-2, 2, size=2)
        
        random_angle = self.np_random.uniform(0, 2 * np.pi)
        q = eulertoq(np.array([0, 0, random_angle]))
        self.wrapped_env.init_qpos[3:3 + 4] = q
    
    def reset(self, seed=None, return_info=True, options=None):
        self._step = 0
        
        # set random position
        self.set_random_position()
        
        self.wrapped_env.reset(seed=seed)
        self.prev_robot_xy = self.wrapped_env.get_body_com("torso")[:2].copy()
        
        self.agent_positions.clear()
        
        info = {"target_xy": self.target_xy}
        return (self.get_current_obs(), info) if return_info else self.get_current_obs()
    
    def step(self, action: np.ndarray):
        action = np.clip(action, a_min=-1, a_max=1)
        
        motor_action = action
        
        self.prev_robot_xy = self.wrapped_env.get_body_com("torso")[:2].copy()
        _, inner_rew, terminated, truncated, info = self.wrapped_env.step(motor_action)
        truncated = False
        
        info['inner_rew'] = inner_rew
        com = self.wrapped_env.get_body_com("torso")
        x, y = com[:2]
        self.agent_positions.append(np.array(com, np.float32))
        info['com'] = com
        
        self._step += 1
        terminated = terminated or self._step >= self._max_episode_steps
        
        reward, info_rew = self.get_reward(action=action, done=terminated)
        
        info.update(info_rew)
        info.update({"target_xy": self.target_xy})
        
        return self.get_current_obs(), reward, terminated, truncated, info
    
    def get_reward(self, action, done):
        
        info = {"position_cost": None}
        
        def drive(intero, target):
            drive_module = -1 * np.abs(intero - target)
            d_ = drive_module.sum()
            return d_, drive_module
        
        # Motor Cost
        lb, ub = self.action_space.low, self.action_space.high
        scaling = (ub - lb) * 0.5
        ctrl_cost = -.5 * np.square(action / scaling).sum()
        
        # Local Posture Cost
        euler = qtoeuler(self.wrapped_env.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4])
        euler_stand = qtoeuler([1.0, 0.0, 0.0, 0.0])  # quaternion of standing state
        head_angle_cost = -np.square(euler[:2] - euler_stand[:2]).sum()  # Drop yaw
        
        # Global Positional Cost (Main Reward)
        if self.position_homeostasis:
            d, dm = drive(self.wrapped_env.get_body_com("torso")[:2], np.zeros(2))
            d_prev, dm_prev = drive(self.prev_robot_xy, np.zeros(2))
            position_cost = d - d_prev
        else:
            body_xy = self.wrapped_env.get_body_com("torso")[:2]
            position_cost = - (body_xy[0] ** 2 + body_xy[1] ** 2)
        
        info["position_cost"] = position_cost
        
        reward = self.coef_ctrl_cost * ctrl_cost + self.coef_head_angle * head_angle_cost + self.coef_main_rew * position_cost
        
        return reward, info
    
    def get_current_robot_obs(self):
        return self.wrapped_env._get_obs()
    
    def get_current_obs(self):
        # return sensor data along with data about itself
        self_obs = self.wrapped_env._get_obs()
        
        body_xy = self.wrapped_env.get_body_com("torso")[:2]
        if self.target_xy is not None:
            body_xy -= self.target_xy
        
        ori = self.get_ori()
        
        return np.concatenate([self_obs, body_xy, [np.cos(ori)], [np.sin(ori)]], dtype=np.float32)
    
    @property
    def multi_modal_dims(self):
        proprioception_dim = self.robot_obs_space.shape[0]
        
        # (proprioception, exteroception, interoception)
        return tuple([proprioception_dim, 2])
    
    @property
    def observation_space(self):
        return self.obs_space
    
    # space of only the robot observations (they go first in the get current obs)
    @property
    def robot_observation_space(self):
        return self.robot_obs_space
    
    @property
    def action_space(self):
        return self.act_space
    
    @property
    def dt(self):
        return self.wrapped_env.dt
    
    def close(self):
        if self.wrapped_env.mujoco_renderer is not None:
            self.wrapped_env.mujoco_renderer.close()
    
    def get_ori(self):
        """
        First it tries to use a get_ori from the wrapped tla_env. If not successfull, falls
        back to the default based on the ORI_IND specified in Maze (not accurate for quaternions)
        """
        obj = self.wrapped_env
        while not hasattr(obj, 'get_ori') and hasattr(obj, 'wrapped_env'):
            obj = obj.wrapped_env
        try:
            return obj.get_ori()
        except (NotImplementedError, AttributeError) as e:
            pass
        return self.wrapped_env.data.qpos[self.__class__.ORI_IND]
    
    def render(
            self,
            mode='human',
            camera_id=None,
            camera_name=None
    ):
        return self.get_image(mode=mode, camera_id=camera_id, camera_name=camera_name)
    
    def get_image(
            self,
            mode='human',
            camera_id=1,
            camera_name=None
    ):
        
        viewers = [self.wrapped_env.mujoco_renderer._get_viewer(render_mode=mode)]
        
        # show movement of the agent
        if self.show_move_line:
            for pos in self.agent_positions:
                for v in viewers:
                    v.add_marker(pos=pos,
                                 label=" ",
                                 type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                 size=(0.05, 0.05, 0.05),
                                 rgba=(1, 0, 0, 0.3),
                                 emission=1)
        
        if self.target_xy is not None:
            for v in viewers:
                v.add_marker(pos=[self.target_xy[0], self.target_xy[1], 0],
                             label="target",
                             type=mujoco.mjtGeom.mjGEOM_SPHERE,
                             size=(0.1, 0.1, 0.1),
                             rgba=(1, 0, 0, 1),
                             emission=1)
        
        im = self.wrapped_env.mujoco_renderer.render(
            mode,
            camera_id,
            camera_name,
        )
        
        # delete unnecessary markers: https://github.com/openai/mujoco-py/issues/423#issuecomment-513846867
        for v in viewers:
            del v._markers[:]
        
        return im
