import enum
import math
import os
import random
import tempfile
import xml.etree.ElementTree as ET
import inspect
from collections import deque

import glfw
import numpy as np

import mujoco

from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import utils
from scipy.spatial.transform import Rotation

BIG = 1e6
DEFAULT_CAMERA_CONFIG = {}


def euler2mat(euler):
    r = Rotation.from_euler('xyz', euler, degrees=False)
    return r.as_matrix()


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


class Commands(enum.Enum):
    NOPE = enum.auto()
    FORWARD = enum.auto()
    TURN0 = enum.auto()
    TURN1 = enum.auto()


class CommandEnv(MujocoEnv, utils.EzPickle):
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
                 fixed_command=None,
                 coef_inner_rew=0.,
                 coef_main_rew=1.,
                 coef_ctrl_cost=0.001,  # Yoshida et al. 2021 setting
                 coef_head_angle=0.005,  # Yoshida et al. 2021 setting
                 dying_cost=-10,
                 max_episode_steps=150,
                 show_sensor_range=False,
                 reward_setting="homeostatic_shaped",
                 reward_bias=None,
                 show_move_line=False,
                 domain_randomization=False,
                 *args, **kwargs):
        """

        :param coef_inner_rew:
        :param coef_main_rew:
        :param coef_cost:
        :param coef_head_angle:
        :param dying_cost:
        :param max_episode_steps:
        :param show_sensor_range: Show range sensor. Default OFF
        :param reward_setting: Setting of the reward definitions. "homeostatic", "homeostatic_shaped", "one", "homeostatic_biased" or "greedy". "homeostatic_shaped" is default. "greedy is not a homeostatic setting"
        :param reward_bias: biasing reward with constant. new_reward = reward + reward_bias
        :param show_move_line: render the movement of the agent in the environment
        :param vision: enable vision outputs
        :param width: vision width
        :param height: vision height
        :param args:
        :param kwargs:
        """
        self.n_bins = 20
        self.sensor_range = 3.
        self.sensor_span = 2 * np.pi
        self.coef_inner_rew = coef_inner_rew
        self.coef_main_rew = coef_main_rew
        self.coef_ctrl_cost = coef_ctrl_cost
        self.coef_head_angle = coef_head_angle
        self.dying_cost = dying_cost
        self._max_episode_steps = max_episode_steps
        self.show_sensor_range = show_sensor_range
        self.reward_setting = reward_setting
        self.reward_bias = reward_bias if reward_bias else 0.
        self.show_move_line = show_move_line
        self.domain_randomization = domain_randomization
        
        utils.EzPickle.__init__(**locals())
        
        # for openai baseline
        self.reward_range = (-float('inf'), float('inf'))
        
        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise Exception("MODEL_CLASS unspecified!")
        import pathlib
        p = pathlib.Path(inspect.getfile(self.__class__))
        self.MODEL_DIR = os.path.join(p.parent, "models", model_cls.FILE)
        
        tree = ET.parse(self.MODEL_DIR)
        
        with tempfile.NamedTemporaryFile(mode='wt', suffix=".xml") as tmpfile:
            file_path = tmpfile.name
            tree.write(file_path)
            
            # build mujoco
            self.wrapped_env = model_cls(
                file_path,
                **kwargs
            )
        
        # Command
        self.is_command_fixed = fixed_command is not None
        self.command = Commands.NOPE if fixed_command is None else self.num_to_command(fixed_command)
        
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
        
        # visualization
        self.agent_positions = deque(maxlen=300)
        
    @property
    def dim_intero(self):
        return 0
    
    @staticmethod
    def num_to_command(n):
        if n is None:
            return None
        elif n == 0:
            return Commands.NOPE
        elif n == 1:
            return Commands.FORWARD
        elif n == 2:
            return Commands.TURN0
        elif n == 3:
            return Commands.TURN1
        else:
            raise ValueError(f"Invalid Command Number: {n}")
    
    def reset(self, seed=None, return_info=True, options=None):
        self._step = 0
        
        # new command if not fixed
        if self.is_command_fixed is False:
            self.command = random.choice([c for c in Commands])
            
        self.wrapped_env.reset(seed=seed)
        
        self.agent_positions.clear()
        
        info = {}
        
        return (self.get_current_obs(), info) if return_info else self.get_current_obs()
    
    def set_and_fix_command(self, new_command: int):
        assert new_command < len(Commands)
        
        self.command = self.num_to_command(new_command)
        self.is_command_fixed = True
    
    def step(self, action: np.ndarray):
        
        motor_action = action.copy()
        
        robot_x_prev, robot_y_prev = self.get_robot_pos()
        robot_dir_prev = self.get_ori()
        
        _, inner_rew, terminated, truncated, info = self.wrapped_env.step(motor_action)
        truncated = False
        
        info['inner_rew'] = inner_rew
        com = self.wrapped_env.get_body_com("torso")
        self.agent_positions.append(np.array(com))
        info['com'] = com
        
        self._step += 1
        terminated = terminated or self._step >= self._max_episode_steps
        
        # Reward
        # Main Reward
        robot_x, robot_y = self.get_robot_pos()
        robot_dir = self.get_ori()
        
        main_reward = 0.
        if self.command is Commands.NOPE:
            # Motor and postural cost only.
            pass
        elif self.command is Commands.FORWARD:
            dir_cost = -np.abs(robot_dir) / np.pi
            forward_rew = robot_x - robot_x_prev
            main_reward += 4 * forward_rew + 0.05 * dir_cost
        elif self.command is Commands.TURN0:
            pos_cost = -np.sqrt(robot_x ** 2 + robot_y ** 2)
            if robot_dir < -0.8 * np.pi and robot_dir_prev > 0.8 * np.pi:
                terminated = True
                robot_dir = np.pi
            turn_rew = robot_dir - robot_dir_prev
            main_reward += turn_rew + 0.02 * pos_cost
        elif self.command is Commands.TURN1:
            pos_cost = -np.sqrt(robot_x ** 2 + robot_y ** 2)
            if robot_dir > 0.8 * np.pi and robot_dir_prev < -0.8 * np.pi:
                terminated = True
                robot_dir = -np.pi
            turn_rew = - (robot_dir - robot_dir_prev)
            main_reward += turn_rew + 0.02 * pos_cost
        else:
            raise ValueError("Invalid Command")
        
        # Motor Cost
        lb, ub = self.action_space.low, self.action_space.high
        scaling = (ub - lb) * 0.5
        ctrl_cost = -.5 * np.square(action / scaling).sum()
        
        # Local Posture Cost
        euler = qtoeuler(self.wrapped_env.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4])
        euler_stand = qtoeuler([1.0, 0.0, 0.0, 0.0])  # quaternion of standing state
        head_angle_cost = -np.square(euler[:2] - euler_stand[:2]).sum()  # Drop yaw
        
        total_cost = self.coef_ctrl_cost * ctrl_cost + self.coef_head_angle * head_angle_cost
        
        reward = self.coef_main_rew * main_reward + total_cost
        
        info = {"task": self.command.name}
        
        return self.get_current_obs(), reward, terminated, truncated, info
    
    def get_command_embed(self):  # equivalent to get_current_maze_obs in maze_env.py
        embed = np.arange(len(Commands)) == (self.command.value - 1)
        return embed.astype(np.float32)
    
    def get_robot_pos(self):
        com = self.wrapped_env.get_body_com("torso")
        return np.array(com[:2], dtype=np.float32)
    
    def get_current_robot_obs(self):
        return self.wrapped_env._get_obs()
    
    def get_current_obs(self):
        # return sensor data along with data about itself
        self_obs = self.wrapped_env._get_obs()
        command_embed = self.get_command_embed()
        return np.concatenate([self_obs, command_embed], dtype=np.float32)
    
    @property
    def multi_modal_dims(self):
        self_obs_dim = len(self.wrapped_env._get_obs())
        reading_dim = len(self.get_command_embed())
        direction_dim = 2
        return tuple([self_obs_dim, reading_dim + direction_dim])
    
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
        
        # Show Command
        for v in viewers:
            # Showing target and agent's direction
            robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
            v.add_marker(
                pos=np.array([robot_x, robot_y, 0.4]),
                label=self.command.name,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=(0.01),
                rgba=(1, 1, 0, 0)
            )
        
        # Show Sensor Range
        if self.show_sensor_range:
            
            robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
            ori = self.get_ori()
            
            sensor_range = np.linspace(start=-self.sensor_span,
                                       stop=self.sensor_span,
                                       num=self.n_bins,
                                       endpoint=True)
            for direction in sensor_range:
                ox = robot_x + self.sensor_range * math.cos(direction + ori)
                oy = robot_y + self.sensor_range * math.sin(direction + ori)
                for v in viewers:
                    v.add_marker(
                        pos=np.array([ox, oy, 0.5]),
                        label=" ",
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=(0.05, 0.05, 0.05),
                        rgba=(0, 0, 1, 0.8)
                    )
        
        im = self.wrapped_env.mujoco_renderer.render(
            mode,
            camera_id,
            camera_name,
        )
        
        # delete unnecessary markers: https://github.com/openai/mujoco-py/issues/423#issuecomment-513846867
        for v in viewers:
            del v._markers[:]
        
        return im
