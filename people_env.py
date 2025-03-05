import gymnasium as gym
import numpy as np
from gymnasium import spaces

import omnigibson as og
from omnigibson.macros import gm
from typing import List
gm.HEADLESS = True
gm.REMOTE_STREAMING = "native"


cfg = dict()

cfg["env"] = {
    "device": "cpu",
}

cfg["scene"] = {
    "type": "Scene",
    # "scene_model": "Rs_int",
    "floor_plane_visible": True,
}

cfg["objects"] = [
    {
        "type": "USDObject",
        "name": "ghost_stain",
        "usd_path": f"{gm.ASSET_PATH}/models/stain/stain.usd",
        "category": "stain",
        "visual_only": True,
        "scale": [1.0, 1.0, 1.0],
        "position": [1.0, 2.0, 0.001],
        "orientation": [0, 0, 0, 1.0],
    },
]

cfg["robots"] = [
    {
        "type": "Fetch",
        "name": "skynet_robot",
        "obs_modalities": ["rgb", "depth"],
        "default_arm_pose": "diagonal30",
        "default_reset_mode": "tuck",
    },
]
env = og.Environment(cfg)

from omnigibson.people import Person
from omni.isaac.core.world import World

PERSON_MODELS = [
    "F_Business_02",
    "F_Medical_01",
    "M_Medical_01",
    "female_adult_police_01_new",
    "female_adult_police_02",
    "female_adult_police_03_new",
    "male_adult_construction_01_new",
    "male_adult_construction_03",
    "male_adult_construction_05_new",
    "male_adult_police_04",
    "original_female_adult_business_02",
    "original_female_adult_medical_01",
    "original_female_adult_police_01",
    "original_female_adult_police_02",
    "original_female_adult_police_03",
    "original_male_adult_construction_01",
    "original_male_adult_construction_02",
    "original_male_adult_construction_03",
    "original_male_adult_construction_05",
    "original_male_adult_medical_01",
    "original_male_adult_police_04"
]


def orca_velocity(person:Person, neighbors:List[Person], time_horizon=5.0):
    """
    简化版 ORCA 算法：
      1. 根据当前位置和目标位置计算期望速度；
      2. 对于靠得较近的邻居增加一个排斥项，避免碰撞；
      3. 将合成速度裁剪到行人的最大速度范围内。
    假设 Person 对象提供 get_position() 与 get_target_position() 方法，
    同时包含属性 max_speed（默认 0.1）和 radius（默认 0.3）。
    """
    pos = np.array(person.get_position())    # [x, y, z]
    target = np.array(person.get_target_position())  # [x, y, z]
    pos_2d = pos[:2]
    target_2d = target[:2]
    
    direction = target_2d - pos_2d
    norm = np.linalg.norm(direction)
    max_speed = getattr(person, 'max_speed', 1)
    if norm < 1e-5:
        v_pref = np.zeros(2)
    else:
        v_pref = direction / norm * max_speed
    
    avoidance = np.zeros(2)
    for other in neighbors:
        other_pos = np.array(other.get_position())[:2]
        diff = pos_2d - other_pos
        dist = np.linalg.norm(diff)
        radius = getattr(person, 'radius', 0.3)
        other_radius = getattr(other, 'radius', 0.3)
        combined_radius = radius + other_radius
        if dist < combined_radius * 1.5:
            if dist > 1e-5:
                avoidance += (diff / dist) * (combined_radius - dist)
    new_velocity = v_pref + avoidance
    speed = np.linalg.norm(new_velocity)
    if speed > max_speed:
        new_velocity = new_velocity / speed * max_speed
    return np.array([new_velocity[0],new_velocity[1],0.0])  # 返回二维速度，假设 z 分量为 0

class OmnigibsonPedestrianEnv(gym.Env):
    """
    基于 omnigibson.people 接口的行人环境：
      - 在 reset 时随机生成一定数量的行人，并为部分行人附加随机目标控制器；
      - 每个 step 中，先更新控制器，再利用简化 ORCA 算法计算行人的运动；
      - 观察值为每个行人的二维位置及目标；
      - 奖励设计为负的所有行人当前到目标距离之和，当所有行人均到达目标时结束。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_persons=5, area_size=(10, 10)):
        super(OmnigibsonPedestrianEnv, self).__init__()
        self._world = World()
        
        self.num_persons = num_persons
        self.area_size = area_size
        
        # 观察空间：每个行人提供 [x, y, target_x, target_y]
        self.observation_space = spaces.Box(
            low=0,
            high=max(area_size),
            shape=(self.num_persons, 4),
            dtype=np.float32
        )
        
        # 清空 PeopleManager 内部存储的行人
        self.people: List[Person] = []
        self._init_people()

    def _init_people(self):
        """
        初始化行人：随机生成行人的初始位置与目标，
        """
        for i in range(self.num_persons):
            # 随机生成三维初始位置（假设 z 均为 0）
            init_pos = np.random.uniform(0, self.area_size[0], size=3)
            init_pos[2] = 0.0
            target_pos = np.random.uniform(0, self.area_size[0], size=3)
            target_pos[2] = 0.0
            
            name = f"person_{i}"
            model_name = np.random.choice(PERSON_MODELS)
            print(f"Creating person {name} with model {model_name}")
            init_yaw = np.random.uniform(-np.pi, np.pi)
            # 创建 Person 对象，注意构造函数参数需与 omnigibson 的定义一致
            person = Person(name, model_name, init_pos=init_pos.tolist(), init_yaw=init_yaw)

            person.update_target_position(target_pos.tolist())
            # 添加到 PeopleManager 中
            self.people.append(person)
        # p2 = Person("person2", "original_male_adult_construction_05", init_pos=[
        #         3.0, 0.0, 0.0], init_yaw=1.0)
        # self.people.append(p1)

        self._world.add_physics_callback(
            "people_step", self.step)
        
    def reset(self):
        self.people = []
        self._init_people()
        return self._get_obs()

    def _get_obs(self):
        """
        返回所有行人的观测信息，每个行人包含 [x, y, target_x, target_y]，
        这里只取二维信息，假设 z 分量均为 0。
        """
        obs = np.zeros((self.num_persons, 4), dtype=np.float32)
        for i, person in enumerate(self.people):
            pos = np.array(person.get_position())[:2]
            target = np.array(person.get_target_position())[:2]
            obs[i] = np.concatenate([pos, target])
        return obs

    def step(self, dt):
        """
        1. 如果行人附加了控制器，先调用其 update(dt) 方法更新目标；
        2. 对每个行人计算 ORCA 修正后的运动速度，并更新位置；
        3. 根据所有行人距离目标的和构造奖励，若所有行人距离目标小于一定阈值，则结束 episode。
        """
        velocities = []
        for person in self.people:
            neighbors = [other for other in self.people if other != person]
            v = orca_velocity(person, neighbors)
            velocities.append(v)
        print("velocities:", velocities)
        # 通过v和dt计算行人目标位置
        for person, v in zip(self.people, velocities):
            pos = person.get_position()
            new_pos = pos + v * dt
            person.update_target_position(new_pos)

        # 计算 reward（所有行人到目标距离之和的负值）以及判断是否结束
        total_distance = 0.0
        done = True
        for person in self.people:
            pos = np.array(person.get_position())[:2]
            target = np.array(person.get_target_position())[:2]
            dist = np.linalg.norm(pos - target)
            total_distance += dist
            if dist >= 0.5:
                done = False
        reward = -total_distance
        
        obs = self._get_obs()
        info = {}
        return obs, reward, done, info

people_sim_env = OmnigibsonPedestrianEnv(num_persons=2, area_size=(10, 10))
    
# p1 = Person("person1", "original_male_adult_construction_05", init_pos=[
#                 3.0, 0.0, 0.0], init_yaw=1.0)

og.sim.enable_viewer_camera_teleoperation()

while True:
    og.sim.step()
