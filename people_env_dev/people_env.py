import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass
import omnigibson as og
from omnigibson.macros import gm
from typing import List
gm.HEADLESS = True
gm.REMOTE_STREAMING = "native"

import rvo2 #TODO: need to install python-rvo2 on your local machine

cfg = dict()

cfg["env"] = {
    "device": "cpu",
}

cfg["scene"] = {
    "type": "Scene",
    # "scene_model": "Rs_int",
    "floor_plane_visible": True,
}

env = og.Environment(cfg)

from omnigibson.people import Person
from omni.isaac.core.world import World
from PeoplePolicy import PeoplePolicy, PersonAction,DefaultPolicy,ORCAPolicy

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

def orca_policy(person:Person, neighbors:List[Person], time_horizon=5.0):
    """
    使用RVO2库实现与simple_orca_velocity相同的避障功能：
    1. 根据当前位置和目标位置计算期望速度
    2. 使用RVO2处理与邻居的避障
    3. 确保速度在最大速度范围内
    
    参数与返回值与simple_orca_velocity保持一致

    # 参数说明:
    # timeStep:        float, 仿真的时间步长
    # neighborDist:    float, 考虑避障的邻居搜索范围（距离阈值）
    # maxNeighbors:    size_t, 在避障计算中考虑的最大邻居数量
    # timeHorizon:     float, 与其他代理（人）避障的时间范围
    # timeHorizonObst: float, 与静态障碍物避障的时间范围
    # radius:          float, 代理（人）的半径
    # maxSpeed:        float, 代理（人）的最大移动速度
    # velocity:        tuple, 初始速度，默认为(0, 0)表示静止状态
    """

    radius = getattr(person, 'radius', 0.3)
    max_speed = getattr(person, 'max_speed', 1.0)
    
    # 创建RVO2模拟器实例
    sim = rvo2.PyRVOSimulator(
        0.1,     # 仿真时间步长
        1.5,       # 邻居搜索范围（与原函数中的1.5倍半径对应）
        10,        # 最大邻居数量
        time_horizon,   # 与其他代理避障的时间范围
        time_horizon,  # 与障碍物避障的时间范围
        radius,  # 代理半径
        max_speed  # 最大速度
    )
    
    # 获取当前位置和目标位置
    pos = np.array(person.get_position())
    target = np.array(person.get_target_position())
    pos_2d = pos[:2]
    target_2d = target[:2]
    
    # 添加主要行人
    agent_no = sim.addAgent(
        (pos_2d[0], pos_2d[1]), # 位置
        1.5, # 邻居搜索范围
        10, # 最大邻居数量
        time_horizon, # 与其他代理避障的时间范围
        time_horizon, # 与障碍物避障的时间范围
        radius, # 代理半径
        max_speed, # 最大速度
        (0, 0) # 初始速度
    )
    
    # 添加所有邻居行人
    for neighbor in neighbors:
        neighbor_pos = np.array(neighbor.get_position())[:2]
        sim.addAgent(
            (neighbor_pos[0], neighbor_pos[1]), # 位置
            1.5, # 邻居搜索范围
            10, # 最大邻居数量
            time_horizon, # 与其他代理避障的时间范围
            time_horizon, # 与障碍物避障的时间范围
            radius, # 代理半径
            max_speed, # 最大速度
            (0, 0) # 初始速度
        )
    
    # 计算期望速度
    direction = target_2d - pos_2d
    dist_to_goal = np.linalg.norm(direction)
    if dist_to_goal < 1e-5:
        pref_velocity = (0, 0)
    else:
        pref_velocity = tuple(direction / dist_to_goal * max_speed)
    
    # 设置期望速度并进行一步模拟
    sim.setAgentPrefVelocity(agent_no, pref_velocity)
    sim.doStep()
    
    # 获取计算得到的新速度
    new_velocity = sim.getAgentVelocity(agent_no)
    
    return PersonAction(velocity=new_velocity)


def default_policy(person:Person, neighbors:List[Person], time_horizon=5.0)->PersonAction: 
    return PersonAction(target_position=person.target_position)
    
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

def rvo2_velocity(person:Person, neighbors:List[Person], time_horizon=5.0):
    """
    使用RVO2库实现与simple_orca_velocity相同的避障功能：
    1. 根据当前位置和目标位置计算期望速度
    2. 使用RVO2处理与邻居的避障
    3. 确保速度在最大速度范围内
    
    参数与返回值与simple_orca_velocity保持一致

    # 参数说明:
    # timeStep:        float, 仿真的时间步长
    # neighborDist:    float, 考虑避障的邻居搜索范围（距离阈值）
    # maxNeighbors:    size_t, 在避障计算中考虑的最大邻居数量
    # timeHorizon:     float, 与其他代理（人）避障的时间范围
    # timeHorizonObst: float, 与静态障碍物避障的时间范围
    # radius:          float, 代理（人）的半径
    # maxSpeed:        float, 代理（人）的最大移动速度
    # velocity:        tuple, 初始速度，默认为(0, 0)表示静止状态
    """
    import rvo2

    radius = getattr(person, 'radius', 0.3)
    max_speed = getattr(person, 'max_speed', 1.0)
    
    # 创建RVO2模拟器实例
    sim = rvo2.PyRVOSimulator(
        0.1,     # 仿真时间步长
        1.5,       # 邻居搜索范围（与原函数中的1.5倍半径对应）
        10,        # 最大邻居数量
        time_horizon,   # 与其他代理避障的时间范围
        time_horizon,  # 与障碍物避障的时间范围
        radius,  # 代理半径
        max_speed  # 最大速度
    )
    
    # 获取当前位置和目标位置
    pos = np.array(person.get_position())
    target = np.array(person.get_target_position())
    pos_2d = pos[:2]
    target_2d = target[:2]
    
    # 添加主要行人
    agent_no = sim.addAgent(
        (pos_2d[0], pos_2d[1]), # 位置
        1.5, # 邻居搜索范围
        10, # 最大邻居数量
        time_horizon, # 与其他代理避障的时间范围
        time_horizon, # 与障碍物避障的时间范围
        radius, # 代理半径
        max_speed, # 最大速度
        (0, 0) # 初始速度
    )
    
    # 添加所有邻居行人
    for neighbor in neighbors:
        neighbor_pos = np.array(neighbor.get_position())[:2]
        sim.addAgent(
            (neighbor_pos[0], neighbor_pos[1]), # 位置
            1.5, # 邻居搜索范围
            10, # 最大邻居数量
            time_horizon, # 与其他代理避障的时间范围
            time_horizon, # 与障碍物避障的时间范围
            radius, # 代理半径
            max_speed, # 最大速度
            (0, 0) # 初始速度
        )
    
    # 计算期望速度
    direction = target_2d - pos_2d
    dist_to_goal = np.linalg.norm(direction)
    if dist_to_goal < 1e-5:
        pref_velocity = (0, 0)
    else:
        pref_velocity = tuple(direction / dist_to_goal * max_speed)
    
    # 设置期望速度并进行一步模拟
    sim.setAgentPrefVelocity(agent_no, pref_velocity)
    sim.doStep()
    
    # 获取计算得到的新速度
    new_velocity = sim.getAgentVelocity(agent_no)
    
    # 返回三维速度（z轴速度为0）
    return np.array([new_velocity[0], new_velocity[1], 0.0])


class PeopleEnv(gym.Env):
    """
    基于 omnigibson.people 接口的行人环境：
      - 在 reset 时随机生成一定数量的行人，并为部分行人附加随机目标控制器；
      - 每个 step 中，先更新控制器，再利用简化 ORCA 算法计算行人的运动；
      - 观察值为每个行人的二维位置及目标；
      - 奖励设计为负的所有行人当前到目标距离之和，当所有行人均到达目标时结束。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_persons=5, area_size=(10, 10)):
        super(PeopleEnv, self).__init__()
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
        
        self.people: List[Person] = []
        self.orca_sim_agents:List[int] = []
        self._init_people()
        # 添加物理回调
        self._world.add_physics_callback(
            "people_step", self.step)
        
    def _init_people(self):
        # Note test case
        p1 = self.spawn_person(name="person1",init_pos=[0,0,0],init_yaw=0)
        p2 = self.spawn_person(name="person2",init_pos=[10,0,0],init_yaw=0)
        p3 = self.spawn_person(name="person3",init_pos=[0,10,0],init_yaw=0)
        p4 = self.spawn_person(name="person4",init_pos=[10,10,0],init_yaw=0)
        p1.update_target_position([10,10,0])
        p2.update_target_position([0,10,0])
        p3.update_target_position([10,0,0])
        p4.update_target_position([0,0,0])


    def spawn_person(self,name,model_name=np.random.choice(PERSON_MODELS),init_pos=[0,0,0],init_yaw=0,policy="default",policy_cfg=None):
        person = Person(name, model_name, init_pos=init_pos, init_yaw=init_yaw)
        person.env = self
        if policy == "default":
            person.policy = DefaultPolicy(policy_cfg)
        elif policy == "orca":
            # check if sim = rvo2.PyRVOSimulator is initialized, if not initialize it
            if not hasattr(self, 'sim'):
                """             
                param of rvo2.PyRVOSimulator:
                float 	timeStep,
                float 	neighborDist,
                size_t 	maxNeighbors,
                float 	timeHorizon,
                float 	timeHorizonObst,
                float 	radius,
                float 	maxSpeed,
                """
                # TODO: 需要从omnigibson.people.Person中获取参数
                self.orca_sim = rvo2.PyRVOSimulator(1/60., 1.5, 5, 1.5, 2, 0.4, 1)
            person.policy = ORCAPolicy(policy_cfg)
            # add agent to sim
            # TODO: 需要从omnigibson.people.Person中获取参数
            agent:int = self.orca_sim.addAgent(init_pos[:2], 1.5, 5, 1.5, 2, 0.4, 1, (0, 0))
            # store agent instance to env
            self.orca_sim_agents.append(agent)
        else:
            raise ValueError(f"Invalid policy type: {policy}")
        
        self.people.append(person)
        return person
    
    def reset(self):
        # Todo 完善reset函数
        self.people = []
        self._init_people()
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.num_persons, 4), dtype=np.float32)
        for i, person in enumerate(self.people):
            pos = person.position[:2]
            target = person.temp_target_position[:2]
            obs[i] = np.concatenate([pos, target])
        return obs

    def step(self, dt):
        for person in self.people:
            person.policy.step(dt)

        done = False
        obs = None
        reward = 0
        info = {}
        return obs, reward, done, info

people_sim_env = PeopleEnv(num_persons=2, area_size=(100, 100))

og.sim.enable_viewer_camera_teleoperation()

while True:
    og.sim.step()
