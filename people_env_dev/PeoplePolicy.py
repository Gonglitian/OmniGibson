from omnigibson.people.core.person import Person
from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class PersonAction:
    target_position: np.ndarray = None
    speed: float = None
    velocity: np.ndarray = None

    def __post_init__(self):
        """
        验证 target_position 和 velocity 不能同时有值
        """
        if (self.target_position is not None) and (self.velocity is not None):
            raise ValueError("target_position and velocity cannot be both set in PersonAction")

class PeoplePolicy:
    def __init__(self, person: Person,policy_cfg:dict=None,*args,**kwargs):
        if policy_cfg is None:
            policy_cfg = {}

        self.policy_cfg = policy_cfg
        self.person = person
        self.radius = self.policy_cfg.get("radius",0.3)     # 行人半径
        self.neighbor_threshold = self.policy_cfg.get("neighbor_threshold",1.0)
        self.neighbors = []   # 邻近的其他行人

    def get_neighbors(self):
        # return other person in the same env which distance is less than neighbor_threshold
        return [p for p in self.person.env.people if p != self.person and np.linalg.norm(p.position - self.person.position) < self.neighbor_threshold]
    
    def step(self, dt: float):
        """更新行人状态"""
        action = self.generate_action(self.person,self.neighbors)
        self.apply_action(action,dt)

    def generate_action(self, person: Person, neighbors: List[Person]):
        """
        生成行人动作
        """
        return NotImplementedError

    def apply_action(self, action:PersonAction,dt:float):
        """
        应用行人动作
        """
        return NotImplementedError

class DefaultPolicy(PeoplePolicy):
    def __init__(self,policy_cfg:dict=None,*args,**kwargs):
        if policy_cfg is None:
            policy_cfg = {}
        super().__init__(policy_cfg,*args,**kwargs)

    def generate_action(self, person: Person, neighbors: List[Person]):
        return ...

    def apply_action(self, action:PersonAction,dt:float):
        return ...

class ORCAPolicy(PeoplePolicy):
    def __init__(self,policy_cfg:dict=None,*args,**kwargs):
        if policy_cfg is None:
            policy_cfg = {}
        super().__init__(policy_cfg,*args,**kwargs)

    def generate_action(self, person: Person, neighbors: List[Person]):
        return ...

    def apply_action(self, action:PersonAction,dt:float):
        return ...