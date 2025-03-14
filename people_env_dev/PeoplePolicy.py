from omnigibson.people.core.person import Person
from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class PersonAction:
    """
    target_position: target position
    temp_target_position: temp target position
    speed: speed
    velocity: velocity
    """
    target_position: np.ndarray = None
    temp_target_position: np.ndarray = None
    speed: float = None
    velocity: np.ndarray = None

class PeoplePolicy:
    def __init__(self, person: Person,policy_cfg:dict=None,*args,**kwargs):
        self.policy_cfg = policy_cfg if policy_cfg is not None else {}
        self.person = person

    def step(self, sim_dt: float):
        """更新行人状态"""
        action = self.generate_action()
        self.apply_action(action,sim_dt)

    def generate_action(self):
        """
        生成行人动作
        """
        return NotImplementedError

    def apply_action(self, action:PersonAction,sim_dt:float):
        """
        应用行人动作
        """
        if action.target_position is not None:
            self.person.update_target_position(action.target_position)
        if action.temp_target_position is not None:
            self.person.temp_target_position = action.temp_target_position
        if action.speed is not None:
            self.person._target_speed = action.speed
        if action.velocity is not None:
            # important: tune sim_dt to make the temp target position change faster
            sim_dt *= 15
            delta_pos = action.velocity * sim_dt
            delta_pos_distance = np.linalg.norm(delta_pos)
            if delta_pos_distance < self.person.stop_radius:
                # warn
                print(f"temp_target_position change is too small, {delta_pos_distance} < {self.person.stop_radius}")
            self.person.temp_target_position = self.person.position + delta_pos
            self.person._target_speed = np.linalg.norm(action.velocity)

                
class DefaultPolicy(PeoplePolicy):
    def __init__(self,policy_cfg:dict=None,*args,**kwargs):
        if policy_cfg is None:
            policy_cfg = {}
        super().__init__(policy_cfg,*args,**kwargs)

    def generate_action(self):
        # set temp target position to target position
        return PersonAction(
            temp_target_position=self.person._target_position,
        )

    def step(self, sim_dt: float):
        super().step(sim_dt)

class ORCAPolicy(PeoplePolicy):
    def __init__(self,policy_cfg:dict=None,*args,**kwargs):
        if policy_cfg is None:
            policy_cfg = {}
        super().__init__(policy_cfg,*args,**kwargs)

    def generate_action(self):
        """
        Generate action using RVO2 for collision avoidance.
        
        Args:
            person (Person): The current person
            neighbors (List[Person]): List of neighboring persons
            
        Returns:
            PersonAction: Action containing the calculated velocity
        """
        # get new vel from orca sim
        new_vel = np.array(self.person.env.orca_sim.getAgentVelocity(self.person.orca_idx))
        # print person position and new_vel
        # for debug
        # if self.person.idx == 0:
        #     print(self.person.position, new_vel)
        # if shape is (2,), change to (3,)
        if new_vel.shape == (2,):
            new_vel = np.concatenate([new_vel, [0]])
        action = PersonAction(
            velocity=new_vel
        )
        return action

    def apply_action(self, action:PersonAction, sim_dt:float):
        super().apply_action(action, sim_dt)
    
    def step(self, sim_dt: float):
        super().step(sim_dt)
    
    @staticmethod
    def calculate_pref_velocity(current_pos, target_pos, target_speed):
        direction = target_pos - current_pos
        dist_to_goal = np.linalg.norm(direction)
        if dist_to_goal < 1e-5:
            pref_velocity = (0, 0)
        else:
            pref_velocity = tuple(direction / dist_to_goal * target_speed)
        return pref_velocity