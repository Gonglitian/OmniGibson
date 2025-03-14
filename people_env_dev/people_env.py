import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass
import omnigibson as og
from omnigibson.macros import gm
from typing import List
from matplotlib import pyplot as plt
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

og_env = og.Environment(cfg)

from omnigibson.people import Person
from omni.isaac.core.world import World
from PeoplePolicy import DefaultPolicy,ORCAPolicy

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

class PeopleEnv(gym.Env):
    """
    A Gym environment for simulating multiple pedestrians in a 2D space.
    
    This environment uses RVO2 (Reciprocal Velocity Obstacles) for local collision avoidance
    and supports different movement policies for pedestrians.
    
    Attributes:
        metadata (dict): Gym environment metadata
        _world (World): The simulation world instance
        n_physics_timesteps_per_render (int): Number of physics steps per rendering step
        step_count (int): Counter for simulation steps
        num_persons (int): Number of pedestrians in the environment
        area_size (tuple): Size of the simulation area (width, height)
        observation_space (spaces.Box): Gym observation space for the environment
        people (List[Person]): List of Person objects in the environment
        orca_sim_agents (List[int]): List of agent IDs in the RVO2 simulator
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, num_persons=5, area_size=(10, 10)):
        """
        Initialize the PeopleEnv environment.
        
        Args:
            num_persons (int): Number of pedestrians to create
            area_size (tuple): Size of the simulation area (width, height)
            n_physics_timesteps_per_render (int): Number of physics steps per rendering step
        """
        super(PeopleEnv, self).__init__()

        # sim frequency related
        self.sim_dt = og.sim._sim_step_dt
        self.sim_freq = int(1.0 / self.sim_dt)
        self.n_physics_timesteps_per_render = og.sim.n_physics_timesteps_per_render
        print(f"sim_dt: {self.sim_dt}, sim_freq: {self.sim_freq}, n_physics_timesteps_per_render: {self.n_physics_timesteps_per_render}")
        self.step_count = 0
        
        # environment related
        self.num_persons = num_persons
        self.area_size = area_size
        self.people: List[Person] = []
        # orca sim related
        self.orca_sim_agents:List[int] = []
        self.orca_sim_hyper_params = {
            "timeStep": self.sim_dt,
            "neighborDist": 1.5,
            "maxNeighbors": 5,
            "timeHorizon": 1.5,
            "timeHorizonObst": 2,
            "radius": 0.5,
            "maxSpeed": 1.0,
        }
        """ 
        param of agent:\n
        const Vector2 & 	position,\n
        float 	neighborDist,\n
        size_t 	maxNeighbors,\n
        float 	timeHorizon,\n
        float 	timeHorizonObst,\n
        float 	radius,\n
        float 	maxSpeed,\n
        const Vector2 & 	velocity = Vector2()
        """ 
        self.orca_agent_hyper_params = {
            "neighborDist": 1.5,
            "maxNeighbors": 5,
            "timeHorizon": 1.5,
            "timeHorizonObst": 2,
            "radius": 0.5,
            "maxSpeed": 1.0,
        }
        self._load_people()
        
        # Add physics callback
        self._world = World()
        self._world.add_physics_callback(
            "people_step", self.step)
        
        # visualize related
        self.history_positions_x = []
        self.history_positions_y = []
    def _load_people(self):
        """
        Initialize test pedestrians with predefined positions and target positions.
        Creates 4 pedestrians moving in a cross pattern.
        """
        # Note test case
        p1 = self.spawn_person(name="person1",init_pos=[-5,-5,0],init_yaw=0)
        p2 = self.spawn_person(name="person2",init_pos=[5,-5,0],init_yaw=0)
        p3 = self.spawn_person(name="person3",init_pos=[-5,5,0],init_yaw=0)
        p4 = self.spawn_person(name="person4",init_pos=[5,5,0],init_yaw=0)
        p1.update_target_position([10,10,0])
        p2.update_target_position([-10,10,0])
        p3.update_target_position([10,-10,0])
        p4.update_target_position([-10,-10,0])


    def spawn_person(self,name,model_name=None,init_pos=[0,0,0],init_yaw=0,policy="orca",policy_cfg=None):
        """
        Create and spawn a new person in the environment.
        
        Args:
            name (str): Unique identifier for the person
            model_name (str): Name of the 3D model to use for the person
            init_pos (list): Initial position [x, y, z]
            init_yaw (float): Initial yaw angle
            policy (str): Movement policy type ("default" or "orca")
            policy_cfg (dict): Configuration for the movement policy
            
        Returns:
            Person: The created person object
        """
        if model_name is None:
            model_name = np.random.choice(PERSON_MODELS)
        person = Person(name, model_name, init_pos=init_pos, init_yaw=init_yaw)
        person.env = self
        if policy == "default":
            person.policy = DefaultPolicy(person,policy_cfg)
        elif policy == "orca":
            # check if orca_sim is initialized, if not initialize it
            if not hasattr(self, 'orca_sim'):
                # TODO: Get parameters from omnigibson.people.Person
                sim_params = [v for k,v in self.orca_sim_hyper_params.items()]
                self.orca_sim = rvo2.PyRVOSimulator(*sim_params)
            # bind policy to person
            person.policy = ORCAPolicy(person,policy_cfg)
            
            # Calculate initial velocity towards target
            # TODO: maxSpeed should be from person
            initial_velocity:tuple[float,float] = ORCAPolicy.calculate_pref_velocity(init_pos[:2], person._target_position[:2], self.orca_agent_hyper_params["maxSpeed"])
            
            # add agent to sim with initial velocity
            agent_params = [v for k,v in self.orca_agent_hyper_params.items()]
            agent:int = self.orca_sim.addAgent(tuple(init_pos[:2]), *agent_params, initial_velocity)
            self.orca_sim.setAgentPrefVelocity(agent, initial_velocity)
            # store agent instance to env
            person.orca_idx = agent
            person.sim = self.orca_sim
        else:
            raise ValueError(f"Invalid policy type: {policy}")
        
        person.idx = len(self.people)
        self.people.append(person)
        return person
    
    def reset(self):
        """
        Reset the environment to its initial state.
        
        Returns:
            np.ndarray: Initial observation
        """
        # Todo: Complete reset function
        self.people = []
        self._load_people()
        return self._get_obs()

    def _get_obs(self):
        """
        Get the current observation of the environment.
        
        Returns:
            np.ndarray: Array containing positions and target positions of all persons
        """
        obs = np.zeros((self.num_persons, 4), dtype=np.float32)
        for i, person in enumerate(self.people):
            pos = person.position[:2]
            target = person.temp_target_position[:2]
            obs[i] = np.concatenate([pos, target])
        return obs

    def step(self, dt):
        """
        Step the environment forward in time.
        
        Args:
            dt (float): Time step size
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # step once when physics steps {{n_physics_timesteps_per_render}} times
        self.step_count += 1
        if self.step_count % self.n_physics_timesteps_per_render == 0:
            # if some people use orca as policy, step the orca sim
            if hasattr(self, 'orca_sim') and self.orca_sim is not None:
                # orca_sim get all people's position and velocity
                for person in self.people:
                    if hasattr(person, 'orca_idx') and person.orca_idx is not None:\
                        # update position and velocity from og env to orca sim
                        pos = tuple(person.position[:2])
                        self.orca_sim.setAgentPosition(person.orca_idx, pos)
                        self.orca_sim.setAgentPrefVelocity(person.orca_idx, ORCAPolicy.calculate_pref_velocity(pos, person._target_position[:2], self.orca_agent_hyper_params["maxSpeed"]))
                # let orca sim do one step simulation
                self.orca_sim.doStep()
            for person in self.people:
                person.policy.step(self.sim_dt)
        # save now position of each person for matplotlib visualization
        self.history_positions_x.append([person.position[0] for person in self.people])
        self.history_positions_y.append([person.position[1] for person in self.people])
        
        done = False
        obs = None
        reward = 0
        info = {}
        # visualize in 5000th step
        if self.step_count == 5000:
            self.visualize()
        return obs, reward, done, info

    def visualize(self):
        """
        Visualize the environment using matplotlib.
        """
        plt.figure(figsize=(10, 10))
        plt.scatter(self.history_positions_x, self.history_positions_y)
        # save as png
        plt.savefig("people_sim_env.png")

people_sim_env = PeopleEnv(num_persons=2, area_size=(100, 100))

og.sim.enable_viewer_camera_teleoperation()

while True:
    og.sim.step()
    # if have action, call `og_env.step(action)`
