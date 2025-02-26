import omnigibson as og
from omnigibson.macros import gm

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

################################
# import people extension
from omnigibson.people import Person, PersonController,PeopleManager
import numpy as np
    
class CirclePersonController(PersonController):
    def __init__(self):
        super().__init__()

        self._radius = 5.0
        self.gamma = 0.0
        self.gamma_dot = 0.3

    def update(self, dt: float):

        # Update the reference position for the person to track
        self.gamma += self.gamma_dot * dt

        # Set the target position for the person to track
        self._person.update_target_position(
            [self._radius * np.cos(self.gamma), self._radius * np.sin(self.gamma), 0.0])
    
person_controller = CirclePersonController()
p1 = Person("person1", "original_male_adult_construction_05", init_pos=[
                3.0, 0.0, 0.0], init_yaw=1.0, controller=person_controller)

p2 = Person("person2", "original_female_adult_business_02",
            init_pos=[2.0, 0.0, 0.0])

print(PeopleManager._people)

p2.update_target_position([10.0, 0.0, 0.0], 1.0)

og.sim.enable_viewer_camera_teleoperation()

while True:
    og.sim.step()
