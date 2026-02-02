
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
from isaacsim.core.api import World

import LulaRRT_ex as ex


world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.01,
                rendering_dt = 0.01)



robot = ex.FrankaRrtExample()
robot.load_example_assets()
world.reset()
robot.setup()

robot.reset()
robot._articulation.initialize()
# world.reset()
world.stop()
while True:
    world.step(render=True)
    if world.is_playing():
        robot.update(1)
