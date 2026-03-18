
from pxr import Usd,UsdShade, Sdf
import numpy as np


def random_material(stage, prim, materials : list):
    materials = [UsdShade.Material.Get(stage, mat.GetPath()) for mat in materials]
    UsdShade.MaterialBindingAPI(prim).Bind(np.random.choice(materials)) 
