import os
import importlib


ENVS = {}


def register_env(name):
    """Registers a env by name for instantiation in rlkit."""

    def register_env_fn(fn):
        if name in ENVS:
            raise ValueError("Cannot register duplicate env {}".format(name))
        if not callable(fn):
            raise TypeError("env {} must be callable".format(name))
        ENVS[name] = fn
        return fn

    return register_env_fn


# automatically import any envs in the envs/ directory
# for file in os.listdir(os.path.dirname(__file__)):
#     if file.endswith('.py') and not file.startswith('_'):
#         module = file[:file.find('.py')]
# 'hopper_rand_params_wrapper', 'walker_rand_params_wrapper'
modules = ['half_cheetah_dir'] #'half_cheetah_dir', humanoid_dir3
for module in modules:
    importlib.import_module('rlkit.envs.' + module)
# importlib.import_module('rlkit.envs.' + module)
