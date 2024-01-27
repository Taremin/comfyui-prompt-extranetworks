import nodes
import os
import sys
import importlib


def load_module(path: str):
    module_name = os.path.splitext(os.path.basename(path))[0]
    module_key = os.path.splitext(".".join(path.split(os.path.sep)[-2:]))[0]

    if module_key in sys.modules:
        module = sys.modules[module_key]
    else:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_key] = module
        spec.loader.exec_module(module)

    return module


path = os.path.join(os.path.dirname(__file__), "data.py")
data = load_module(path)
load_custom_nodes = nodes.load_custom_nodes


def on_custom_nodes_loaded(function):
    if callable(function):
        data.on_custom_nodes_loaded_callbacks.append(function)


def hooked_load_custom_nodes(*args):
    retval = load_custom_nodes(*args)

    data.NODE_CLASS_MAPPINGS = nodes.NODE_CLASS_MAPPINGS
    for callback in data.on_custom_nodes_loaded_callbacks:
        callback(data.NODE_CLASS_MAPPINGS)

    return retval


nodes.load_custom_nodes = hooked_load_custom_nodes
