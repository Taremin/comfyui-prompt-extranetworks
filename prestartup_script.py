import nodes
import os
import sys
import importlib
import inspect

dir = os.path.dirname(__file__)
me = os.path.basename(dir)
path = os.path.join(dir, "data.py")

print_orig = print


def print(*args):
    print_orig(*[f"[{me}]:", *args])


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


data = load_module(path)

if hasattr(nodes, "load_custom_nodes"):
    load_custom_nodes = nodes.load_custom_nodes
elif hasattr(nodes, "init_external_custom_nodes"):
    load_custom_nodes = nodes.init_external_custom_nodes
else:
    print("Unsupported ComfyUI version")
    raise AttributeError


def on_custom_nodes_loaded(function):
    print(f"register on custom nodes loaded callback: {function}")
    if callable(function):
        data.on_custom_nodes_loaded_callbacks.append(function)


def hooked_load_custom_nodes(*args):
    retval = load_custom_nodes(*args)

    print("on custom nodes loaded")
    data.NODE_CLASS_MAPPINGS = nodes.NODE_CLASS_MAPPINGS
    for callback in data.on_custom_nodes_loaded_callbacks:
        callback(data.NODE_CLASS_MAPPINGS)

    return retval


async def hooked_async_init_external_custom_nodes(*args):
    retval = await load_custom_nodes(*args)

    print("on custom nodes loaded")
    data.NODE_CLASS_MAPPINGS = nodes.NODE_CLASS_MAPPINGS
    for callback in data.on_custom_nodes_loaded_callbacks:
        callback(data.NODE_CLASS_MAPPINGS)

    return retval


if hasattr(nodes, "load_custom_nodes"):
    nodes.load_custom_nodes = hooked_load_custom_nodes
elif hasattr(nodes, "init_external_custom_nodes"):
    if inspect.iscoroutinefunction(load_custom_nodes):
        nodes.init_external_custom_nodes = hooked_async_init_external_custom_nodes
    else:
        nodes.init_external_custom_nodes = hooked_load_custom_nodes
else:
    print("Unsupported ComfyUI version")
    raise AttributeError
