import re
from re import Match
from nodes import LoraLoader
from comfy_extras.nodes_hypernetwork import HypernetworkLoader
from .prestartup_script import on_custom_nodes_loaded

extra_networks_pattern = re.compile(r"<(\w+):([^>]+)>")

LoraLoaderBlockWeight = None

NAME = "ComfyUI Prompt Extranetworks"


def on_load(mappings: dict):
    for key, value in mappings.items():
        if "LoraLoaderBlockWeight" == value.__name__:
            global LoraLoaderBlockWeight
            LoraLoaderBlockWeight = value

            print(f"[{NAME}] LoraLoaderBlockWeight found")

            return


on_custom_nodes_loaded(on_load)


class ExtraNetworksParam:
    def __init__(self, items=[]):
        self.positional = []
        self.named = {}

        for item in items:
            parts = item.split("=", 2)
            if len(parts) == 2:
                self.named[parts[0]] = self.process_args(parts[1])
            else:
                self.positional.append(self.process_args(item))

    def process_args(self, value: str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


class PromptExtraNetworks:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "forceInput": True},
                ),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
        "STRING",
    )
    FUNCTION = "process"
    CATEGORY = "loaders"

    def process(self, model, clip, prompt):
        replaced_prompt, extra_networks = self.parse(prompt=prompt)

        if "lora" in extra_networks:
            model, clip = self.process_lora(model, clip, extra_networks)
        if "hypernet" in extra_networks:
            (model,) = self.process_hypernetwork(model, extra_networks)

        return (model, clip, replaced_prompt)

    def process_lora(self, model, clip, extra_networks):
        for params in extra_networks["lora"]:
            lora_name = params.positional[0]

            if len(params.positional) == 1:
                strength_model = 1.0
                strength_clip = 1.0
            elif len(params.positional) == 2:
                strength_model = strength_clip = params.positional[1]
            elif len(params.positional) >= 3:
                strength_model = params.positional[1]
                strength_clip = params.positional[2]

            if LoraLoaderBlockWeight is not None and "lbw" in params.named:
                loader = LoraLoaderBlockWeight()
                func_name = (
                    LoraLoaderBlockWeight.FUNCTION
                    if hasattr(LoraLoaderBlockWeight, "FUNCTION")
                    else "execute"
                )
                func = getattr(loader, func_name, None)

                if func is None or not callable(func):
                    continue

                seed = int(params.named["seed"]) if "seed" in params.named else 0
                preset = params.named["lbw"]
                A = float(params.named["A"]) if "A" in params.named else 1.0
                B = float(params.named["B"]) if "B" in params.named else 1.0

                input_types = LoraLoaderBlockWeight.INPUT_TYPES()
                presets = {}
                if "required" in input_types and "preset" in input_types["required"]:
                    for block_vector in input_types["required"]["preset"][0][1:]:
                        key, value = block_vector.split(sep=":", maxsplit=2)
                        presets[key] = value
                if preset in presets:
                    print(f"[{NAME}] preset: {preset}={presets[preset]}")
                    preset = presets[preset]
                else:
                    print(f"[{NAME}] block weight:", preset)

                try:
                    model, clip, populated_vector = func(
                        model=model,
                        clip=clip,
                        lora_name=lora_name,
                        strength_model=strength_model,
                        strength_clip=strength_clip,
                        inverse=False,
                        seed=seed,
                        A=A,
                        B=B,
                        preset=preset,
                        block_vector=preset,
                    )
                except Exception as e:
                    print("LoraBlockWeightLoader Error:", e)
                    print(f"\tFile: {lora_name}")
            else:
                loader = LoraLoader()
                func_name = (
                    LoraLoader.FUNCTION
                    if hasattr(LoraLoader, "FUNCTION")
                    else "execute"
                )
                func = getattr(loader, func_name, None)

                if func is None or not callable(func):
                    continue

                try:
                    model, clip = func(
                        model=model,
                        clip=clip,
                        lora_name=lora_name,
                        strength_model=strength_model,
                        strength_clip=strength_clip,
                    )
                except Exception as e:
                    print("LoraLoader Error:", e)
                    print(f"\tFile: {lora_name}")

        return (model, clip)

    def process_lora_block_weight(self, model, clip, extra_networks):
        for params in extra_networks["lora"]:
            loader = LoraLoader()
            func_name = (
                LoraLoader.FUNCTION if hasattr(LoraLoader, "FUNCTION") else "execute"
            )
            func = getattr(loader, func_name, None)

            if func is None or not callable(func):
                continue

            lora_name = params.positional[0]

            if len(params.positional) == 1:
                strength_model = 1.0
                strength_clip = 1.0
            elif len(params.positional) == 2:
                strength_model = strength_clip = params.positional[1]
            elif len(params.positional) >= 3:
                strength_model = params.positional[1]
                strength_clip = params.positional[2]

            try:
                model, clip = func(
                    model=model,
                    clip=clip,
                    lora_name=lora_name,
                    strength_model=strength_model,
                    strength_clip=strength_clip,
                )
            except Exception as e:
                print("LoraLoader Error:", e)
                print(f"\tFile: {lora_name}")

        return (model, clip)

    def process_hypernetwork(self, model, extra_networks):
        for params in extra_networks["hypernet"]:
            loader = HypernetworkLoader()
            func_name = (
                HypernetworkLoader.FUNCTION
                if hasattr(LoraLoader, "FUNCTION")
                else "execute"
            )
            func = getattr(loader, func_name, None)

            if func is None or not callable(func):
                continue

            hypernetwork_name = params.positional[0]

            if len(params.positional) == 1:
                strength = 1.0
            elif len(params.positional) >= 2:
                strength = params.positional[1]

            try:
                model = func(
                    model=model,
                    hypernetwork_name=hypernetwork_name,
                    strength=strength,
                )
            except Exception as e:
                print("HypernetworkLoader Error:", e)
                print(f"\tFile: {hypernetwork_name}")

        return (model,)

    def parse(self, prompt: str):
        extra_networks = {}

        def on_match(match: Match):
            groups = match.groups()
            if groups[0] not in extra_networks:
                extra_networks[groups[0]] = []
            extra_networks[groups[0]].append(
                ExtraNetworksParam(items=groups[1].split(":"))
            )
            return ""

        replaced = extra_networks_pattern.sub(repl=on_match, string=prompt)
        return (replaced, extra_networks)


NODE_CLASS_MAPPINGS = {"PromptExtraNetworks": PromptExtraNetworks}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptExtraNetworks": "PromptExtraNetworks"}
