import re
from re import Match
from nodes import LoraLoader
from comfy_extras.nodes_hypernetwork import HypernetworkLoader

extra_networks_pattern = re.compile(r"<(\w+):([^>]+)>")


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
