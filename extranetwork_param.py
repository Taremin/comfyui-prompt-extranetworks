import re
from re import Match

extra_networks_pattern = re.compile(
    r"(?<![\\])<\s*(\w+)\s*:\s*(.*?)(?<!\\)>", flags=re.M | re.S
)


def prepare_search(s):
    return re.sub(r"\\\\", r"\0", s)


def post_search(s):
    return re.sub(r"\0", r"\\\\", s)


def strip_escape(s):
    s = s.encode().decode("unicode_escape")
    s = re.sub(r"\\([^\n])", r"\1", s)
    s = re.sub(r"\0", r"\\", s)
    return s


class ExtraNetworksParam:
    def __init__(self, items=[]):
        self.positional = []
        self.named = {}

        for item in items:
            parts = [strip_escape(part) for part in re.split(r"(?<!\\)=", item, 2)]
            # print("extra param:", parts)
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

    @classmethod
    def parse(self, prompt: str, target: list[str] | None = None):
        extra_networks = {}

        def on_match(match: Match):
            groups = match.groups()

            if target is not None and groups[0] not in target:
                return match.group(0)

            if groups[0] not in extra_networks:
                extra_networks[groups[0]] = []
            extra_networks[groups[0]].append(
                ExtraNetworksParam(items=re.split(r"\s*(?<!\\):\s*", groups[1]))
            )
            return ""

        prompt = prepare_search(prompt)
        replaced = extra_networks_pattern.sub(repl=on_match, string=prompt)
        prompt = post_search(prompt)

        return (replaced, extra_networks)
