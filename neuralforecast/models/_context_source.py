"""Load pinned, explicitly supplied official model sources, without vendoring.

All imported local source files and Aurora's constructor JSONs are hash checked.
This is namespace isolation and compatibility checking, not a security sandbox.
No source/checkpoint downloads or package installations happen in this loader.
"""

import ast
import hashlib
import importlib.machinery
from pathlib import Path
import sys
import threading
import types
from typing import Any

SOURCES: dict[str, dict[str, Any]] = {'VoT': {'repository': 'decisionintelligence/VoT',
         'revision': 'f769621da9efde3e82a1975f478efdec83dcbd10',
         'entry': 'models/PatchTST_clip.py',
         'files': {'layers/Transformer_EncDec.py': 'dabf4c2a5e1e11a04eb4ed67785033dd259ee738',
                   'utils/masking.py': 'a19cbf63b8d1d1927eceabcbe4a1b5313238b75b',
                   'layers/SelfAttention_Family.py': '584fbed44d0347fcfd798654e9ac06a368eba2f7',
                   'layers/Embed.py': '977e25568d37b9dd0efd442dcc5b33eab9843d71',
                   'models/PatchTST_clip.py': 'e0f1c00b07a9b49a2a49093cb84cbfddb8440c72'}},
 'GPT4MTS': {'repository': 'Flora-jia-jfr/GPT4MTS-Prompt-based-Large-Language-Model-for-Multimodal-Time-series-Forecasting',
             'revision': 'c85e74b2fd7048f5ef601453e5bcb9f6073cd3b4',
             'entry': 'models/GPT4MTS.py',
             'files': {'utils/rev_in.py': 'd0b19c42d0dde9be97382d81488cd7fcc88fb052',
                       'models/GPT4MTS.py': 'daa340a478d882ff90c1d935153319131e31fe82'}},
 'UniTime': {'repository': 'liuxu77/UniTime',
             'revision': '09acfbe9c63fc67db7539590094ac5f328230654',
             'entry': 'models/unitime.py',
             'files': {'models/unitimegpt2.py': '333c59c6f5e9431f7233cc72fb4819b47615b180',
                       'models/unitime.py': '39f34b47309ff555657837b453cdb05cbdd0b12f'}},
 'LangTime': {'repository': 'niuwz/LangTime',
              'revision': '1feac37753b221a99c293eb0fa7d223a841283b3',
              'entry': 'models/LangTime.py',
              'files': {'layers/Embed.py': '378d8e5dd3fc2c44f182a42a5912155e310e2c63',
                        'utils/masking.py': '5f62b59b939fedc07e79d7a72ab974d801d24d82',
                        'layers/SelfAttention_Family.py': '188f439907378f9ba8c218c282c8076389aae211',
                        'layers/Transformer_EncDec.py': '900a3a8ac217ed98b980513caf08784038ee0bfb',
                        'layers/Norms.py': '2d3f475c7f72842646b9b2c4fec232fa9ada5740',
                        'layers/Adapters.py': 'af04c5d8e4887b33ce7c3de42816b6f40362b4c6',
                        'configs/log_config.py': '9cee8c8cecdd7bfebcad312a0ce281832836e77e',
                        'models/LangTime.py': 'cfce491d30176daffd76712d9825ac78646887e7'}},
 'Aurora': {'repository': 'decisionintelligence/Aurora',
            'revision': 'a247760abbc9d17a861bc365c032368d317815f2',
            'entry': 'aurora/modeling_aurora.py',
            'files': {'aurora/bert_config/config.json': '45a2321a7ecfdaaf60a6c1fd7f5463994cc8907d',
                      'aurora/bert_config/tokenizer.json': '949a6f013d67eb8a5b4b5b46026217b888021b88',
                      'aurora/bert_config/tokenizer_config.json': 'e5c73d8a50df1f56fb5b0b8002d7cf4010afdccb',
                      'aurora/vit_config/config.json': '254071bbf72dd0fd535b61768d9cd87adcb776f4',
                      'aurora/vit_config/preprocessor_config.json': '70fbc148eb26a06bac351d46fddc0a23037b4ce4',
                      'aurora/configuration_aurora.py': '88b12160341cc0de372b6f45b34b249395a1c6ff',
                      'aurora/util_functions.py': '263e97985904024130f354f37a58b1ae83c162ed',
                      'aurora/flow_loss.py': 'f40a186ddf6147246f778f04b464793a4a399dee',
                      'aurora/modality_connector.py': '3feae84d69ac66faf9d6f1fafa2b61eb488ea54b',
                      'aurora/prototype_retriever.py': '387a431a7330e525b9fe2fbff8c8e7cdfacaab7c',
                      'aurora/ts_generation_mixin.py': '86f0495728d510fe78805bdde7e1baaab56872e9',
                      'aurora/modeling_aurora.py': '8ba49f55bf45f72d06e55bcade5c760d4a9e626d'}},
 'ChatTime': {'repository': 'ForestsKing/ChatTime',
              'revision': '8c2d4c209302d2b2bd6cc3c154586842341e3247',
              'entry': 'model/model.py',
              'files': {'utils/prompt.py': '89f65f4609266f0dde0e2bce9ba00ece9850f61f',
                        'utils/tools.py': 'ec41d9b88296a197d12fb7d811f4f8e6afe7d79e',
                        'model/model.py': 'af932802db789a7293b2a4bf35eec1877b8db379'}}}

# Reviewed compatibility fixes, not replacement forecasting architectures.
_PATCHES = {
    ("GPT4MTS", "models/GPT4MTS.py"): [
        ("GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True, cache_dir='/drive2/florajia/huggingface_cache/')",
         "GPT2Model.from_pretrained(configs.backbone_path, local_files_only=True, use_safetensors=True)"),
        ("GPT2Config()", "GPT2Config(**configs.backbone_config)"),
        ("summary.mean(dim=-1).squeeze()", "summary.mean(dim=-1)"),
    ],
    ("UniTime", "models/unitime.py"): [
        ("from_pretrained(args.model_path)", "from_pretrained(args.model_path, local_files_only=True)"),
    ],
    ("LangTime", "models/LangTime.py"): [
        ("model_args.backbone_config.hidden_size = 768", "model_args.backbone_config.hidden_size = config.hidden_size"),
    ],
    ("ChatTime", "model/model.py"): [
        ("low_cpu_mem_usage=True,", "low_cpu_mem_usage=True, local_files_only=True, use_safetensors=True,"),
        ("torch_dtype=torch.float16", "torch_dtype=torch.float32"),
        ('device_map="auto"', 'device_map=None'),
        ("trust_remote_code=True", "local_files_only=True"),
        ("np.NaN", "np.nan"),
    ],
    ("ChatTime", "utils/tools.py"): [("np.NaN", "np.nan")],
}
_LOCK = threading.RLock()


def official_module(source_dir, kind):
    """Load reviewed Python in dependency order using private package names."""
    root = Path(source_dir).expanduser().resolve()
    spec = SOURCES[kind]
    contents = {}
    for relative, expected in spec["files"].items():
        filename = (root / relative).resolve()
        if not filename.is_relative_to(root) or not filename.is_file():
            raise FileNotFoundError(f"Missing trusted {kind} source: {filename}")
        data = filename.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != expected:
            raise ValueError(f"Changed {kind} source {relative}; use revision {spec['revision']}.")
        if relative.endswith(".py"):
            contents[relative] = data.decode("utf-8")
    prefix = "_nf_context_" + hashlib.sha256((kind + str(root)).encode()).hexdigest()[:20]
    entry = prefix + "." + spec["entry"].removesuffix(".py").replace("/", ".")
    local_roots = {Path(p).parts[0] for p in contents}
    with _LOCK:
        if entry in sys.modules:
            return sys.modules[entry]
        created = []
        try:
            for relative in contents:
                parts = Path(relative).parts
                for depth in range(len(parts)):
                    name = ".".join([prefix] + list(parts[:depth]))
                    if name not in sys.modules:
                        module = types.ModuleType(name)
                        module.__path__ = [str(root.joinpath(*parts[:depth]))]
                        module.__package__ = name
                        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
                        sys.modules[name] = module
                        created.append(name)
            for relative, text in contents.items():
                for old, new in _PATCHES.get((kind, relative), []):
                    if old not in text:
                        raise ValueError(f"Missing reviewed patch target in {relative}.")
                    text = text.replace(old, new)
                tree = ast.parse(text, filename=str(root / relative))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        if node.module.split(".")[0] in local_roots:
                            node.module = prefix + "." + node.module
                    elif isinstance(node, ast.Import):
                        if any(n.name.split(".")[0] in local_roots for n in node.names):
                            raise ImportError("Unexpected local import; review the official source.")
                name = prefix + "." + relative.removesuffix(".py").replace("/", ".")
                module = types.ModuleType(name)
                module.__file__ = str(root / relative)
                module.__package__ = name.rsplit(".", 1)[0]
                module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
                sys.modules[name] = module
                created.append(name)
                exec(compile(tree, module.__file__, "exec"), module.__dict__)
            return sys.modules[entry]
        except Exception:
            for name in reversed(created):
                sys.modules.pop(name, None)
            raise
