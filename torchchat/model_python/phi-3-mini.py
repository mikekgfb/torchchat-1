
import torch
import torch.nn as nn
import types

from transformers import AutoModelForCausalLM # , AutoTokenizer, pipeline
from torchchat.model import ModelArgs, ModelType, TextOnlyModel, TransformerArgs

def model_builder(builder_args) -> nn.Module:
    torch.random.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # let's get a default config SentencePiece
    # PS: Mostly default values, but we assert it in the constructor for documentation
    model_config = ModelArgs(
        transformer_args={},
        model_type=ModelType.TextOnly,
        use_tiktoken=False)
    model.config = TransformerArgs()
    import types

    def setup_caches(self, max_batch_size, dtype):
        print(f"setup caches for {self}, no-op")

    model.setup_caches = types.MethodType(setup_caches, model)
    print(model)

    model = TextOnlyModel(model_config, {"text" : model})
    print(model)
    
    return model


