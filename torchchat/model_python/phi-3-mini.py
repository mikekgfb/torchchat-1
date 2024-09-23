
import torch
import torch.nn as nn
import types

from transformers import AutoModelForCausalLM # , AutoTokenizer, pipeline
from torchchat.model import ModelArgs, ModelType, TextOnlyModel, TransformerArgs

class ModelWrapper(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    def forward(self, *args, **kwargs) -> torch.Tensor:
        outputs = self.model.forward(*args, **kwargs)
        return outputs.logits

    def setup_caches(self, max_batch_size, dtype):
        if hasattr(self.model, "setup_caches"):
            self.model.setup_caches(max_batch_size, dtype)
        else:
            print(f"setup caches for {self} ignored")

    def reset_caches(self):
        if hasattr(self.model, "reset_caches"):
            self.model.reset_caches()
        else:
            print(f"reset caches for {self} ignored")


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

    model = ModelWrapper(TransformerArgs(), model)

    model = TextOnlyModel(model_config, {"text" : model})
    print(model)
    
    return model


