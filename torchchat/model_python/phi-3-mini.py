
import torch
import torch.nn as nn
import types
from typing import Optional

from transformers import AutoModelForCausalLM # , AutoTokenizer, pipeline
from torchchat.model import ModelArgs, ModelType, TextOnlyModel, TransformerArgs

class ModelWrapper(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model.eval()

    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            # print(f"args: {args} kwargs: {kwargs}")
            attention_mask=torch.ones(dtype=torch.int64, size=(1,input_pos[-1]+1))
            outputs = self.model.forward(input_ids=x, position_ids=input_pos.unsqueeze(0), attention_mask=attention_mask)
            #print(f"outputs.shape: {outputs.logits.shape}")
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
        do_sample=False,
    )

    # let's get a default config SentencePiece
    # PS: Mostly default values, but we assert it in the constructor for documentation
    model_config = ModelArgs(
        transformer_args={},
        model_type=ModelType.TextOnly,
        use_tiktoken=False)

    from types import MethodType

    def forward(self, *args, **kwargs):
        print(f"args {args} kwargs: {kwargs}")
        output = self.orig_forward(*args, **kwargs)
        print(f"output.logits: {output.logits}")
        return output

    model.orig_forward = model.forward
    model.forward = MethodType(forward, model)

    model = ModelWrapper(TransformerArgs(), model)

    model = TextOnlyModel(model_config, {"text" : model})
    print(model)
    
    return model


