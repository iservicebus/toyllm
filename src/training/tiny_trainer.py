import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

from data.tiny_dataset import TinyTextDataset
from core.model_config import ModelConfig
from core.base_model import BaseModel
from models.casual_head import CasualHead

def main():
    config = ModelConfig(

        batch_size = 4,
        epochs = 1,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=8,
        n_inner=5,
    )
    # List available datasets
    tiny_data = load_dataset("tiny_shakespeare")
    text = tiny_data["train"][0]["text"]
    tiny_dataset = TinyTextDataset(config,text)

   # config.vocab_size = tiny_dataset.get_vocab_size()

    config.vocab_size = 50304

    # DataLoaders creation:
    tiny_dataloader = DataLoader(
        tiny_dataset,
        shuffle=True, 
        batch_size=config.batch_size
    )

    lm_head = CasualHead(config)
    model = BaseModel(config, lm_head)


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    # adamw optimizer
    learning_rate = 6e-4 # max learning rate
    weight_decay = 1e-1
    
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)



    # Loop for the given number of epochs
    for epoch in range(config.epochs):
        model.train()
        data_iter = iter(tiny_dataloader)
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(tiny_dataloader)
                batch = next(data_iter)
            batch = [t for t in batch]
            x, y = batch
            out = model(
                    input_ids=x,
                    labels =y,
                    )
            loss = out["loss"]
            logits =out["logits"]
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            optimizer.step()


if __name__ == '__main__':
    main()