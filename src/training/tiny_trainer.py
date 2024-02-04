import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

from data.tiny_dataset import TinyTextDataset
from core.model_config import ModelConfig
from core.base_model import BaseModel
from models.casual_head import CasualHead
from utils import get_logging

def main():
    init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
    ck_interval = 1
    ck_iter_num = 0

    build_dir = os.path.join(os.path.dirname(__file__), '../../build')
    os.makedirs(build_dir, exist_ok=True)
    iter_num = 0

    logging =get_logging()
    logging.info("Training a new model from scratch!")
    

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

    tokens_per_iter = config.batch_size * config.n_positions
    logging.info(f"tokens per iteration will be: {tokens_per_iter:,}")

    t0 = time.time()

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
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    logging.info(f"loading dataset  time {dt*1000:.2f}ms")

    lm_head = CasualHead(config)
    model = BaseModel(config, lm_head)

    # init a new model from scratch
    logging.info("Initializing a new model from scratch")
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
        
    if init_from == 'resume':
        logging.info(f"Resuming training from {build_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(build_dir, 'ckpt.pt')

        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            ck_iter_num = checkpoint['iter_num']
            
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)

            optimizer.load_state_dict(checkpoint['optimizer'])
            checkpoint = None # free up memory

    elif init_from.startswith('gpt2'):
        logging.info(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=0.0)
        from fine_tuning.small_gpt import from_pretrained
        model = from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'n_positions', 'bias', 'vocab_size']:
            config[k] = getattr(model.config, k)



    logging.info("compiling the model... (takes a ~minute)")

    model = torch.compile(model) # requires PyTorch 2.0
    t1 = time.time()
    dt = t1 - t0
    logging.info(f"compiling time {dt*1000:.2f}ms")

    # Loop for the given number of epochs
    for epoch in range(config.epochs):
        model.train()
        data_iter = iter(tiny_dataloader)
        while True:
            t0 = t1
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(tiny_dataloader)
                batch = next(data_iter)

            if ck_iter_num <= iter_num:
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

                if iter_num % ck_interval == 0:
                    logging.info(f"save model for checkup at step {iter_num}")
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                    }
                    print(f"saving checkpoint to {build_dir}")
                    torch.save(checkpoint, os.path.join(build_dir, 'ckpt.pt'))

                lossf = loss.item() 

                # timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                logging.info(f"epoch {epoch}   iter {iter_num}:  loss {lossf:.4f},  time {dt*1000:.2f}ms")

            iter_num += 1


if __name__ == '__main__':
    main()