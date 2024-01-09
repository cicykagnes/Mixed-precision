from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
import torch
import time
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

block_size = 128
mixed_precision = True

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
dataset = load_dataset('databricks/databricks-dolly-15k')

def preprocess(row): # row is a list because you're batching
    return tokenizer([' '.join(x) for x in zip(row['instruction'], row['response'])])

tokenized = dataset.map(
    preprocess,
    batched=True,
    num_proc=4,
    #remove_columns=['context', 'category'], # just for debugging
    remove_columns=dataset['train'].column_names,
)

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # split by chunks of block_size, drop the rest
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result

lm_dataset = tokenized.map(group_texts, batched=True, num_proc=4)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_dataloader = DataLoader(lm_dataset['train'], shuffle=False, batch_size=100, collate_fn=data_collator)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

def get_mem_usage():
    return f'{round(torch.cuda.memory_allocated(device) / 1048576, 2)} MB'
print(f'memory allocated after sending model to device: {get_mem_usage()}')

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    start = time.time()
    num_tokens = 0
    for batch in train_dataloader:
        num_tokens += batch.input_ids.size(0)
        batch = {k: v.to(device) for k, v in batch.items()}
        print(f'memory allocated after sending a batch to device: {get_mem_usage()}')
        if mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                print(f'memory allocated after forward pass: {get_mem_usage()}')
                loss = outputs.loss
                print(f'memory allocated after backward pass: {get_mem_usage()}')
                loss.backward()
        else:
            outputs = model(**batch)
            print(f'memory allocated after forward pass: {get_mem_usage()}')
            loss = outputs.loss
            print(f'memory allocated after backward pass: {get_mem_usage()}')
            loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    print('********* one epoch done!')
    print(f'epoch: {epoch}, training loss: {loss}')
    print(f'memory allocated at the end of an epoch: {get_mem_usage()}')
    time_per_epoch = round(time.time() - start, 2)
    print(f'time per epoch: {time_per_epoch} seconds')
    print(f'tokens per second: {num_tokens / time_per_epoch} tokens')
    path = f"model_{'mixed' if mixed_precision else 'full'}_epoch{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, path)
    checkpoint_size = f'{round(os.stat(path).st_size * 1e-9, 2)} GB'
    print(f'checkpoint size: {checkpoint_size}')
