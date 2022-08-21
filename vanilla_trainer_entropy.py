import torch
import accelerate
from accelerate import Accelerator
from torch import nn
from models.collator import DataCollatorForWholeWordMask
from datasets import load_from_disk
from torch.utils.data import DataLoader
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EPOCHS = 2
BATCH_SIZE = 20
GRAD_ACCUM_STEPS = 2000 // BATCH_SIZE


def get_dataloaders(tokenizer, path_tokenized_dataset):
    """
    Load loaders with data
    Args:
        tokenizer (transformers.AutoTokenizer): Data tokenizer
        path_tokenized_dataset (str): Path to dict of datasets.arrow_dataset.Dataset dataset

    Returns:
          train (torch.utils.data.dataloader.DataLoader): Train dataloader
          test (torch.utils.data.dataloader.DataLoader): Test dataloader
    """
    tokenized_datasets = load_from_disk(path_tokenized_dataset)
    train_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=0.15)
    test_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=0.15)
    train_set = tokenized_datasets['train']
    test_set = tokenized_datasets['test']
    train = DataLoader(dataset=train_set, shuffle=True, batch_size=BATCH_SIZE, collate_fn=train_collator)
    test = DataLoader(dataset=test_set, shuffle=False, batch_size=BATCH_SIZE, collate_fn=test_collator)

    return train, test


def run():
    """
    Setup experiment with plain MLM training and train
    """
    # init accelerator
    from accelerate import DeepSpeedPlugin
    # deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=2)
    accelerator = Accelerator()
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    device = accelerator.device

    from transformers import AutoTokenizer, AutoModelForMaskedLM, set_seed

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    teacher = AutoModelForMaskedLM.from_pretrained("allegro/herbert-base-cased")
    train_loader, test_loader = get_dataloaders(tokenizer, 'data/tokenized_dataset_demo2')

    # set accelerator
    from transformers import AdamW, get_cosine_schedule_with_warmup
    from trainers.utils_model import configure_optimizer

    deepspeed_cond = lambda x: True#deepspeed_plugin is None or x not in deepspeed_plugin.deepspeed_config
    optim_wrapper = AdamW if deepspeed_cond("optimizer") else accelerate.utils.DummyOptim
    # from deepspeed.ops.lamb import FusedLamb
    # optim_wrapper = FusedLamb
    optim = configure_optimizer(optim_wrapper, teacher, None, lr_backbone=5e-4, lr_head=None, weight_decay=1e-3)
    NUM_TRAINING_STEPS = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    print('Num steps', NUM_TRAINING_STEPS, int(0.2 * NUM_TRAINING_STEPS))
    # from deepspeed.runtime.lr_schedules import WarmupDecayLR
    # lr_scheduler = WarmupDecayLR(optim, warmup_max_lr=5e-5, total_num_steps=NUM_TRAINING_STEPS,
    #                              warmup_num_steps=int(0.01 * NUM_TRAINING_STEPS))
    if deepspeed_cond("scheduler"):
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optim,
            num_cycles=EPOCHS,
            num_warmup_steps=int(0.2 * NUM_TRAINING_STEPS),
            num_training_steps=NUM_TRAINING_STEPS)
    else:
        lr_scheduler = accelerate.utils.DummyScheduler(optim, warmup_max_lr=5e-4, total_num_steps=NUM_TRAINING_STEPS,
                                                    warmup_num_steps=int(0.2 * NUM_TRAINING_STEPS), warmup_type='linear')

    train_loader, test_loader, teacher, optim, lr_scheduler = accelerator.prepare(
        train_loader, test_loader, teacher, optim, lr_scheduler)

    loaders = {'train': train_loader, 'test': test_loader}
    print('optim:', optim)
    print('lr_scheduler:', lr_scheduler)

    from trainers.vanillaTrainer_hf_collator import VanillaTrainer
    params_trainer = {
        'model': teacher,
        'tokenizer': tokenizer,
        'loaders': loaders,
        'criterion': nn.CrossEntropyLoss().to(device),
        'optim': optim,
        'lr_scheduler': lr_scheduler,
        'accelerator': accelerator,
        'device': device
    }
    trainer = VanillaTrainer(**params_trainer)

    import collections
    config_run_epoch = collections.namedtuple('RE', ['save_interval', 'grad_accum_steps', 'running_step'])(110000,
                                                                                                           GRAD_ACCUM_STEPS,
                                                                                                           40)
    params_run = {
        'epoch_start': 0,
        'epoch_end': EPOCHS,
        'exp_name': f'ZeRO2, WWM, teacher_finetuning, lr:5e-4, warmup_p: 0.2',
        'config_run_epoch': config_run_epoch,
        'random_seed': 42
    }
    trainer.run_exp(**params_run)

if __name__ == '__main__':
    run()
