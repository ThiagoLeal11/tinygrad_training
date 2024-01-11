import math
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import tqdm

import wandb
from bert_modeling import BertConfig, BertForPreTraining
from extra.datasets.wikipedia import load_dataset
from extra.lr_scheduler import OneCycleLR
from tinygrad import nn, Tensor, GlobalCounters
from tinygrad.helpers import getenv
from tinygrad.jit import TinyJit
from multiprocessing import Process, Queue


checkpoint_root = '/home/main/projects/tinygrad_training/checkpoints'


@dataclass
class HyperParameters:
  # TODO custom input files
  # TODO custom output chekpoints
  # TODO custom bert config
  train_batch_size: int = getenv("BS", 24)
  eval_batch_size: int = getenv("EVAL_BS", 8*8)  # 64

  train_steps: int = getenv("TRAIN_STEPS", 107538)  # Number of examples per training epoch
  eval_steps: int = getenv("EVAL_STEPS", math.floor(1250/8))  # Number of examples for validation
  warmup_steps: int = getenv("WARMUP_STEPS", 1562)
  iter_per_loop: int = getenv("ITER_PER_LOOP", 6250)  # TODO: what is this?
  # iter_per_loop: int = getenv("ITER_PER_LOOP", 10)

  # epochs: int = getenv("EPOCHS", 30)  # Find why?
  lr: float = getenv("LR", 0.0001)

  # TODO: num_gpus=getenv("NUM_GPUS", 1),
  # TODO: --max_predictions_per_seq=76
  # TODO: --max_seq_length=512
  # TODO: --save_checkpoints_steps=6250
  # TODO: --iterations_per_loop? what is this?


hyp = HyperParameters()


def to_tensor(features: dict[str, np.ndarray]) -> dict[str, Tensor]:
  return {
    k: Tensor(v, requires_grad=False)  # TODO: Require_grads = False?
    for k, v in features.items()
  }


@TinyJit
def evaluate_step(model: BertForPreTraining, features: dict[str, Tensor]):
  outputs = model(
    input_ids=features["input_ids"], token_type_ids=features['segment_ids'],
    attention_mask=features["input_mask"]
  )
  prediction_scores, seq_relationship_score = [o.realize().numpy() for o in outputs]
  masked_lm_labels = features["masked_lm_labels"].realize().numpy()
  next_sentence_label = features["next_sentence_labels"].realize().numpy()
  acc = (
    np.sum(np.argmax(prediction_scores, axis=2) == masked_lm_labels) / masked_lm_labels.size +
    np.sum(np.argmax(seq_relationship_score, axis=1) == next_sentence_label) / next_sentence_label.size
  ) / 2
  return acc


@TinyJit
def train_step(model: BertForPreTraining, opt, lr, features: dict[str, Tensor]):
  # num_acc_steps = 24
  # size = int(features['input_ids'].shape[0] / num_acc_steps)
  opt.zero_grad()
  # for i in range(num_acc_steps):
  #   s, e = i * size, (i + 1) * size
  #   print(f'accum {i}:', s, e)
  loss = model(
    input_ids=features["input_ids"], token_type_ids=features['segment_ids'],
    attention_mask=features["input_mask"], masked_lm_labels=features["masked_lm_labels"],
    next_sentence_label=features["next_sentence_labels"]
  )  # / float(num_acc_steps)
  loss.backward()
    # loss = loss.realize()
    # print('done')
  opt.step()
  lr.step()
  return loss.realize()


def log_stats(step: int, st: time.time, et: time.time, ct: time.time, loss: float, lr: float, world_count: int, log_type: str = "train"):
  run_time = (ct - st) * 1000.0
  python_time = (et - st) * 1000.0
  cpu_time = (ct - et) * 1000.0
  mem = world_count * GlobalCounters.mem_used / 1e9
  gflops = world_count * GlobalCounters.global_ops * 1e-9 / run_time
  stats = (
    f"{step:3d} {run_time:7.2f} ms run, {python_time:7.2f} ms python, {cpu_time:7.2f} ms CPU, "
    f"{loss:7.2f} loss, {lr:.6f} LR, {mem:.2f} GB used, {gflops:9.2f} GFLOPS"
  )
  loss_or_acc = "loss" if log_type == "train" else "acc"
  wandb.log({log_type: {
    "run_time": run_time, "python_time": python_time, "cl_time": cpu_time, loss_or_acc: loss, "lr": lr, "mem_used": mem,
    "gflops": gflops
  }})
  return stats


def eval_fase(bert_config: BertConfig, checkpoint_path: str, step: int, signal: Queue):
  print(f'>> Evaluating {step}')
  model = BertForPreTraining(bert_config)
  eval_dataset = load_dataset(bs=hyp.eval_batch_size, is_validation=True)
  if checkpoint_path:
    load_checkpoint(step, model, None)

  Tensor.training = False
  pbar = tqdm.trange(hyp.eval_steps, desc="Eval")
  start_time = time.monotonic()
  accs = [
    evaluate_step(model, to_tensor(next(eval_dataset)[0]))
    for _ in pbar
  ]
  end_time = time.monotonic()
  avg_acc = sum(accs) / len(accs)
  stats = log_stats(step, start_time, end_time, end_time, avg_acc, 0, 1, "eval")
  print(f'Eval = {stats}')
  signal.put(avg_acc)


def train_fase(train_dataset, bert_config: BertConfig, checkpoint_path: str, step: int, signal: Queue):
  print(f'>> Training {step}')
  model = BertForPreTraining(bert_config)
  # TODO: almost the same as adamw except for the adam=False
  optimizer = nn.optim.LAMB(
    params=nn.state.get_parameters(model), lr=hyp.lr, b1=0.9, b2=0.999, eps=1e-6, wd=0.01, adam=False
  )
  # TODO: Exclude LayerNorm, and bias from weight decay
  lr_scheduler = OneCycleLR(
    optimizer, max_lr=hyp.lr, div_factor=1e6, final_div_factor=1e12,
    total_steps=hyp.train_steps, pct_start=hyp.warmup_steps / hyp.train_steps
  )

  # Try to load
  if checkpoint_path:
    load_checkpoint(step, model, optimizer)

  for _ in tqdm.trange(step, desc='LR', disable=step == 0):
    lr_scheduler.step()

  Tensor.training = True
  pbar = tqdm.trange(hyp.iter_per_loop, desc='Train')
  for i in pbar:
    start_time = time.monotonic()
    GlobalCounters.reset()  # Count memory usage and ops
    loss = train_step(model, optimizer, lr_scheduler, to_tensor(next(train_dataset)[0]))
    end_time = time.monotonic()

    loss_cpu = loss.numpy()
    cpu_time = time.monotonic()

    stats = log_stats(step + i, start_time, end_time, cpu_time, loss_cpu, optimizer.lr.numpy()[0], 1, "train")
    pbar.set_description_str(f'Train = {stats}')
  step += hyp.iter_per_loop

  create_checkpoint(step, model, optimizer)
  signal.put('Done')


def create_checkpoint(step: int, model: BertForPreTraining, opt: nn.optim):
  model.save(checkpoint_root + f"/model_{step}.pt")
  nn.state.safe_save(nn.state.get_state_dict(opt), checkpoint_root + f"/opt_{step}.pt")


def load_checkpoint(step: int, model: BertForPreTraining, opt: nn.optim = None):
  model.load(checkpoint_root + f"/model_{step}.pt")
  if opt:
    nn.state.load_state_dict(opt, nn.state.safe_load(checkpoint_root + f"/opt_{step}.pt"), verbose=False)


def pretrain_bert():
  # TODO: make this configurable
  BASE_PATH = pathlib.Path(__file__).parent.parent.parent.parent
  config_file = BASE_PATH / "extra" / "datasets" / "wiki" / "bert_config.json"

  bert_config = BertConfig.from_json(config_file)
  # wandb.init(project="pretrain_bert_tinygrad", config=dict(model=bert_config.__dict__, hyperparameters=hyp.__dict__))
  wandb.init(project="burn", config=dict(model=bert_config.__dict__, hyperparameters=hyp.__dict__))

  train_dataset = load_dataset(bs=hyp.train_batch_size)

  signal = Queue()

  step = 0
  prev_checkpoint = False
  while step < hyp.train_steps:
    train = Process(target=train_fase, args=(train_dataset, bert_config, prev_checkpoint, step, signal))
    train.start()
    _ = signal.get()
    train.terminate()

    prev_checkpoint = True
    step += hyp.iter_per_loop

    test = Process(target=eval_fase, args=(bert_config, prev_checkpoint, step, signal))
    test.start()
    acc = signal.get()
    test.terminate()

    if acc >= 0.72:
      print('training completed')
      break


if __name__ == "__main__":
  # Login to wheigth and biases (https://wandb.ai/authorize)
  wandb.login(key='ae38d649d60523043a4ffdc34a31b2b521758ede')

  with Tensor.train():
    pretrain_bert()
