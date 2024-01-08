import pathlib
import time
from dataclasses import dataclass
from tinygrad.helpers import getenv
from bert_modeling import BertConfig, BertForPreTraining
from tinygrad import nn, Tensor, GlobalCounters
from extra.datasets.wikipedia import load_dataset
from extra.lr_scheduler import OneCycleLR
import tqdm



@dataclass
class HyperParameters:
  # TODO custom input files
  # TODO custom output chekpoints
  # TODO custom bert config
  train_batch_size: int = getenv("BS", 24)
  eval_batch_size: int = getenv("EVAL_BS", 8)

  train_steps: int = getenv("TRAIN_STEPS", 107538)  # Number of examples per training epoch
  eval_steps: int = getenv("EVAL_STEPS", 1250)  # Number of examples for validation
  warmup_steps: int = getenv("WARMUP_STEPS", 1562)
  iter_per_loop: int = getenv("ITER_PER_LOOP", 6250)

  epochs: int = getenv("EPOCHS", 30)
  lr: float = getenv("LR", 0.0001)

  # TODO: num_gpus=getenv("NUM_GPUS", 1),
  # TODO: --max_predictions_per_seq=76
  # TODO: --max_seq_length=512
  # TODO: --save_checkpoints_steps=6250
  # TODO: --iterations_per_loop? what is this?


hyp = HyperParameters()


def evaluate_step(model: BertForPreTraining, features: dict[str, Tensor]):
  Tensor.training = False
  loss = model(
    input_ids=features["input_ids"], token_type_ids=features['segment_ids'],
    attention_mask=features["input_mask"], masked_lm_labels=features["masked_lm_labels"],
    next_sentence_label=features["next_sentence_labels"]
  )
  acc = loss.realize()
  Tensor.training = True
  return acc


def train_step(model: BertForPreTraining, opt, lr, features: dict[str, Tensor]):
  # TODO: validate this
  loss = model(
    input_ids=features["input_ids"], token_type_ids=features['segment_ids'],
    attention_mask=features["input_mask"], masked_lm_labels=features["masked_lm_labels"],
    next_sentence_label=features["next_sentence_labels"]
  )
  opt.zero_grad()
  loss.backward()
  opt.step()
  lr.step()
  return loss.realize()


def pretrain_bert():
  # TODO: make this configurable
  BASE_PATH = pathlib.Path(__file__).parent.parent.parent.parent
  config_file = BASE_PATH / "extra" / "datasets" / "wiki" / "bert_config.json"

  bert_config = BertConfig.from_json(config_file)
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

  train_dataset = load_dataset(bs=hyp.train_batch_size)
  eval_dataset = load_dataset(bs=hyp.eval_batch_size, is_validation=True)

  acc_achieved = False
  step = 0

  # TODO: tqdm
  while step < hyp.train_steps or not acc_achieved:
    if step % hyp.iter_per_loop == 0:
      losses = [
        evaluate_step(model, next(eval_dataset)[0])
        for _ in tqdm.trange(hyp.eval_steps)
      ]
      acc = sum(losses) / len(losses)
      print(f"Step: {step}, Loss: {acc}")

    start_time = time.monotonic()
    GlobalCounters.reset()  # Count memory usage and ops
    loss = train_step(model, optimizer, lr_scheduler, next(train_dataset)[0])
    end_time = time.monotonic()

    loss_cpu = loss.numpy()
    cpu_time = time.monotonic()

    run_time = (cpu_time - start_time) * 1000.0
    python_time = (end_time - start_time) * 1000.0
    cpu_time = (cpu_time - end_time) * 1000.0
    world_size = 1  # For distributed training.
    lr = optimizer.lr.numpy()[0]
    mem = world_size * GlobalCounters.mem_used / 1e9
    gflops = world_size * GlobalCounters.global_ops * 1e-9 / run_time
    print(f"{step:3d} {run_time:7.2f} ms run, {python_time:7.2f} ms python, {cpu_time:7.2f} ms CL, {loss_cpu:7.2f} loss, {lr:.6f} LR, {mem:.2f} GB used, {gflops:9.2f} GFLOPS")
    step += 1


if __name__ == "__main__":
  # pretrain_bert()
  # Login to wheigth and biases (https://wandb.ai/authorize)
  # wandb.login(key='ae38d649d60523043a4ffdc34a31b2b521758ede')

  with Tensor.train():
    pretrain_bert()