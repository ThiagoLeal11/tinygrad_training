import json
import pathlib
from dataclasses import dataclass
from typing import Iterable

from tinygrad.helpers import getenv
from tinygrad.nn import optim
from extra.models.bert import Bert
from tinygrad.tensor import Tensor, dtypes

import tinygrad as tg

ACT2FN = {
  "gelu": tg.Tensor.gelu,
}





def initializer(shape: Iterable[int], *, initializer_range: float = 0.02, dtype=tg.dtypes.float32):
  return Tensor.normal(*shape, std=initializer_range, dtype=dtype)


class Dense:
  def __init__(self, input_dim: int, output_dim: int, *, activation=Tensor.gelu, bias: bool = True):
    self.weight = initializer((input_dim, output_dim))
    self.bias = initializer((output_dim,)) if bias else None
    # TODO: custom activation

  def __call__(self, x: Tensor):
    return x.linear(self.weight, self.bias).gelu()


def gather_indexes(sequence_tensor: Tensor, positions: Tensor):
  assert len(sequence_tensor.shape) == 3, f"Expected tensor to have rank 3, but got {len(sequence_tensor.shape)}"
  batch_size, seq_length, width = sequence_tensor.shape

  flat_offset = (
    Tensor.arange(0, batch_size, requires_grad=False, dtype=dtypes.int32) * seq_length
  ).reshape([-1, 1])
  flat_positions = (positions + flat_offset).reshape([-1])
  flat_sequence_tensor = sequence_tensor.reshape([batch_size * seq_length, width])
  output_tensor = flat_sequence_tensor[flat_positions]
  return output_tensor


class BertForPreTraining(Bert):
  def __init__(self, config: dict):
    super().__init__(
      hidden_size=config["hidden_size"],
      intermediate_size=config["intermediate_size"],
      max_position_embeddings=config["max_position_embeddings"],
      num_attention_heads=config["num_attention_heads"],
      num_hidden_layers=config["num_hidden_layers"],
      type_vocab_size=config["type_vocab_size"],
      vocab_size=config["vocab_size"],
      attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
      hidden_dropout_prob=config["hidden_dropout_prob"]
    )
    self.pooled_weights = initializer((config["hidden_size"], config["hidden_size"]))
    self.masked_weights = initializer((config["vocab_size"], config["hidden_size"]))
    self.masked_bias = Tensor.zeros(config["vocab_size"])
    self.next_sentence_weights = initializer((2, config["hidden_size"]))
    self.next_sentence_bias = Tensor.zeros(2)

    self.masked_lm_dense = Dense(config["vocab_size"], config["hidden_size"])

  def __call__(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor):
    # Persist output for convenience
    output = super().__call__(input_ids, attention_mask, token_type_ids)
    self.sequence_output = output
    return output

  def embedding_table(self):
    return self.embeddings.word_embeddings.weight

  def pooled_output(self):
    out = self.sequence_output[:, 0]
    return out.linear(self.pooled_weights).tanh()

  def masked_lm_output(self, features: dict[str, Tensor]):
    in_tensor = gather_indexes(self.sequence_output, features["masked_lm_positions"])
    input_tensor = self.masked_lm_dense(in_tensor).layernorm()

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    print(output_bias.name)


# # Reference
# # https://github.com/mlcommons/training/blob/master/language_model/tensorflow/bert/run_pretraining.py#L276
def init_model(config: dict):
  model = Bert(
    hidden_size=config["hidden_size"],
    intermediate_size=config["intermediate_size"],
    max_position_embeddings=config["max_position_embeddings"],
    num_attention_heads=config["num_attention_heads"],
    num_hidden_layers=config["num_hidden_layers"],
    type_vocab_size=config["type_vocab_size"],
    vocab_size=config["vocab_size"],
    attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
    hidden_dropout_prob=config["hidden_dropout_prob"]
  )
  return model


def pooled_output(output: Tensor, weights: Tensor):
  pooled_output = output[:, 0]
  return Tensor.tanh(pooled_output.linear(weights))


@tg.jit.TinyJit
def train_step(model, opt, lr, features: dict[str, Tensor]):
  output = model(
    input_ids=features["input_ids"], attention_mask=features["input_mask"], token_type_ids=features["segment_ids"]
  )

  masked_lm_loss = get_masked_lm_output(model, output, features)









import json, math, time
from pathlib import Path
import numpy as np
from tinygrad.helpers import getenv
from tinygrad.jit import TinyJit
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters
from tinygrad.device import Device
from tinygrad.tensor import Tensor, dtypes
from tinygrad import GlobalCounters
from extra.lr_scheduler import OneCycleLR
from extra.models.bert import Bert
from extra.datasets.wikipedia import load_dataset
from extra import dist
import wandb


if __name__ == "__main__":
  if getenv("DIST"):
    dist.preinit()
    from extra.dist import collectives

if getenv('HALF', 0):
  np_dtype = np.float16
else:
  np_dtype = np.float32

BS, EVAL_BS, STEPS, MAX_EVAL_STEPS, WARMUP_STEPS, EPOCH, MAX_LR  = getenv("BS", 24), getenv('EVAL_BS', 8), getenv("STEPS", 100000), getenv("MAX_EVAL_STEPS", 100), getenv("WARMUP_STEPS", 10000), getenv("EPOCHS", 30), getenv('MAX_LR', 2.0)
EVAL_FREQ = math.floor(min(0.05*(230.23 * BS + 3000000), 25000))

def get_model_and_config(path:str):
  with open(path, 'r') as f:
    config = json.load(f)
  model = Bert(
    config["hidden_size"],
    config["intermediate_size"],
    config["max_position_embeddings"],
    config["num_attention_heads"],
    config["num_hidden_layers"],
    config["type_vocab_size"],
    config["vocab_size"],
    config["attention_probs_dropout_prob"],
    config["hidden_dropout_prob"]
  )
  embedding_table = model.embeddings.word_embeddings.weight
  s_weights = Tensor.uniform(*(2, config["hidden_size"]), low=-0.1, high=0.1) #TODO: change init range modeling.create_initializer(bert_config.initializer_range))
  s_bias = Tensor.zeros(2)
  m_weights = Tensor.uniform(*(config["hidden_size"], config["hidden_size"]), low=-0.1, high=0.1) #TODO: change init range
  m_bias = Tensor.zeros((config["vocab_size"],))
  p_weights = Tensor.uniform(*(config["hidden_size"], config["hidden_size"]), low=-0.1, high=0.1) #TODO: change init range
  return model, embedding_table, s_weights, s_bias, m_weights, m_bias, p_weights

def one_hot(arr:Tensor, num_classes=3):
  res = Tensor.eye(num_classes)[arr.reshape(-1)]
  return res.reshape(list(arr.shape) + [num_classes])

def pool_output(output:Tensor, weights:Tensor):
  pooled_output = output[:, 0]
  return Tensor.tanh(pooled_output.linear(weights))

def gather_indexes(sequence_tensor:Tensor, positions:Tensor):
  assert len(sequence_tensor.shape) == 3, f"Expected tensor to have rank 3, but got {len(sequence_tensor.shape)}"
  sequence_shape = list(sequence_tensor.shape)
  batch_size, seq_length, width = sequence_shape[0], sequence_shape[1], sequence_shape[2]

  flat_offsets = Tensor.arange(0, batch_size, requires_grad=False).reshape([1, -1]) * seq_length
  flat_positions = (positions + flat_offsets.reshape(-1, 1)).reshape([-1])
  flat_sequence_tensor = sequence_tensor.reshape([batch_size * seq_length, width])
  return flat_sequence_tensor[flat_positions]

def get_masked_lm_output(input_tensor:Tensor, output_weights:Tensor, transform_weights:Tensor, transform_bias:Tensor, positions:Tensor, label_ids:Tensor):
  input_tensor = gather_indexes(input_tensor, positions)
  input_tensor = Tensor.gelu(input_tensor.matmul(transform_weights))
  input_tensor = Tensor.layernorm(input_tensor)
  output = input_tensor.matmul(output_weights.transpose()).add(transform_bias)
  return output.sparse_categorical_crossentropy(label_ids.flatten())

def get_masked_lm_accuracy(input_tensor:Tensor, output_weights:Tensor, transform_weights:Tensor, transform_bias:Tensor, positions:Tensor, label_ids:Tensor):
  input_tensor = gather_indexes(input_tensor, positions)
  input_tensor = Tensor.gelu(input_tensor.matmul(transform_weights))
  input_tensor = Tensor.layernorm(input_tensor)
  logits = input_tensor.matmul(output_weights.transpose()).add(transform_bias)
  log_probs = logits.log_softmax()
  predictions = log_probs.argmax(axis=-1)
  correct_predictions = predictions == label_ids.flatten()
  return correct_predictions.float().mean()

def get_next_sentence_output(input_tensor:Tensor, labels: Tensor, weights:Tensor, bias:Tensor):
  output = input_tensor.matmul(weights.transpose()).add(bias)
  one_hot_labels = Tensor.eye(2)[labels]

  # _log_probs = output.log_softmax()
  # _per_example_loss = -(one_hot_labels * _log_probs).sum(axis=-1)
  # loss = _per_example_loss.mean(axis=-1)
  return output.log_softmax().binary_crossentropy_logits(one_hot_labels)

def pretrain():
  model, embedding_table, s_weights, s_bias, m_weights, m_bias, p_weights = get_model_and_config(str(Path(__file__).parent.parents[2] / "extra" / "datasets" / "wiki" / "bert_config.json"))
  optimizer = optim.LAMB(get_parameters(model), 1 / WARMUP_STEPS, eps=1e-6, wd=0.01, adam=True) # TODO: Keep in FP32?, Exclude LayerNorm, and bias from weight decay
  lr_scheduler = OneCycleLR(optimizer, MAX_LR, MAX_LR * WARMUP_STEPS, MAX_LR * 1e12, STEPS, WARMUP_STEPS / STEPS)

  config = dict(
    lr=MAX_LR,
    batch_size=BS,
    eval_batch_size=EVAL_BS,
    steps=STEPS,
    max_eval_steps=MAX_EVAL_STEPS,
    warmup_steps=WARMUP_STEPS,
    epochs=EPOCH,
    max_lr=MAX_LR,
    eval_freq=EVAL_FREQ
  )


  from extra.dist import OOB
  assert OOB is not None or not getenv("DIST"), "OOB should be initialized"
  rank, world_size = getenv("RANK", 0), getenv("WORLD_SIZE", 1)

  @TinyJit
  def eval_step_jitted(model, embedding_table, input_ids, input_mask, segment_ids, masked_lm_ids, masked_lm_positions):
    Tensor.training = False
    output = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
    acc = get_masked_lm_accuracy(output, embedding_table, m_weights, m_bias, masked_lm_positions, masked_lm_ids)
    Tensor.training = True
    return acc.realize()

  @TinyJit
  def train_step_jitted(model, embedding_table, optimizer, lr_scheduler, input_ids, input_mask, segment_ids, masked_lm_ids, masked_lm_positions, next_sentence_labels):
    # for div in range(4):
    output = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
    pooled_output = pool_output(output, p_weights)

    masked_lm_loss = get_masked_lm_output(output, embedding_table, m_weights, m_bias, masked_lm_positions, masked_lm_ids)
    next_sentence_loss = get_next_sentence_output(pooled_output, next_sentence_labels, s_weights, s_bias)
    loss = masked_lm_loss + next_sentence_loss

    if not getenv('DISABLE_BACKWARD', 0):
      optimizer.zero_grad()
      loss.backward()

      if getenv("DIST"):
        bucket, offset = [], 0
        for v in get_parameters(model):
          if v.grad is not None: bucket.append(v.grad.flatten())
        grads = collectives.allreduce(Tensor.cat(*bucket), cache_id="grads")
        for v in get_parameters(model):
          if v.grad is not None:
            v.grad.assign(grads[offset:offset+v.grad.numel()].reshape(*v.grad.shape))
            offset += v.grad.numel()

      optimizer.step()
      lr_scheduler.step()
    return loss.realize()

  def get_data(X, rank=0):
    device = f"{Device.DEFAULT}:{rank}"
    input_ids = Tensor(X["input_ids"])
    input_mask = Tensor(X["input_mask"])
    segment_ids = Tensor(X["segment_ids"])
    masked_lm_ids = Tensor(X["masked_lm_ids"], dtype=dtypes.int16)
    masked_lm_positions = Tensor(X["masked_lm_positions"], dtype=dtypes.int16)
    next_sentence_labels = Tensor(X["next_sentence_labels"], dtype=dtypes.int16)
    if getenv('DIST'):
      input_ids = input_ids.chunk(world_size, 0)[rank]
      input_mask = input_mask.chunk(world_size, 0)[rank]
      segment_ids = segment_ids.chunk(world_size, 0)[rank]
      masked_lm_ids = masked_lm_ids.chunk(world_size, 0)[rank]
      masked_lm_positions = masked_lm_positions.chunk(world_size, 0)[rank]
      next_sentence_labels = next_sentence_labels.chunk(world_size, 0)[rank]
    return input_ids.to(device).realize(), input_mask.to(device).realize(), segment_ids.to(device).realize(), masked_lm_ids.to(device).realize(), masked_lm_positions.to(device).realize(), next_sentence_labels.to(device).realize()

  train_batcher = load_dataset(bs=BS)
  eval_batcher = load_dataset(bs=EVAL_BS, is_validation=True)
  accuracy_achieved = False
  wallclock_start = time.monotonic()

  wandb.init(project="pretrain_bert_tinygrad", config=config)

  for _ in range(EPOCH):
    i = 0
    while i <= STEPS:
      if i % EVAL_FREQ == 0 and i != 0:
        e = 0
        while e <= MAX_EVAL_STEPS:
          st = time.monotonic()
          X, _ = next(eval_batcher)
          input_ids, input_mask, segment_ids, masked_lm_ids, masked_lm_positions, next_sentence_labels = get_data(X, rank)
          acc = eval_step_jitted(model, embedding_table, input_ids, input_mask, segment_ids, masked_lm_ids, masked_lm_positions)
          et = time.monotonic()
          acc = acc.numpy()
          cl = time.monotonic()
          if getenv('DIST'):
            if rank == 0:
              accs = []
              for j in range(1, min(world_size, 8)):
                accs.append(OOB.recv(j))
            elif rank < min(world_size, 8):
              OOB.send(acc, 0)

          if rank == 0:
            acc = (sum(acc) / len(acc))*100.0 if getenv('DIST') else acc
            print(f"MLM accuarcy: {acc:.2f}%, val_loss STEP={i} (in {(time.monotonic()-st)*1e3:.2f} ms)")
            if acc > 72.0:
              wallclock_end = time.monotonic()
              hours, remainder = divmod(wallclock_end - wallclock_start, 3600)
              minutes, seconds = divmod(remainder, 60)
              print(f"MLM accuracy achieved in {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
              accuracy_achieved = True
              break
          e += 1
          st = cl
      if accuracy_achieved or STEPS == 0 or i==STEPS: break

      if accuracy_achieved: break

      st = time.monotonic()
      X, _ = next(train_batcher)
      input_ids, input_mask, segment_ids, masked_lm_ids, masked_lm_positions, next_sentence_labels = get_data(X, rank)
      GlobalCounters.reset()
      loss = train_step_jitted(model, embedding_table, optimizer, lr_scheduler, input_ids, input_mask, segment_ids, masked_lm_ids, masked_lm_positions, next_sentence_labels)

      et = time.monotonic()
      loss_cpu = loss.numpy()
      cl = time.monotonic()

      if not getenv("DIST"):
        print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {optimizer.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
      else:
        print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {optimizer.lr.numpy()[0]:.6f} LR, {world_size*GlobalCounters.mem_used/1e9:.2f} GB used, {world_size*GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
      wandb.log({"run_time": (cl-st)*1000.0, "python_time": (et-st)*1000.0, "cl_time": (cl-et)*1000.0, "loss": loss_cpu, "lr": optimizer.lr.numpy()[0], "mem_used": GlobalCounters.mem_used/1e9, "gflops": GlobalCounters.global_ops*1e-9/(cl-st)}, step=i)
      i += 1

def train():
  if not getenv("DIST"):
    pretrain()
  else:
    if getenv("HIP"):
      devices = [f"hip:{i}" for i in range(6)] # find way to query device count
    else:
      devices = [f"gpu:{i}" for i in range(6)] # find way to query device count
    world_size = len(devices)

    assert BS % world_size == 0, f"batch size {BS} is not divisible by world size {world_size}"
    assert EVAL_BS % min(world_size, 5) == 0, f"evaluation batch size {EVAL_BS} is not divisible by world size {min(world_size, 5)}"
    assert EVAL_BS < 10000, "EVAL_BS exceeds eval sample (10000) count"

    dist.init_oob(world_size)

    processes = []
    for rank, device in enumerate(devices):
       processes.append(dist.spawn(rank, device, fn=pretrain, args=()))
    for p in processes: p.join()

if __name__ == "__main__":
  # pretrain_bert()
  # Login to wheigth and biases (https://wandb.ai/authorize)
  wandb.login(key='ae38d649d60523043a4ffdc34a31b2b521758ede')

  with Tensor.train(): train()
