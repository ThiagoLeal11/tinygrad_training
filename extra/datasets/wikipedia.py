import itertools
import queue
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Generator, Optional

import numpy as np
import tqdm

from extra.tokenizer.bert import BertTokenizer
from tinygrad.helpers import getenv

DOCUMENT = list[list[str]]
BASEDIR = Path(__file__).parent / "wiki"


@dataclass
class Instance:
  tokens: list[str]
  segment_ids: list[int]
  masked_positions: list[int]
  masked_labels: list[str]
  is_random_next: bool

  def __post_init__(self):
    assert len(self.tokens) == len(self.segment_ids)
    assert len(self.masked_positions) == len(self.masked_labels)


def join_lists(lists: Iterable[list[str]]) -> list[str]:
  return list(itertools.chain.from_iterable(lists))


def concatenate_dict(dicts: list[dict]) -> dict:
  return {
    k: np.concatenate([d[k] for d in dicts], axis=0)
    for k in dicts[0].keys()
  }


def read_document(filepath: str, load_all: bool = False) -> Generator[DOCUMENT, None, None]:
  with open(filepath, "r", encoding="utf-8") as f:
    document = []

    for line in f:
      line = line.strip()
      if line:
        document.append(line)
      elif document:
        yield document
        document = []


def read_all_document(filepath: str) -> list[DOCUMENT]:
  with open(filepath, "r", encoding="utf-8") as f:
    split_docs = [[]]
    for line in f.readlines():
      line = line.strip()
      if not line:
        split_docs.append([])
      else:
        split_docs[-1].append(line)

    return [s for s in split_docs if s]


def tokenize_doc(doc: DOCUMENT, tokenizer: BertTokenizer) -> DOCUMENT:
  return [tokenizer.tokenize(line) for line in doc]


def select_from_document(doc: DOCUMENT, target_len: int, rand_gen, rand_start: bool, rand_end: bool) -> tuple[int, int]:
  start = 0 if not rand_start else rand_gen.randint(0, len(doc) - 1)

  max_lines = 1
  for max_lines, _ in enumerate(doc[start:], start=1):
    size = sum(len(line) for line in doc[start:start + max_lines])
    if size >= target_len:
      break

  size = max_lines
  if rand_end and max_lines > 1:
    size = rand_gen.randint(1, max_lines - 1)
  return start, size


def truncate_seq_pair(a: list[str], b: list[str], max_num_tokens: int) -> tuple[list[str], list[str]]:
  total_length = len(a) + len(b)
  if total_length <= max_num_tokens:
    return a, b

  # Try to keep equal number of tokens on each side
  diff = total_length - max_num_tokens
  mid = max_num_tokens // 2
  a_end = len(a) - max(len(a) - mid, 0)
  b_end = len(b) - max(len(b) - mid, 0)

  return a[:a_end], b[:b_end]


def mask_tokens(tokens: list[str], vocab_words: list[str], rand_gen) -> tuple[list[str], list[int], list[str]]:
  num_to_predict = min(
    getenv('MAX_PREDICTIONS_PER_SEQ', 20),
    max(1, int(round(len(tokens) * 0.15)))
  )

  output = list(tokens)
  valid_indexes = [i for i, t in enumerate(tokens) if t not in ("[CLS]", "[SEP]")]
  rand_gen.shuffle(valid_indexes)

  masked = []
  for idx in valid_indexes:
    has_masked_enough = len(masked) >= num_to_predict
    if has_masked_enough:
      break

    if rand_gen.random() < 0.8:
      output[idx] = "[MASK]"
    elif rand_gen.random() < 0.5:
      output[idx] = rand_gen.choice(vocab_words)

    masked.append((idx, tokens[idx]))

  masked = sorted(masked, key=lambda x: x[0])
  masked_pos = [x[0] for x in masked]
  masked_labels = [x[1] for x in masked]

  return output, masked_pos, masked_labels


def create_instance_from_document(rand_gen, doc: DOCUMENT, rand_doc: DOCUMENT, vocab_words: list[str]) -> Optional[
  Instance]:
  max_num_tokens = getenv('MAX_SEQ_LENGTH', 128) - 3  # [CLS] + 2 * [SEP]
  target_seq_length = max_num_tokens

  if rand_gen.random() < 0.1:  # 10% of the time, pick a random sized document
    target_seq_length = rand_gen.randint(2, max_num_tokens)

  _, a_size = select_from_document(doc, target_seq_length, rand_gen, False, True)
  a_tokens = join_lists(doc[:a_size])

  # Next sentence from random document
  is_random_next = rand_doc != doc and (len(doc) == 1 or rand_gen.random() < 0.5)
  if is_random_next:
    target_b_len = target_seq_length - len(a_tokens)

    # pick random document
    b_start, b_size = select_from_document(rand_doc, target_b_len, rand_gen, True, False)
    b_tokens = join_lists(rand_doc[b_start:b_start + b_size])
  else:
    # Next sentence from same document
    b_tokens = join_lists(doc[a_size:])

  a, b = truncate_seq_pair(list(a_tokens), list(b_tokens), target_seq_length)
  tokens = ["[CLS]"] + a + ["[SEP]"] + b + ["[SEP]"]
  segment_ids = [0] * (len(a) + 2) + [1] * (len(b) + 1)

  masked_tokens, masked_pos, masked_labels = mask_tokens(tokens, vocab_words, rand_gen)

  return Instance(
    tokens=masked_tokens,
    segment_ids=segment_ids,
    masked_positions=masked_pos,
    masked_labels=masked_labels,
    is_random_next=is_random_next,
  )


def get_next_documents(filepaths: list[str], read_all: bool = False) -> Generator[list[DOCUMENT], None, None]:
  if read_all:
    opened_files = [read_all_document(f) for f in filepaths]
    yield from opened_files
    return

  opened_files = [read_document(f) for f in filepaths]
  has_docs = True
  while has_docs:
    current_docs = []
    for f in opened_files:
      try:
        current_docs.append(next(f))
      except StopIteration:
        pass

    has_docs = any(current_docs)
    if has_docs:
      yield current_docs


def instance_to_features(instance: Instance, tokenizer: BertTokenizer) -> dict:
  max_seq_len = getenv('MAX_SEQ_LENGTH', 128)
  mas_pred_len = getenv('MAX_PREDICTIONS_PER_SEQ', 20)
  assert len(instance.tokens) <= max_seq_len, f"instance length {len(instance.tokens)} exceeds max length {max_seq_len}"
  assert len(
    instance.masked_labels) <= mas_pred_len, f"instance masked length {len(instance.masked_labels)} exceeds max length {mas_pred_len}"

  # Padding
  pad_size = max_seq_len - len(instance.tokens)
  mask_pad_size = mas_pred_len - len(instance.masked_labels)

  # Expanded masked labels
  masked_labels = [-100] * max_seq_len
  masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_labels)
  for idx, label in zip(instance.masked_positions, instance.masked_labels):
    masked_labels[idx] = tokenizer.convert_tokens_to_ids([label])[0]

  def padded_array(arr: list, pad_size: int = 0, dtype=np.int16) -> np.ndarray:
    return np.expand_dims(
      np.array(arr + [0] * pad_size, dtype=dtype),
      axis=0
    )

  return {
    "input_ids": padded_array(tokenizer.convert_tokens_to_ids(instance.tokens), pad_size),
    "input_mask": padded_array([1] * len(instance.tokens), pad_size),
    "segment_ids": padded_array(instance.segment_ids, pad_size),
    "masked_lm_positions": padded_array(instance.masked_positions, mask_pad_size),
    "masked_lm_ids": padded_array(masked_lm_ids, mask_pad_size),
    "masked_lm_labels": padded_array(masked_labels, 0),
    "next_sentence_labels": np.array([int(instance.is_random_next)]),
  }


# def parallel_read_tokenized(tokenizer, file_list: list[str]):


def generate_features(tokenizer, file_list: list[str], read_all: bool = False, skip_n: int = 0) -> Generator[
  tuple[dict, Instance], None, None]:
  # TODO accumulate in a queue
  """Convert raw text to masked NSP samples"""
  rand_gen = random.Random(getenv('SEED', 12345))
  duplication_factor = getenv('DUP_FACTOR', 10)
  vocab_words = list(tokenizer.vocab.keys())

  pbar = tqdm.tqdm(total=skip_n, disable=skip_n==0, desc='Skip dataset')

  for docs in get_next_documents(file_list, read_all):
    num_docs = len(docs) * duplication_factor
    if skip_n > num_docs:
      skip_n -= num_docs
      pbar.update(num_docs)
      continue
    if skip_n:
      pbar.update(skip_n)
    pbar.close()

    docs = [[tokenizer.tokenize(line) for line in doc] for doc in docs]
    for _ in range(duplication_factor):
      docs_indexes = list(range(len(docs)))
      rand_gen.shuffle(docs_indexes)

      for idx in docs_indexes:
        rand_idx = rand_gen.randint(0, len(docs) - 1)
        rand_doc = docs[rand_idx] if rand_idx != idx else docs[(rand_idx + 1) % len(docs)]
        instance = create_instance_from_document(rand_gen, docs[idx], rand_doc, vocab_words)
        if not instance:
          continue
        features = instance_to_features(instance, tokenizer)
        yield features, instance


def buffer_to_queue(gen: Generator, buffer: queue.Queue):
  while True:
    try:
      buffer.put(next(gen), block=True)
    except StopIteration:
      break


def load_dataset(bs: int, is_validation: bool = False, skip_n: int = 0) -> Generator[tuple[dict, Instance], None, None]:
  files = [BASEDIR / f"results4/part-{part:05d}-of-00500" for part in range(500)]

  if is_validation:
    files = [BASEDIR / "results4/eval.txt"]

  buffer = queue.Queue(maxsize=1024)
  tokenizer = BertTokenizer(Path(__file__).parent / "wiki" / "vocab.txt", is_lower_case=True)
  gen = generate_features(tokenizer, [str(f) for f in files], skip_n=skip_n)
  buffer_thread = threading.Thread(target=buffer_to_queue, args=(gen, buffer))
  buffer_thread.start()

  while True:
    results = []
    for _ in range(bs):
      # TODO: Should discard the last batch if it's not full
      results.append(buffer.get(block=True))
    features, instances = zip(*results)

    yield concatenate_dict(features), instances

  buffer_thread.join()


if __name__ == "__main__":
  import time

  count = 0

  # TODO: colocar script pra baixar tudo e descompactar, se não existir

  st = time.time()
  for X, Y in load_dataset(32, False, skip_n=100_000):
    # print('batch', count, 'time', time.time() - st)
    count += 1
    st = time.time()

    if count % 1_000 == 0:
      print('batch', count, 'time', time.time() - st)

    if count >= 107538:
      break
