import json
import pathlib
from dataclasses import dataclass

from tinygrad import nn
from tinygrad import Tensor

from extra.models.bert import Bert
import itertools

ACT2FN = {
  "gelu": Tensor.gelu,
}


@dataclass
class BertConfig:
  vocab_size: int = 30522
  hidden_size: int = 768
  num_hidden_layers: int = 12
  num_attention_heads: int = 12
  intermediate_size: int = 3072
  hidden_act: str = "gelu"
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  max_position_embeddings: int = 512
  type_vocab_size: int = 2
  initializer_range: float = 0.02

  @classmethod
  def from_json(cls, path: pathlib.Path):
    return cls(**json.loads(path.read_text()))


class BertLayerNorm:
  def __init__(self, config: BertConfig, variance_epsilon: float = 1e-12):
    self.gamma = Tensor.ones(config.hidden_size)
    self.beta = Tensor.zeros(config.hidden_size)
    self.variance_epsilon = variance_epsilon

  def __call__(self, x: Tensor):
    u = x.mean(-1, keepdim=True)
    s = (x - u).pow(2).mean(-1, keepdim=True)
    x = (x - u) / (s + self.variance_epsilon).sqrt()
    return self.gamma * x + self.beta


class BertPoller:
  def __init__(self, config: BertConfig):
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.activation = Tensor.tanh

  def __call__(self, hidden_states: Tensor):
    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token.
    first_token_tensor = hidden_states[0]
    pooled_output = self.dense(first_token_tensor)
    pooled_output = self.activation(pooled_output)  # TODO: fix typing error
    return pooled_output


class BertPredictionHeadTransform:
  def __init__(self, config: BertConfig):
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.act = ACT2FN[config.hidden_act]
    self.layer_norm = BertLayerNorm(config)

  def __call__(self, hidden_state: Tensor):
    return self.layer_norm(self.act(self.dense(hidden_state)))


class BertLMPredictionHead:
  def __init__(self, config: BertConfig, bert_model_embedding_weights: Tensor):
    self.transform = BertPredictionHeadTransform(config)
    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    self.decoder = nn.Linear(bert_model_embedding_weights.shape[0], bert_model_embedding_weights.shape[1], bias=True)
    self.decoder.weight = bert_model_embedding_weights
    self.decoder.bias = Tensor.zeros(bert_model_embedding_weights.shape[0])

  def __call__(self, hidden_state: Tensor):
    return self.decoder(self.transform(hidden_state))


class BertPreTrainingHeads:
  def __init__(self, config: BertConfig, bert_model_embedding_weights: Tensor):
    self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
    self.seq_relationship = nn.Linear(config.hidden_size, 2)

  def __call__(self, sequence_output: Tensor, pooled_output: Tensor):
    prediction_scores = self.predictions(sequence_output)
    seq_relationship_score = self.seq_relationship(pooled_output)
    return prediction_scores, seq_relationship_score


class PreTrainedBertModel:
  """ An abstract class to handle weights initialization and
      a simple interface for downloading and loading pretrained models.
  """
  def __init__(self, config: BertConfig):
    self.config = config

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight = Tensor.normal(*module.weight.shape, mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, BertLayerNorm):
      module.gamma = Tensor.normal(*module.gamma.shape, mean=0.0, std=self.config.initializer_range)
      module.beta = Tensor.normal(*module.beta.shape, mean=0.0, std=self.config.initializer_range)

    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias = Tensor.zeros(*module.bias.shape)

  @classmethod
  def from_pretrained(cls, pretrained_model_name, cache_dir=None, *inputs, **kwargs):
    # TODO: implement this
    pass


def unroll_lists(lst: list[list]) -> list:
  result = []
  for r in lst:
    if isinstance(r, (list, tuple)):
      result += list(unroll_lists(r))
    else:
      result.append(r)
  return result


def list_children_recursively(cls) -> list:
  primitives = (int, float, str, bool, Tensor)
  containers = (list, tuple)

  raw_objects = [getattr(cls, x) for x in dir(cls) if not x.startswith("__")]
  objects = unroll_lists([x if not isinstance(x, containers) else list(x) for x in raw_objects])
  should_recurse = [x for x in objects if not isinstance(x, primitives)]

  return unroll_lists(
    should_recurse + [list_children_recursively(s) for s in should_recurse]
  )


class BertModel(Bert):
  def __init__(self, config: BertConfig):
    super().__init__(
      hidden_size=config.hidden_size,
      intermediate_size=config.intermediate_size,
      max_position_embeddings=config.max_position_embeddings,
      num_attention_heads=config.num_attention_heads,
      num_hidden_layers=config.num_hidden_layers,
      type_vocab_size=config.type_vocab_size,
      vocab_size=config.vocab_size,
      attention_probs_dropout_prob=config.attention_probs_dropout_prob,
      hidden_dropout_prob=config.hidden_dropout_prob,
    )
    self.poller = BertPoller(config)

  def __call__(self, input_ids, attention_mask, token_type_ids):
    encoder_outputs = super().__call__(input_ids, attention_mask, token_type_ids)
    sequence_output = encoder_outputs[-1]
    pooled_output = self.poller(sequence_output)
    return encoder_outputs, pooled_output


class BertForPreTraining(PreTrainedBertModel):
  def __init__(self, config: BertConfig):
    super().__init__(config)

    self.bert = BertModel(config)
    self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
    self.config = config
    for children in list_children_recursively(self.bert):
      self._init_weights(children)

  def __call__(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor, masked_lm_labels: Tensor, next_sentence_label: Tensor):
    sequence_output, pooled_output = self.bert(input_ids, attention_mask, token_type_ids)

    prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

    if masked_lm_labels and next_sentence_label:
      masked_lm_loss = (
        prediction_scores
        .reshape(-1, self.config.vocab_size)
        .sparse_categorical_crossentropy(masked_lm_labels.reshape(-1), ignore_index=-1)
      )
      next_sentence_loss = (
        seq_relationship_score
        .reshape(-1, 2)
        .sparse_categorical_crossentropy(next_sentence_label.reshape(-1), ignore_index=-1)
      )
      return masked_lm_loss + next_sentence_loss
    return prediction_scores, seq_relationship_score


if __name__ == '__main__':
  cfg = BertConfig.from_json(pathlib.Path(__file__).parent / "bert_config.json")
  model = BertForPreTraining(cfg)
  print(model)