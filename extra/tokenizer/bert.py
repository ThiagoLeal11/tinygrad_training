import unicodedata

def _is_whitespace(char):
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  return unicodedata.category(char) == "Zs"

def _is_control(char):
  if char == "\t" or char == "\n" or char == "\r":
    return False
  return unicodedata.category(char).startswith("C")

def _is_punctuation(char):
  # range(33, 48) -> ! " # $ % & ' ( ) * + , - . /
  # range(58, 65) -> : ; < = > ? @
  # range(91, 97) -> [ \ ] ^ _
  # range(123, 127) -> { | } ~
  cp = ord(char)
  if (33 <= cp < 48) or (58 <= cp < 65) or (91 <= cp < 97) or (123 <= cp < 127):
    return True
  return unicodedata.category(char).startswith("P")

def _run_split_on_punc(text):
  if text in ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"):
    return [text]
  start_new_word = True
  output = []
  for i in range(len(text)):
    if _is_punctuation(char := text[i]):
      output.append([char])
      start_new_word = True
    else:
      if start_new_word:
        output.append([])
      start_new_word = False
      output[-1].append(char)
  return ["".join(x) for x in output]

def _run_strip_accents(text):
  output = []
  for char in unicodedata.normalize("NFD", text):
    if unicodedata.category(char) != "Mn":
      output.append(char)
  return "".join(output)

def _clean_text(text):
  output = []
  for char in text:
    if not ((cp := ord(char)) == 0 or cp == 0xfffd or _is_control(char)):
      output.append(" " if _is_whitespace(char) else char)
  return "".join(output)


def _wordpiece_tokenize(text, vocab):
  text = text.decode("utf-8", "ignore") if isinstance(text, bytes) else text
  output_tokens = []
  for token in text.strip().split():
    chars = list(token)
    if len(chars) > 200:
      output_tokens.append("[UNK]")
      continue

    is_bad = False
    start = 0
    sub_tokens = []
    while start < len(chars):
      end = len(chars)
      cur_substr = None
      while start < end:
        substr = "".join(chars[start:end])
        if start > 0: substr = "##" + substr
        if substr in vocab:
          cur_substr = substr
          break
        end -= 1
      if cur_substr is None:
        is_bad = True
        break
      sub_tokens.append(cur_substr)
      start = end

    if is_bad: output_tokens.append("[UNK]")
    else: output_tokens.extend(sub_tokens)
  return output_tokens


class BertTokenizer:
  def __init__(self, vocab_file, is_lower_case: bool = True):
    self.vocab = {}
    with open(vocab_file) as f:
      for line in f:
        line = line.decode("utf-8", "ignore") if isinstance(line, bytes) else line
        if (token := line.strip()) and token not in self.vocab: 
          self.vocab[token] = len(self.vocab)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.is_lower_case = is_lower_case

  def tokenize(self, text):
    text = _clean_text(text.decode("utf-8", "ignore") if isinstance(text, bytes) else text)
    if self.is_lower_case: 
      text = text.lower()
    # BasicTokenizer
    split_tokens = []
    for token in text.strip().split():
      split_tokens.extend(_run_split_on_punc(_run_strip_accents(token.lower())))
    split_tokens = " ".join(split_tokens).strip().split()
    # WordPieceTokenizer
    tokens = []
    for token in split_tokens:
      tokens.extend(_wordpiece_tokenize(token, self.vocab))
    return tokens

  def convert_tokens_to_ids(self, tokens): 
    return [self.vocab[token] for token in tokens]
  
  def convert_ids_to_tokens(self, ids): 
    return [self.inv_vocab[id] for id in ids]
  

if __name__ == "__main__":
  from pathlib import Path
  tokenizer = BertTokenizer(Path(__file__).parent.parent / "datasets/wiki/vocab.txt")
  lorem_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum!"
  for _ in range(10_000):
    tokenizer.tokenize(lorem_text)