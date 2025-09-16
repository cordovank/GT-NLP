
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from collections import Counter

import time
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import re
import os
import json
import random
import numpy as np
import pickle
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

import unidecode


MY_DRIVE = "LOCAL" # Set this to "GOOGLE" if saving to Google Drive
path = 'drive/MyDrive/' if MY_DRIVE == "GOOGLE" else ''
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def save_data(path, data, name="data"):
    with open(f"{path}{name}", "wb") as f:
        pickle.dump(data, f, protocol=None, fix_imports=True, buffer_callback=None)

def load_data(path, name="data"):
    with open(f"{path}{name}", "rb") as f:
        data = pickle.load(f)
    return data



UNK = 'unk'
UNK_ID = '0'

def tokenize(line):
    line = re.sub(r'[^a-zA-Z0-9]', ' ', unidecode.unidecode(line)) # remove punctuation
    line = line.lower().split()  # lower case
    return line

class Vocab:
    def __init__(self, name = 'vocab'):
        self.name = name
        self._word2index = {}
        self._word2count = {}
        self._index2word = {}
        self._n_words = 0

    def get_words(self):
      return list(self._word2count.keys())

    def num_words(self):
      return self._n_words

    def word2index(self, word):
      return self._word2index[word]

    def index2word(self, word):
      return self._index2word[word]

    def word2count(self, word):
      return self._word2count[word]

    def add_sentence(self, sentence):
        for word in tokenize(sentence):
            self.add_word(word)

    def add_word(self, word):
        if word not in self._word2index:
            self._word2index[word] = self._n_words
            self._word2count[word] = 1
            self._index2word[self._n_words] = word
            self._n_words += 1
        else:
            self._word2count[word] += 1

# ——————————————————————————————————————————————————————————
# MY CODE BELOW
# ——————————————————————————————————————————————————————————

# Part D: Training on the Full Data

# Question generator
# 1. Extract top relations: find most common relations
# 2. Define templates for each relation: containing the person's name and words representative of the relation, even if they don’t exactly match.
# 3. Create function to generate natural language questions: given a person and a relation, returns several questions.

def filter_db_by_person(DB, max_train=500, max_test=100):
    all_people = list(DB.keys())
    train_people = all_people[:max_train]
    test_people = all_people[max_train:max_train+max_test]

    train_db = {p: DB[p] for p in train_people}
    test_db = {p: DB[p] for p in test_people}

    return train_db, test_db


def get_top_relations(DB, top_n=20):
    ignore_keys={'name', 'article_title'}
    relation_counter = Counter()

    for entity in DB.values():
        for key in entity:
            if key not in ignore_keys:
                relation_counter[key] += 1

    return [relation for relation, _ in relation_counter.most_common(top_n)]

def load_template(filepath="question_template.json"):
    with open(filepath, 'r') as f:
        templates = json.load(f)
    return templates

# Load templates
# q_template = load_template()
# Check
# print(q_template.get(top_relations[0], []))
# Output: ['What office did [NAME] hold?', 'Which position did [NAME] serve in?', "What was [NAME]'s role in office?", 'What official position did [NAME] occupy?']

def generate_questions(person_name, relation, q_template, num_questions=2):
    templates = q_template.get(relation, ["What is [NAME]'s [RELATION]?"])
    selected_templates = random.sample(templates, min(num_questions, len(templates)))
    questions = [template.replace("[NAME]", person_name).replace("[RELATION]", relation.replace("_", " ")) for template in selected_templates]
    return questions

# Sample a name and relation
# name = list(DB.keys())[0]
# print(generate_questions(name, top_relations[0], q_template))
# OUTPUT: ['What office did j. p. featherston hold?', "What was j. p. featherston's role in office?"]


### Expand Vocabulary
# We want to ensure that the vocabulary includes all tokens from the question templates.
# Since we are using templates that introduce new words or phrases not present in the original vocabulary created from the DB, we need to expand the vocabulary to include these new tokens.
# This is done by tokenizing the question templates and adding the tokens to the vocabulary. The `make_vocab` function is modified to include this step. UNK tokens are also added to the vocabulary for safety.

def make_vocab(DB, question_templates):
    # Make the vocab object
    vocab = Vocab()

    # Tokenize DB
    tokens = tokenize(str(DB))
    for t in tqdm(tokens, desc="Adding DB tokens to vocab"):
        vocab.add_word(t)

    # Tokenize and add question templates
    for templates in question_templates.values():
        for template in templates:
            tokens = tokenize(template)
            for t in tokens:
                vocab.add_word(t)

    vocab.add_word(UNK)

    return vocab

# MY KVMemNet model

class KVMemNet(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(KVMemNet, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Define linear projections
        self.A = nn.Linear(vocab_size, embed_dim, bias=False)  # For questions and keys
        self.B = nn.Linear(vocab_size, embed_dim, bias=False)  # For values

    def forward(self, question, keys, values):
        """
        question: (batch_size, vocab_size)
        keys:     (batch_size, num_keys, vocab_size)
        values:   (batch_size, num_values, vocab_size)
        """
        # Project inputs into embedding space
        q = self.A(question)                  # (batch_size, embed_dim)
        k = self.A(keys)                      # (batch_size, num_keys, embed_dim)
        v = self.A(values)                    # (batch_size, num_values, embed_dim)

        # Attention scores: dot product between q and k
        attention_scores = torch.bmm(k, q.unsqueeze(2)).squeeze(2)
        attention_probs = F.softmax(attention_scores, dim=1)            # (batch_size, num_keys)

        # Weighted sum over values
        output = torch.bmm(attention_probs.unsqueeze(1), v).squeeze(1)  # (batch_size, embed_dim)

        return output


# PREPARE DATASET WITH DATA LOADER

def safe_multihot(s, vocab, preserve_counts=False):
    # multihot() assumes that all tokens are in the vocab, but if vocab is reduced we get KeyError
    # Merging multihot() and unkit() into a safe multihot function: Replaces unknown tokens with UNK_ID, while avoiding double tokenization.

    """
    Make a bag-of-words vector from a string s using the provided vocabulary.
    Handles out-of-vocab words: If a word is missing in vocab, it maps the unknown token to UNK_ID
    The returned vector is of size (vocab_size,).
    """
    tokens = tokenize(s)
    token_ids = [vocab._word2index.get(token, vocab._word2index[UNK]) for token in tokens]

    # Initialize multihot vector
    mhot = np.zeros((len(token_ids), vocab.num_words()))
    mhot[np.arange(len(token_ids)), token_ids] = 1
    if preserve_counts:
        return mhot.sum(0)
    else:
        return mhot.sum(0) >= 1

def get_distractor_keys_values(DB, current_person, num_distractors):
    # Prepare distractor keys and values from other entities

    distractor_keys = []
    distractor_values = []

    # Filter out entities that have no facts
    other_entities = [e for e, facts in DB.items() if e != current_person and len(facts) > 0]

    for _ in range(num_distractors):
        if not other_entities:
            break

        entity = random.choice(other_entities)
        facts = DB[entity]

        if not facts:
            continue

        k = random.choice(list(facts.keys()))
        v = facts[k]

        distractor_keys.append(k)
        distractor_values.append(v)

    return distractor_keys, distractor_values

def encode_texts_to_multihot(texts, vocab):
    # Convert list of texts to multihot numpy arrays

    return np.array([safe_multihot(text, vocab) for text in texts])

def prep_dataset(DB, VOCAB, q_templates, top_relations, num_questions_per_relation=2, n_distractors=5):
    data = []

    # Filter entities that have at least one of the top relations
    filtered_entities = [person for person, facts in DB.items() if any(rel in facts for rel in top_relations)]

    for person in tqdm(filtered_entities, desc="Preparing dataset"):
        person_facts = DB[person]

        # Get person's keys and values
        person_keys = list(person_facts.keys())
        person_values = list(person_facts.values())

        # Add distractors (random keys and values from other entities)
        distractor_keys, distractor_values = get_distractor_keys_values(DB, person, n_distractors)

        # Prepare multihots
        keys = person_keys + distractor_keys
        values = person_values + distractor_values

        keys_encoded = encode_texts_to_multihot(keys, VOCAB)        # (n_keys, vocab_size)
        values_encoded = encode_texts_to_multihot(values, VOCAB)    # (n_values, vocab_size)

        for relation in person_facts:
            if relation not in top_relations:
                continue

            questions = generate_questions(person, relation, q_templates, num_questions=num_questions_per_relation)

            # Prepare multihot sample for each question
            for question in questions:
                question_encoded = safe_multihot(question, VOCAB)

                # find the index of the correct answer in values, otherwise skip
                try:
                    target_idx = values.index(person_facts[relation])
                except ValueError:
                    continue

                sample = {
                    "question": question_encoded,           # np.array (vocab_size,)
                    "keys": keys_encoded,                   # np.array (n_keys,   vocab_size)
                    "values": values_encoded,               # np.array (n_keys,   vocab_size)
                    "target_idx": target_idx,               # int ground truth

                }
                data.append(sample)

    return data


class QADataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        return {
            'question': torch.from_numpy(s['question']).float(),      # (vocab_size,)
            'keys':     torch.from_numpy(s['keys']).float(),          # (n_keys,   vocab_size)
            'values':   torch.from_numpy(s['values']).float(),        # (n_values, vocab_size)
            'target':   torch.tensor(s['target_idx'], dtype=torch.long)
        }


# pad variable‑length keys & values
def qa_collate_fn(batch):
    """
    batch: list of dicts, each with
      'question': (V,)        tensor
      'keys':     (K_i, V)    tensor
      'values':   (M_i, V)    tensor
      'target':   ()          tensor
    """
    questions = [b['question'] for b in batch]
    keys_list  = [b['keys']     for b in batch]
    vals_list  = [b['values']   for b in batch]
    targets    = torch.stack([b['target'] for b in batch])

    # stack questions: (B, V)
    q_batch = torch.stack(questions)

    # pad keys to max_K
    max_k = max(k.shape[0] for k in keys_list)
    # pad along dim=0 (the sequence dim), keep dim=1 (vocab dim) unchanged
    k_padded = torch.stack([
        F.pad(k, (0, 0, 0, max_k - k.shape[0])) for k in keys_list
    ])  # (B, max_k, V)

    # pad values to max_M
    max_m = max(v.shape[0] for v in vals_list)
    v_padded = torch.stack([
        F.pad(v, (0, 0, 0, max_m - v.shape[0])) for v in vals_list
    ])  # (B, max_m, V)

    return q_batch, k_padded, v_padded, targets

# USE WITH DATALOADER
import time
def trainModel(model, train_loader, test_loader=None, num_epochs=5, criterion=None, optimizer=None, save_path="model.pt"):
    start_time = time.time()

    train_losses = []
    test_losses  = []
    test_accs    = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for q, k, v, tgt in train_loader:
            q, k, v, tgt = q.to(device), k.to(device), v.to(device), tgt.to(device)

            o = model(q, k, v)                    # (B, E)
            v_emb = model.B(v)                    # (B, M, E)

            # scores: (B, M)
            scores = torch.bmm(o.unsqueeze(1), v_emb.transpose(1,2)).squeeze(1)
            loss = criterion(scores, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Monitor progress
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"{'='*50}")
        print(f"Epoch {epoch+1:>2}/{num_epochs} | Train Loss: {avg_train_loss:.4f}")

        # Evaluate on test data
        if test_loader is not None:
            test_loss, test_acc = evaluateModel(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            print(f"              Eval Loss: {test_loss:.4f},  Acc: {test_acc:.4%}")

        # Save checkpoint
        torch.save(model.state_dict(), save_path)

    end_time = time.time()

    # Final Test Accuracy
    if test_loader is not None:
        final_loss, final_acc = evaluateModel(model, test_loader, criterion, device)
        print(f"{'='*50}")
        print("\n\n Final Evaluation:")
        print(f"{'-'*50}")
        print(f"Final Test Loss:     {final_loss:.4f}")
        print(f"Final Test Accuracy: {final_acc:.4f}")
        print(f"{'-'*50}\n")

    print(f"Time taken: {end_time-start_time:.2f} sec")
    return train_losses, test_losses, test_accs


def evaluateModel(model, test_loader, criterion, device):
    """
    Evaluate the model on the test_loader.

    Args:
        model     : KVMemNet instance
        test_loader: DataLoader for the test set (q, k, v, tgt)
        criterion : loss function (e.g. nn.CrossEntropyLoss())
        device    : torch.device ('cuda' or 'cpu')

    Returns:
        avg_loss  : float, average test loss
        accuracy  : float, fraction of correct predictions
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for q_batch, k_batch, v_batch, tgt in test_loader:
            # move to device
            q = q_batch.to(device)       # (B, V)
            k = k_batch.to(device)       # (B, K, V)
            v = v_batch.to(device)       # (B, M, V)
            t = tgt.to(device)           # (B,)

            # forward through A
            o = model(q, k, v)           # (B, E)

            # embed values via B
            v_emb = model.B(v)           # (B, M, E)

            # compute scores and loss
            # scores[b,i] = o[b] · v_emb[b,i]
            scores = torch.bmm(o.unsqueeze(1), v_emb.transpose(1,2)).squeeze(1)  # (B, M)
            loss   = criterion(scores, t)

            # accumulate
            total_loss += loss.item() * q.size(0)
            preds = scores.argmax(dim=1)
            correct += (preds == t).sum().item()
            total   += q.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    model.train()  # back to train mode
    return avg_loss, accuracy

def plot_learning_curve(train_losses, test_losses=None, test_accs=None):
    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    if test_losses:
        plt.plot(epochs, test_losses, marker='o', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve — Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot test accuracy if available
    if test_accs:
        plt.plot(epochs, test_accs, marker='o', label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve — Test Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()


# ——————————————————————————————————————————————————————————

# Part E: Use the Model

# —————————————————————————————
# 1. Entity & relation extraction
# —————————————————————————————

def extract_entity(question: str, DB: dict) -> str:
    """Return the best matching DB key (person) in the question."""
    q_low = question.lower()
    candidates = [name for name in DB if name.lower() in q_low]
    if not candidates:
        raise ValueError("No entity found in question")
    # pick the longest match (to avoid short substrings)
    return max(candidates, key=len)

def extract_relation(question: str, top_relations: list) -> str:
    """Return the top_relation whose keywords best match the question."""
    q_low = question.lower()
    # 1) direct substring match of "birth date" → birth_date
    for rel in top_relations:
        phrase = rel.replace('_',' ')
        if phrase in q_low:
            return rel
    # 2) fallback: max token overlap
    q_tokens = set(q_low.split())
    best, best_score = None, -1
    for rel in top_relations:
        score = len(q_tokens & set(rel.split('_')))
        if score > best_score:
            best, best_score = rel, score
    return best

# —————————————————————————————
# 2. Candidate construction
# —————————————————————————————

def build_candidates(entity: str,
                     DB: dict,
                     n_distractors: int = 5):
    """
    For `entity`, return:
      keys:   List[str] of that person’s keys + distractor keys.
      values: List[str] corresponding to each key.
    """
    facts = DB[entity]
    person_keys   = list(facts.keys())
    person_values = list(facts.values())
    dk, dv = get_distractor_keys_values(DB, entity, n_distractors)
    return person_keys + dk, person_values + dv

# —————————————————————————————
# 3. Encoding helpers
# —————————————————————————————

def encode_batch(question: str,
                 keys: list[str],
                 values: list[str],
                 vocab) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (q_t, k_t, v_t) ready for model input. Shapes:
      q_t: (1, V)
      k_t: (1, K, V)
      v_t: (1, M, V)
    """
    q_vec = safe_multihot(question, vocab)
    k_mat = encode_texts_to_multihot(keys,   vocab)
    v_mat = encode_texts_to_multihot(values, vocab)

    # to torch and add batch dim
    q_t = torch.from_numpy(q_vec[None]).float()
    k_t = torch.from_numpy(k_mat[None]).float()
    v_t = torch.from_numpy(v_mat[None]).float()
    return q_t, k_t, v_t

# —————————————————————————————
# 4. Scoring & selection
# —————————————————————————————

def score_and_select(model, q_t, k_t, v_t, values: list[str]) -> str:
    """
    Runs the model forward, scores each value, picks argmax, and returns the text.
    """
    model.eval()
    q_t = q_t.to(device)
    k_t = k_t.to(device)
    v_t = v_t.to(device)

    with torch.no_grad():
        o     = model(q_t, k_t, v_t)        # (1, E)
        v_emb = model.B(v_t)                # (1, M, E)
        scores = torch.bmm(
            o.unsqueeze(1),
            v_emb.transpose(1,2)
        ).squeeze(1)                        # (1, M) → (M,)
        idx = scores.argmax().item()

    return values[idx]

def QASystem(model, question, data, vocab, top_relations, n_distractors=5):
    # 1. Extract
    entity  = extract_entity(question, data)
    relation = extract_relation(question, top_relations)

    # 2. Build candidates
    keys, values = build_candidates(entity, data, n_distractors)

    # 3. (Optional) Filter to only those key/value pairs matching `relation`
    #    or reorder so that the true relation is first.
    #    E.g.:
    #    kv = [(k,v) for k,v in zip(keys,values)]
    #    kv.sort(key=lambda x: x[0]!=relation)
    #    keys, values = zip(*kv)

    # 4. Encode
    q_t, k_t, v_t = encode_batch(question, keys, values, vocab)

    # 5. Score & return
    return score_and_select(model, q_t, k_t, v_t, values)



# ——————————————————————————————————————————————————————————


# Load the data
q_template = load_template()
DB = load_data(path, name="data")

# Prep dataset
top_relations = get_top_relations(DB, top_n=5)
VOCAB_EXP = load_data(path, name="vocab_expanded")

train_DB, test_DB = filter_db_by_person(DB, max_train=500, max_test=100)
TRAIN_DATA = load_data(path, name="TRAIN")
TEST_DATA = load_data(path, name="TEST")
# Train samples: 4374
# Test samples: 866


# PART D - TRAINING

# FULL DATA HYPERPARAMETERS
VOCAB_SIZE = VOCAB_EXP.num_words()
EMBED_SIZE = 200
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

from torch.utils.data import DataLoader

train_ds = QADataset(TRAIN_DATA)
test_ds  = QADataset(TEST_DATA)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=qa_collate_fn,
    pin_memory=True      # for GPU
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=qa_collate_fn
)

KVMemNetModel = KVMemNet(VOCAB_SIZE, EMBED_SIZE).to(device)
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.AdamW(KVMemNetModel.parameters(), lr=LEARNING_RATE)

train_losses, test_losses, test_accs = trainModel(
    model=KVMemNetModel,
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=NUM_EPOCHS,
    criterion=CRITERION,
    optimizer=OPTIMIZER,
    save_path="model_KVMemNet.pt"
)

# RESULTS
#  Final Evaluation:
# --------------------------------------------------
# Final Test Loss:     2.2167
# Final Test Accuracy: 0.5958
# --------------------------------------------------
# Time taken: 951.25 sec

test_loss, test_acc = evaluateModel(KVMemNetModel, test_loader, CRITERION, device)

# ——————————————————————————————————————————————————————————

# Part E: USE THE MODEL

model = torch.load("model_KVMemNet.pt")

# question = "When was Alexander Hamilton born?"
question = "when was jean-marie le pen born?"

answer = QASystem(
    model          = model,
    question       = question,
    data           = DB,
    vocab          = VOCAB_EXP,
    top_relations  = top_relations,
)

print(f"Q: {question}\nA: {answer}")

# OUTPUT:
# Q: when was jean-marie le pen born?
# A: la trinité brittany france

# GROUND TRUTH DB
# {'name': 'jean le pen',
#  'office': 'president of the national front member of the european parliament member of the french national assembly regional councillor municipal councillor',
#  'term_start': '5 october 1972 10 june 2004 24 june 1984 2 april 1986 19 january 1956 21 march 2010 22 march 1992 16 march 1986 13 march 1983',
#  'term_end': '15 january 2011 10 april 2003 14 march 1988 9 october 1962 24 february 2000 22 march 1992 19 march 1989',
#  'predecessor': 'position established',
#  'successor': 'marine le pen',
#  'constituency': 'south france france paris 3rd district of the seine provence d provence d Île 20th arrondissement of paris',
#  'birth_date': '20 june 1928',
#  'birth_place': 'la trinité brittany france',
#  'nationality': 'france',
#  'ethnicity': 'breton april 2012',
#  'party': 'independent 2015 national front 1972 cni 1958 uff 1956 action francaise from 1947',
#  'spouse': 'pierrette lalanne 1960 1987 jeanne paschos 1991 present',
#  'relations': 'marine le pen daughter marion maréchal pen granddaughter',
#  'children': '3',
#  'religion': 'roman catholic april 2012',
#  'signature': 'jean le pen signature',
#  'allegiance': 'france',
#  'branch': 'french army',
#  'unit': '20px foreign legion 1st foreign parachute regiment',
#  'serviceyears': '1953 55 1956',
#  'rank': 'intelligence officer',
#  'battles': 'first indochina war suez crisis algerian war',
#  'awards': '20px cross for military valour 20px combatant cross 20px colonial medal 20px indochina 20px north africa 20px middle east',
#  'article_title': 'jean le pen'}


# OUTPUT:
# Q: When was Alexander Hamilton born?
# A: january 31 1795 june 15 1800 march 2 1789 june 21 1783

# GROUND TRUTH DB
# {'name': 'alexander hamilton',
#  'office': '1st united states secretary of the treasury senior officer of the army delegate to the congress of the confederation from new york',
#  'president': 'george washington john adams',
#  'term_start': 'september 11 1789 december 14 1799 november 3 1788 november 4 1782',
#  'term_end': 'january 31 1795 june 15 1800 march 2 1789 june 21 1783',
#  'predecessor': 'position established george washington egbert benson seat established',
#  'successor': 'oliver wolcott jr james wilkinson seat abolished seat abolished',
#  'birth_date': '11 january 1755',
#  'birth_place': 'charlestown nevis british west indies',
#  'death_date': 'july 12 1804 aged 47 or 49',
#  'death_place': 'new york city new york u',
#  'party': 'federalist',
#  'spouse': 'elizabeth schuyler',
#  'children': 'philip angelica alexander james alexander john church william stephen eliza holly phil',
#  'alma_mater': 'kings college new york',
#  'religion': 'presbyterian episcopalian convert',
#  'signature': 'alexander hamilton signaturert',
#  'allegiance': 'flag_of_new_york _ 1778 svg 23px new york 1775 1777 united states 1795 23px 1777 1800',
#  'branch': 'flag_of_new_york _ 1778 svg 23px new york company of artillery united states 1777 23px continental army 25px united states army',
#  'serviceyears': '1775 1776 militia 1776 1781 1798 1800',
#  'rank': '23px',
#  'article_title': 'alexander hamilton'}