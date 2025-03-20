# data/dataset.py
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

def read_jsonl(file_path: str) -> List[Dict]:
    """Read JSONL file and return list of parsed dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Strip whitespace and skip empty lines
                line = line.strip()
                if not line:
                    continue
                # Parse JSON line
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")
                continue
    return data

class ConversationDataset(Dataset):
   def __init__(self, texts: List[str], tokenizer: AutoTokenizer):
       self.texts = texts
       self.tokenizer = tokenizer

   def __len__(self):
       return len(self.texts)

   def __getitem__(self, idx):
       text = self.texts[idx]
       return {
           'input_text': text,
           'target_text': text
       }

def collate_conversations(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
   """Custom collate function for padding variable length conversations"""
   input_texts = [item['input_text'] for item in batch]
   target_texts = [item['target_text'] for item in batch]
   
   input_lengths = [len(text.split()) for text in input_texts]
   sorted_indices = sorted(range(len(input_lengths)), 
                         key=lambda k: input_lengths[k], 
                         reverse=True)
   
   input_texts = [input_texts[i] for i in sorted_indices]
   target_texts = [target_texts[i] for i in sorted_indices]
   
   return {
       'input_text': input_texts,
       'target_text': target_texts,
       'lengths': torch.tensor(sorted([len(text.split()) for text in input_texts]))
   }

def prepare_input(
   mamba_model,
   llm_model,
   llm_tokenizer: AutoTokenizer,
   system_prompt: str,
   input_texts: List[str],
   device: str = 'cuda',
   end_sym: str = '\n',
   max_length: int = 512
):
    logger.info(f"{input_texts}")
    logger.info(f"batch size: {len(input_texts)}")
   # Tokenize inputs with padding
    input_ids = llm_tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512
    ).to(device)

    logger.info(f"input_ids: {input_ids}")  

    memory_features = mamba_model(input_ids).to(torch.float16)
    atts_memory = torch.ones(
            (memory_features.size(0), memory_features.size(1)),
            dtype=torch.long,
    ).to(device)
   
    # Add system prompt
    system_encodings = llm_tokenizer(
        [system_prompt] * len(input_texts),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=64
    ).to(device)
    system_embeds = llm_model.model.embed_tokens(system_encodings['input_ids']) # (batch, seq, hidden)
    memory_features = torch.cat([system_embeds, memory_features], dim=1)
    atts_memory = atts_memory[:, :1].expand(-1, memory_features.size(1))

    target_texts = [t + end_sym for t in input_texts]
    to_regress_tokens = llm_tokenizer(
        target_texts,
        truncation=True,
        return_tensors="pt",
        padding="longest",
        max_length=512,
    ).to(device)
    targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == llm_tokenizer.pad_token_id, -100
            )
    empty_targets = (
                torch.ones([memory_features.shape[0], memory_features.shape[1]],
                        dtype=torch.long).to(device).fill_(-100)
            )
    targets = torch.cat([empty_targets, targets], dim=1)

    to_regress_embeds = llm_model.model.embed_tokens(to_regress_tokens.input_ids)
    input_embeds = torch.cat([memory_features, to_regress_embeds], dim=1)
    attention_mask = torch.cat([atts_memory, to_regress_tokens.attention_mask], dim=1)

    return {
        'input_embeds': input_embeds,
        'attention_mask': attention_mask,
        'labels': targets
    }

"""
jsonl data format: 
{"Utterance": "But I thought you said it came out of nowhere. You were watching tv, then suddenly...", "Emotion": "neutral", "Strategy": "Restatement", "Dialogue_ID": 10, "history_chat_mamba": "", "history_chat": [], "path_to_vid_user_most_recent": ["dia10_utt0_288.mp4", "dia10_utt1_289.mp4", "dia10_utt2_290.mp4", "dia10_utt3_291.mp4", "dia10_utt4_292.mp4"], "utt_user_most_recent": "I mean, shit, don't you know that men are the new women? Obsessed with weddings and children. And lately he's lost touch with reality. He he thinks I'm seeing someone. That's how this whole thing started. It's unbelievable.", "get_emotion_user_most_recent": "anger", "situation": "My boyfriend suddenly proposed breaking up without getting married while we were having dinner and watching TV.", "problem_type": "Breakups or Divorce", "set": "test"}
{"Utterance": "So you were the one who came up with it.", "Emotion": "neutral", "Strategy": "Interpretation", "Dialogue_ID": 10, "history_chat_mamba": "user ( user Emotion : anger , user Expresstion : I mean, shit, don't you know that men are the new women? The speaker seems to be in a calm and relaxed state. She is sitting on a couch and looking at the camera with a smile on her face.The speaker's emotional expression in this picture could be due to a variety of reasons, including her personal life, relationships, or work-related stress. However, without further information, it is difficult to determine the specific cause of her emotional state. . Obsessed with weddings and children. The speaker seems to be in a state of contemplation or deep thought, as he is looking down and talking on the phone while sitting at a table.The speaker's emotional expression could be due to a personal or professional problem that he is trying to solve. . And lately he's lost touch with reality. The speaker seems to be in a calm and relaxed state, as she is sitting on a couch and talking to someone.The speaker's emotional expression in this picture could be a sign of sadness or depression, as she is sitting on a couch and talking to someone in a calm and relaxed state. . He he thinks I'm seeing someone. That's how this whole thing started. The speaker appears to be calm and relaxed throughout the video.The speaker's emotional expression could be due to a variety of reasons, such as stress, anxiety, or frustration. However, without more context, it is difficult to determine the specific cause of the speaker's emotional state. . It's unbelievable. The speaker appears to be in a state of contemplation, as he is sitting in a chair and looking down. He seems to be deep in thought, possibly pondering a difficult decision or problem.The speaker's emotional expression could be a sign of stress or frustration, as he is sitting in a chair and looking down. It is possible that he is dealing with a difficult situation or problem that is causing him to feel overwhelmed or upset.)  : I mean, shit, don't you know that men are the new women? Obsessed with weddings and children. And lately he's lost touch with reality. He he thinks I'm seeing someone. That's how this whole thing started. It's unbelievable. <MEM>  sys ( sys Emotion : neutral , sys Strategy : Restatement ) : But I thought you said it came out of nowhere. You were watching tv, then suddenly... <MEM> ", "history_chat": [{"role": "user", "content": "( user emotion : anger , user expresstion : I mean, shit, don't you know that men are the new women? The speaker seems to be in a calm and relaxed state. She is sitting on a couch and looking at the camera with a smile on her face.The speaker's emotional expression in this picture could be due to a variety of reasons, including her personal life, relationships, or work-related stress. However, without further information, it is difficult to determine the specific cause of her emotional state. . Obsessed with weddings and children. The speaker seems to be in a state of contemplation or deep thought, as he is looking down and talking on the phone while sitting at a table.The speaker's emotional expression could be due to a personal or professional problem that he is trying to solve. . And lately he's lost touch with reality. The speaker seems to be in a calm and relaxed state, as she is sitting on a couch and talking to someone.The speaker's emotional expression in this picture could be a sign of sadness or depression, as she is sitting on a couch and talking to someone in a calm and relaxed state. . He he thinks I'm seeing someone. That's how this whole thing started. The speaker appears to be calm and relaxed throughout the video.The speaker's emotional expression could be due to a variety of reasons, such as stress, anxiety, or frustration. However, without more context, it is difficult to determine the specific cause of the speaker's emotional state. . It's unbelievable. The speaker appears to be in a state of contemplation, as he is sitting in a chair and looking down. He seems to be deep in thought, possibly pondering a difficult decision or problem.The speaker's emotional expression could be a sign of stress or frustration, as he is sitting in a chair and looking down. It is possible that he is dealing with a difficult situation or problem that is causing him to feel overwhelmed or upset. ) "}, {"role": "assistant", "content": "( assistant emotion : neutral  , assistant strategy : Restatement ) : But I thought you said it came out of nowhere. You were watching tv, then suddenly... "}], "path_to_vid_user_most_recent": ["dia10_utt8_296.mp4", "dia10_utt9_297.mp4", "dia10_utt10_298.mp4", "dia10_utt11_299.mp4", "dia10_utt12_300.mp4", "dia10_utt13_301.mp4", "dia10_utt14_302.mp4", "dia10_utt15_303.mp4"], "utt_user_most_recent": "Well, we were watching tv.But before that we were having dinner and he had this look on his face. He had this sad look and I could just tell something was up. I mean, you can just tell, huh? And I said to him, what's going on? And he said, I don't know what I want. I just know I don't want this on-and-off thing. So I said to him, Sure, then let's just decide: either we get married or we split.", "get_emotion_user_most_recent": "neutral", "situation": "My boyfriend suddenly proposed breaking up without getting married while we were having dinner and watching TV.", "problem_type": "Breakups or Divorce", "set": "test"}
{"Utterance": "The ultimatum.", "Emotion": "neutral", "Strategy": "Interpretation", "Dialogue_ID": 10, "history_chat_mamba": "user ( user Emotion : anger : I mean, shit, don't you know that men are the new women? Obsessed with weddings and children. And lately he's lost touch with reality. He he thinks I'm seeing someone. That's how this whole thing started. It's unbelievable. <MEM>  sys ( sys Emotion : neutral , sys Strategy : Restatement ) : But I thought you said it came out of nowhere. You were watching tv, then suddenly... <MEM>  user ( user Emotion : neutral , user Expresstion : I mean, you can just tell, huh? The speaker seems to be in a serious and focused state, as he is giving a lecture or presentation. He appears to be engaged and attentive to his audience.The speaker's emotional expression in this picture might be related to the challenges and difficulties he is facing in his personal or professional life. . And I said to him, what's going on? And he said, The speaker seems to be in a state of confusion or disbelief, as he is looking at the camera with a confused expression on his face.The speaker's emotional expression could be a result of his personal struggles or difficulties in his life. . I don't know what I want. I just know I don't want The speaker seems to be in a state of confusion or disbelief, as she is looking at the camera with a surprised expression on her face.The speaker's emotional expression in this picture might be due to her confusion or disbelief at the situation she is in, as she is looking at the camera with a surprised expression on her face. . this on-and-off thing. So I said to him, The speaker seems to be in a state of confusion or disbelief. She is looking at the camera with a mix of emotions on her face.The speaker's emotional expression in this picture could be due to a variety of reasons, including life distress, confusion, or disbelief. However, without further context, it is difficult to determine the specific cause of her emotional state. . Sure, then let's just decide: either we get married or we split. The speaker seems to be in a calm and relaxed state, as there is no visible emotion on her face.The speaker's emotional expression in this picture might be due to the fact that she is a young woman who is sitting in a restaurant and looking sad.)  : Well, we were watching tv.But before that we were having dinner and he had this look on his face. He had this sad look and I could just tell something was up. I mean, you can just tell, huh? And I said to him, what's going on? And he said, I don't know what I want. I just know I don't want this on-and-off thing. So I said to him, Sure, then let's just decide: either we get married or we split. <MEM>  sys ( sys Emotion : neutral , sys Strategy : Interpretation ) : So you were the one who came up with it. <MEM> ", "history_chat": [{"role": "user", "content": "( user emotion : anger ) : I mean, shit, don't you know that men are the new women? Obsessed with weddings and children. And lately he's lost touch with reality. He he thinks I'm seeing someone. That's how this whole thing started. It's unbelievable. "}, {"role": "assistant", "content": "( assistant emotion : neutral  , assistant strategy : Restatement ) : But I thought you said it came out of nowhere. You were watching tv, then suddenly... "}, {"role": "user", "content": "( user emotion : neutral , user expresstion : I mean, you can just tell, huh? The speaker seems to be in a serious and focused state, as he is giving a lecture or presentation. He appears to be engaged and attentive to his audience.The speaker's emotional expression in this picture might be related to the challenges and difficulties he is facing in his personal or professional life. . And I said to him, what's going on? And he said, The speaker seems to be in a state of confusion or disbelief, as he is looking at the camera with a confused expression on his face.The speaker's emotional expression could be a result of his personal struggles or difficulties in his life. . I don't know what I want. I just know I don't want The speaker seems to be in a state of confusion or disbelief, as she is looking at the camera with a surprised expression on her face.The speaker's emotional expression in this picture might be due to her confusion or disbelief at the situation she is in, as she is looking at the camera with a surprised expression on her face. . this on-and-off thing. So I said to him, The speaker seems to be in a state of confusion or disbelief. She is looking at the camera with a mix of emotions on her face.The speaker's emotional expression in this picture could be due to a variety of reasons, including life distress, confusion, or disbelief. However, without further context, it is difficult to determine the specific cause of her emotional state. . Sure, then let's just decide: either we get married or we split. The speaker seems to be in a calm and relaxed state, as there is no visible emotion on her face.The speaker's emotional expression in this picture might be due to the fact that she is a young woman who is sitting in a restaurant and looking sad. ) "}, {"role": "assistant", "content": "( assistant emotion : neutral  , assistant strategy : Interpretation ) : So you were the one who came up with it. "}], "path_to_vid_user_most_recent": ["dia10_utt17_305.mp4"], "utt_user_most_recent": "With what?", "get_emotion_user_most_recent": "neutral", "situation": "My boyfriend suddenly proposed breaking up without getting married while we were having dinner and watching TV.", "problem_type": "Breakups or Divorce", "set": "test"}

when training MambaCompressor on single utterances only, the Utterance field will be used. the <MEM> token will be added to the end of the input text.
the prepare_single_utterances_data function will read the jsonl file, extract the utterance, convert it into format:
"user ( user Emotion: <Emotion> ) : <Utterance> <MEM>"
or "sys ( sys Emotion: <Emotion>, sys Strategy: <Strategy> ): <Utterance> <MEM>"

when training on multiple utterances, the prepare_multiple_utterances_data function  will read the jsonl file and return list of string extracted from history_chat_mamba field
"""

def prepare_single_utterances_data(jsonl_path: str) -> List[str]:
    """Prepare single utterance training data."""
    data = read_jsonl(jsonl_path)
    formatted_data = []
    
    for item in data:
        if 'Strategy' in item:  # System utterance
            text = (f"sys ( sys Emotion: {item['Emotion']}, "
                   f"sys Strategy: {item['Strategy']} ): {item['Utterance']} <MEM>")
        else:  # User utterance
            text = f"user ( user Emotion: {item['Emotion']} ) : {item['Utterance']} <MEM>"
        
        formatted_data.append(text)
    
    return formatted_data

def prepare_multiple_utterances_data(jsonl_path: str) -> List[str]:
    """Prepare conversation training data."""
    data = read_jsonl(jsonl_path)
    formatted_data = []
    
    for item in data:
        if item.get('history_chat_mamba'):
            formatted_data.append(item['history_chat_mamba'])
    
    return formatted_data