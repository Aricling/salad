import os
import torch
import torch.nn as nn
# import clip
from os.path import join as pjoin

from transformers import AutoModel, AutoTokenizer
from transformers.utils import move_cache

import clip
from utils.lora_util import apply_lora_attn_mlp, init_finetuned_clip_and_freeze
import z_config

from transformers import CLIPModel
class FrozenCLIPTextEncoder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, opt):
        super().__init__()
        move_cache()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.opt = opt
        # self.model, _ = clip.load(opt.clip_version, jit=False, device="cpu", download_root=pjoin(opt.checkpoints_dir, "clip"))
        # clip.model.convert_weights(self.model)
        # self.model.to(opt.device)
        # if opt.clip_version == "ViT-B/32":
        #     self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        #     self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        # elif opt.clip_version == "ViT-L/14":
        #     self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        #     self.model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
        # else:
        #     raise ValueError(f"Invalid CLIP version: {opt.clip_version}")
        clip_version="ViT-B/32"
        self.clip_model, _ = clip.load(clip_version, device='cpu',
                                                jit=False)
        
        self.clip_model = apply_lora_attn_mlp(self.clip_model, encoder_type='text', mlp=True, attn=True)
        self.clip_model, self.clip_model_avg = init_finetuned_clip_and_freeze(self.clip_model)
        self.encode_text = self.clip_encode_text
        print(f"Loaded CLIP text encoder version {clip_version}")

    def clip_encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 75  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length <= default_context_length
            texts, texts_tokens = clip.tokenize(raw_text, context_length=context_length, truncate=True) # [bs, context_length] # if n_tokens > context_length -> will truncate
            texts_lens_list = [len(text_token) for text_token in texts_tokens]
            texts_tokens_padded=torch.zeros([texts.shape[0], default_context_length], dtype=texts.dtype, device=texts.device)
            for i, text_tokens in enumerate(texts_tokens):
                texts_tokens_padded[i, :texts_lens_list[i]] = torch.tensor(text_tokens)

        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate

        texts = texts.to(device)
        texts_tokens_padded = texts_tokens_padded.to(device)
        
        if z_config.get_diy_config().model.use_avg_clip_model:
            return self.clip_model_avg.encode_text(texts, texts_lens_list=texts_lens_list).float(), texts_lens_list
        return self.clip_model.encode_text(texts, texts_lens_list=texts_lens_list).float(), texts_lens_list

    def freeze(self):
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    # @torch.no_grad()
    # def encode_text(self, text):
    #     # text: [B, T]
    #     # CLIP embedding dimension D
    #     # tokens = clip.tokenize(text, truncate=True).to(self.opt.device)
    #     # word_emb = self.model.token_embedding(tokens).type(dtype)
    #     # word_emb = word_emb + self.model.positional_embedding.type(dtype) # [B, T, D]
    #     # word_emb = word_emb.permute(1, 0, 2)
    #     # word_emb = self.model.transformer(word_emb)
    #     # word_emb = word_emb.permute(1, 0, 2)
    #     # word_emb = self.model.ln_final(word_emb).type(dtype) # [B, T, D]
    #     tokens = self.tokenizer(text,
    #                             padding="max_length",
    #                             truncation=True,
    #                             max_length=self.max_length,
    #                             return_tensors="pt")
    #     text_input_ids = tokens.input_ids.to(self.model.device)
    #     text_attn_mask = tokens.attention_mask.to(self.model.device).bool()
    #     if text_input_ids.shape[-1] > self.max_length:
    #         text_input_ids = text_input_ids[:, :self.max_length]
        
    #     word_emb = self.model.text_model(text_input_ids).last_hidden_state

    #     return word_emb, text_attn_mask, text_input_ids.argmax(dim=-1)
    
    @torch.no_grad()
    def tokenize(self, text):
        tokens = self.tokenizer(text,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt")
        return tokens

    @torch.no_grad()
    def decode_text_from_tokens(self, tokens):
        return self.tokenizer.decode(tokens)