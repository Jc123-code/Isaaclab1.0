import os
from transformers import AutoModel, pipeline, AutoTokenizer, CLIPTextModelWithProjection

os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
tokenizer = "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32"

# 延迟加载模型
lang_emb_model = None
tz = None

def get_lang_emb_model():
    """延迟加载CLIP模型"""
    global lang_emb_model
    if lang_emb_model is None:
        try:
            lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
                tokenizer,
                cache_dir=os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/tmp"), "clip"))
            ).eval()
        except Exception as e:
            print(f"Warning: Failed to load CLIP model: {e}")
            print("Language embedding features will be disabled")
    return lang_emb_model

def get_tokenizer():
    """延迟加载tokenizer"""
    global tz
    if tz is None:
        try:
            tz = AutoTokenizer.from_pretrained(tokenizer, TOKENIZERS_PARALLELISM=True)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
    return tz

LANG_EMB_OBS_KEY = "lang_emb"

def get_lang_emb(lang):
    if lang is None:
        return None
    
    # 使用函数获取tokenizer和模型
    current_tz = get_tokenizer()
    current_model = get_lang_emb_model()
    
    if current_tz is None or current_model is None:
        return None
    
    tokens = current_tz(
        text=lang,                   # the sentence to be encoded
        add_special_tokens=True,             # Add [CLS] and [SEP]
        max_length=25,  # maximum length of a sentence
        padding="max_length",
        return_attention_mask=True,        # Generate the attention mask
        return_tensors="pt",               # ask the function to return PyTorch tensors
    )
    lang_emb = current_model(**tokens)['text_embeds'].detach()[0]

    return lang_emb

def get_lang_emb_shape():
    return list(get_lang_emb('dummy').shape)