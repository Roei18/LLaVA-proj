import argparse
import shutil
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import llava.mm_utils as mm_utils


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, anyres  =None):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.anyres = anyres
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0  # safe fallback

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        if self.anyres  :
            patches_height = 2
            patches_width = 2
            full_width = patches_width * 336
            full_height = patches_height * 336
            grid_pinpoints = [[full_width, full_height]]  # i.e., [[1344, 1344]]
            image_tensor, num_of_patches = mm_utils.process_anyres_image(image, self.image_processor, grid_pinpoints)
        else:
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


# def collate_fn(batch):
#     input_ids, image_tensors, image_sizes = zip(*batch)
#     input_ids = torch.stack(input_ids, dim=0)
#     image_tensors = torch.stack(image_tensors, dim=0)
#     return input_ids, image_tensors, image_sizes

import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_token_id=0):
    """
    Collate function for batches containing variable-length input_ids
    plus image tensors and image sizes.

    Args:
        batch: list of (input_ids, image_tensor, image_size)
        pad_token_id: int, the padding value for input_ids

    Returns:
        input_ids_padded: (B, T) tensor
        attention_mask:   (B, T) tensor (1 = real token, 0 = padding)
        image_tensors:    (B, ...) stacked tensor
        image_sizes:      list of image sizes
    """
    input_ids, image_tensors, image_sizes = zip(*batch)

    # Pad input_ids to max length in this batch
    input_ids_padded = pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )

    # Build attention mask (1 where not padding)
    attention_mask = (input_ids_padded != pad_token_id).long()

    # Stack images (assumes same size within batch; else you'd need resize/collate separately)
    image_tensors = torch.stack(image_tensors, dim=0)

    return input_ids_padded, attention_mask, image_tensors, image_sizes

# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, image_aspect_ratio=None):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, anyres=(image_aspect_ratio == 'anyres'))
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def create_7b_dups(path):
    # creates a 7b duplicate to each file that has 13b in it's name in the path
    # if a file, simply replace the 13b with 7b
    import os
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if "13b" in filename:
                new_filename = filename.replace("13b", "7b")
                shutil.copy(os.path.join(path, filename), os.path.join(path, new_filename))

    else:
        if '13b' in path:
            new_filename = path.replace("13b", "7b")
            os.rename(path, new_filename)

    print(f"Created 7b duplicates in {path} for files with '13b' in their name.")

import torch

def probe_issues(model, tokenizer, data_loader):
    import torch

    # --- 0) Constants for image tokens (handle scope/import gracefully)
    try:
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        )
    except Exception:
        DEFAULT_IMAGE_TOKEN = "<image>"
        DEFAULT_IM_START_TOKEN = "<im_start>"
        DEFAULT_IM_END_TOKEN = "<im_end>"

    # --- 1) Vocab / tokenizer alignment
    emb_in = model.get_input_embeddings().weight.shape[0]
    emb_out = model.get_output_embeddings().weight.shape[0]
    tok_v = len(tokenizer)
    print("[Vocab] emb_in:", emb_in, "emb_out:", emb_out, "tokenizer_len:", tok_v)
    assert emb_in == emb_out == tok_v, "Vocab mismatch between model and tokenizer!"
    print("[Tokens] pad/eos/unk:", tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id)
    try:
        print("[Tokenizer] added vocab (head):", list(tokenizer.get_added_vocab().keys())[:10])
    except Exception as e:
        print("[Tokenizer] get_added_vocab() not available:", e)

    # --- 2) Grab one batch
    batch = next(iter(data_loader))
    # data_loader returns (input_ids, attention_mask, image_tensors, image_sizes)
    input_ids, attn_mask, image_tensors, image_sizes = batch

    # Shapes / dtypes
    print("[Batch] input_ids:", tuple(input_ids.shape), input_ids.dtype)
    if attn_mask is not None:
        print("[Batch] attention_mask:", tuple(attn_mask.shape), attn_mask.dtype,
              "| row sums:", attn_mask.sum(dim=1).tolist())
    else:
        print("[Batch] attention_mask: None")
    print("[Batch] image_tensors:", tuple(image_tensors.shape), image_tensors.dtype)
    print("[Batch] image_sizes (first):", image_sizes[0] if isinstance(image_sizes, (list, tuple)) else image_sizes)

    # --- 3) Decode prompt to verify image tokens appear in-text
    # (skip_special_tokens=False so we can actually *see* <image> / <im_start>/<im_end>)
    try:
        decoded0 = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
    except Exception:
        decoded0 = tokenizer.batch_decode(input_ids[:1], skip_special_tokens=False)[0]

    head = decoded0[:400].replace("\n", "\\n")
    print("[Prompt head <=400 chars]", head)

    has_img = (DEFAULT_IMAGE_TOKEN in decoded0)
    has_imwrap = (DEFAULT_IM_START_TOKEN in decoded0) and (DEFAULT_IM_END_TOKEN in decoded0)
    print(f"[Prompt] has DEFAULT_IMAGE_TOKEN={has_img} | has <im_start>/<im_end>={has_imwrap}")
    print("[Prompt] model.config.mm_use_im_start_end:", getattr(model.config, "mm_use_im_start_end", None))

    if not has_img:
        print("!!! Prompt does not contain the image token. The model may ignore the image.")
    if getattr(model.config, "mm_use_im_start_end", False) and not has_imwrap:
        print("!!! Model expects <im_start>/<im_end> but prompt lacks them.")
    if (not getattr(model.config, "mm_use_im_start_end", False)) and has_imwrap:
        print("!!! Prompt uses <im_start>/<im_end> but model config says not to.")

    # --- 4) Attention mask sanity
    if attn_mask is not None:
        if attn_mask.dtype not in (torch.long, torch.int64, torch.bool):
            print("!!! attention_mask has odd dtype:", attn_mask.dtype)
        zero_rows = (attn_mask.sum(dim=1) == 0).nonzero(as_tuple=True)[0].tolist()
        if zero_rows:
            print(f"!!! attention_mask has all-zeros rows at indices: {zero_rows}")
    else:
        print("(!) No attention_mask provided; HF will infer it from padding.")

    # --- 5) Image tensor stats (quick NaN/scale check)
    with torch.no_grad():
        mean_val = float(image_tensors.mean())
        std_val = float(image_tensors.std())
        has_nans = torch.isnan(image_tensors).any().item()
    print(f"[Images] mean={mean_val:.4f} std={std_val:.4f} nans={has_nans}")

    # --- 6) Vision tower quick sanity (if available)
    vt = getattr(model, "get_vision_tower", None)
    if callable(vt):
        try:
            vt_mod = vt()
            print("[Vision] tower type:", type(vt_mod))
            with torch.no_grad():
                feats = vt_mod(image_tensors)
                if isinstance(feats, (list, tuple)): feats = feats[0]
                if isinstance(feats, dict) and "last_hidden_state" in feats:
                    feats = feats["last_hidden_state"]
                if torch.is_tensor(feats):
                    print("[Vision] feats:", tuple(feats.shape), feats.dtype,
                          "nan?", torch.isnan(feats).any().item())
                else:
                    print("[Vision] unexpected feature type:", type(feats))
        except Exception as e:
            print("[Vision] forward exception:", repr(e))
    else:
        print("[Vision] get_vision_tower() not found; skipping tower probe.")

    # --- 7) Text-only control (is the LM itself sane?)
    try:
        test_prompt = "You are a helpful assistant.\nUser: Say 'hello' twice.\nAssistant:"
        ids = tokenizer([test_prompt], return_tensors="pt", padding=True).input_ids.to(input_ids.device)
        mask = ids.ne(tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else None
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                attention_mask=mask,
                temperature=0.0,
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        print("[Text-only control]", tokenizer.decode(out[0], skip_special_tokens=True))
    except Exception as e:
        print("[Text-only control] generation failed:", repr(e))

    print("== Probe complete ==")



def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    create_7b_dups(args.question_file)
    create_7b_dups(args.answers_file)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, batch_size=2, image_aspect_ratio=args.image_aspect_ratio)
    q_ptr = 0
    probe_issues(model, tokenizer, data_loader, max_new_tokens=args.max_new_tokens)
    return
    for (input_ids, attention_mask, image_tensors, image_sizes) in tqdm(data_loader, total=len(data_loader)):
        batch_size = input_ids.size(0)
        batch_questions = questions[q_ptr : q_ptr + batch_size]
        q_ptr += batch_size

        # move to device
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        attention_mask = attention_mask.to(device="cuda", non_blocking=True)
        image_tensors = image_tensors.to(device="cuda", dtype=torch.float16, non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                # attention_mask=attention_mask,              # <- now provided
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id 
            )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        decoded = [s.strip() for s in decoded]

        # write one line per item in the batch
        for i in range(batch_size):
            idx = batch_questions[i]["question_id"]
            cur_prompt = batch_questions[i]["text"]
            ans_id = shortuuid.uuid()

            record = {
                "question_id": idx,
                "prompt": cur_prompt,
                "text": decoded[i],
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {}
            }
            ans_file.write(json.dumps(record) + "\n")
            print(f"Question ID: {idx}, Prompt: {cur_prompt}, Output: {decoded[i]}")

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--image_aspect_ratio", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
