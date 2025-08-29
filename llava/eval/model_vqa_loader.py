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

def probe_issues(model, tokenizer, data_loader, max_new_tokens=16):
    # ---------- 0) Vocab integrity ----------
    emb_in = model.get_input_embeddings().weight.shape[0]
    emb_out = model.get_output_embeddings().weight.shape[0]
    tok_v = len(tokenizer)
    print("Vocab sizes -> emb_in:", emb_in, "emb_out:", emb_out, "tok:", tok_v)
    assert emb_in == emb_out == tok_v, "Vocab mismatch!"
    print("Pad/EOS/UNK ids:", tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id)
    print("Added vocab keys (head):", list(tokenizer.get_added_vocab().keys())[:10])

    # Ensure pad id exists (some trainings set pad=eos/unk)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad token was None; set pad_token to eos. New pad id:", tokenizer.pad_token_id)

    # ---------- 1) Take one batch ----------
    input_ids, attention_mask, image_tensors, image_sizes = next(iter(data_loader))
    bsz = input_ids.size(0)
    print(f"\nBatch sizes -> input_ids: {tuple(input_ids.shape)}, mask: {tuple(attention_mask.shape)}, "
          f"images: {tuple(getattr(image_tensors, 'shape', ('list', len(image_tensors))))}, "
          f"image_sizes(len): {len(image_sizes)}")

    # Mask sanity
    with torch.no_grad():
        sums = attention_mask.sum(dim=1).tolist()
    print("Attention mask sum per row:", sums[:8], ("..." if bsz > 8 else ""))

    # Peek at tokens/decoded prompt head
    print("First 60 token ids:", input_ids[0, :60].tolist())
    try:
        head_decoded = tokenizer.decode(input_ids[0, :256], skip_special_tokens=False)
        print("Decoded prompt head (≤256 toks):\n", head_decoded[:400])
    except Exception as e:
        print("Decode error:", e)

    # ---------- 2) Devices & dtypes ----------
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    print("Model device:", device)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    # image_tensors may be a tensor or a list of tensors (anyres/sizes)
    if isinstance(image_tensors, torch.Tensor):
        image_tensors = image_tensors.to(device, dtype=torch.float16 if any(
            getattr(model.config, k, False) for k in ("fp16", "bf16")) else torch.float32)
        print("Image tensor -> dtype/device:", image_tensors.dtype, image_tensors.device,
              "mean/std:", float(image_tensors.mean()), float(image_tensors.std()))
    else:
        # list path
        for i, t in enumerate(image_tensors[:2]):
            print(f"Image[{i}] shape/dtype/device:", tuple(t.shape), t.dtype, t.device)

    # ---------- 3) Vision tower quick check ----------
    vt = getattr(model, "get_vision_tower", None)
    if callable(vt):
        try:
            vision_tower = vt()
            print("Vision tower:", type(vision_tower))
            if isinstance(image_tensors, torch.Tensor):
                with torch.no_grad():
                    feats = vision_tower(image_tensors)
                    if isinstance(feats, (list, tuple)):
                        feats = feats[0]
                    if torch.is_tensor(feats):
                        print("Vision feats:", tuple(feats.shape), feats.dtype,
                              "NaN?", bool(torch.isnan(feats).any().item()))
        except Exception as e:
            print("Vision forward exception:", repr(e))
    else:
        print("No get_vision_tower() found on model (may be fine).")

    # ---------- 4) Deterministic generation (with mask) ----------
    gen_kwargs = dict(
        input_ids=input_ids,                       # use explicit input_ids
        attention_mask=attention_mask,             # first A/B includes mask
        images=image_tensors,
        image_sizes=image_sizes,
        temperature=0.0, top_p=None, num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    print("\n=== GENERATE (with attention_mask) ===")
    try:
        with torch.no_grad():
            out1 = model.generate(**gen_kwargs)
        dec1 = tokenizer.batch_decode(out1, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print("Decoded[0]:", (dec1[0][:400] + ("..." if len(dec1[0]) > 400 else "")))
    except Exception as e:
        print("Generate-with-mask exception:", repr(e))

    # ---------- 5) A/B: generation WITHOUT attention_mask ----------
    gen_kwargs_nomask = dict(gen_kwargs)
    gen_kwargs_nomask.pop("attention_mask", None)
    print("\n=== GENERATE (without attention_mask) ===")
    try:
        with torch.no_grad():
            out2 = model.generate(**gen_kwargs_nomask)
        dec2 = tokenizer.batch_decode(out2, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print("Decoded[0]:", (dec2[0][:400] + ("..." if len(dec2[0]) > 400 else "")))
    except Exception as e:
        print("Generate-no-mask exception:", repr(e))

    # ---------- 6) Text-only control (LM health) ----------
    print("\n=== TEXT-ONLY CONTROL ===")
    try:
        test_prompt = "You are a helpful assistant.\nUser: Say 'hello' twice.\nAssistant:"
        ids = tokenizer([test_prompt], return_tensors="pt", padding=True).input_ids.to(device)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        msk = ids.ne(tokenizer.pad_token_id).long()
        with torch.no_grad():
            out_txt = model.generate(
                input_ids=ids, attention_mask=msk,
                temperature=0.0, top_p=None, num_beams=1,
                max_new_tokens=12,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        print("TEXT-ONLY decoded:", tokenizer.decode(out_txt[0], skip_special_tokens=True))
    except Exception as e:
        print("Text-only generate exception:", repr(e))

    print("\nProbe complete.")




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
