#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def handle_fga(model, compute_dtype, device):
    print('FGA detected in model, adding...')
    patches_height = 2
    patches_width = 2
    num_of_patches = patches_height * patches_width + 1
    sizes = [None] 
    sizes.extend([576 for _ in range(num_of_patches)])
    text_dimension = model.config.hidden_size
    vision_dimension = model.get_vision_tower().config.hidden_size
    util_e = [text_dimension] + [vision_dimension for _ in range(num_of_patches)]
    sharing_factor = {}

    # First image patch - full images.
    sharing_factor[1] = (1, [0])
    # Following patches - image patches. Similar modalities. 
    similar_modalities = [[i for i in range(2, num_of_patches + 1)]]
    sharing_factor[2] = (1, [0])

    model.fga = model.initialize_fga(util_e, sharing_factor, False, sizes, size_force=False, similar_modalities=similar_modalities).to(dtype=compute_dtype, device=device)
    print(hasattr(model, "atten") and model.atten is not None)

def move_all_tensors_to_device(model, device):
    for param in model.parameters():
        param.data = param.data.to(device)
    print(f"Moved all model parameters to {device}")
    model.to(device)
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    
    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            elif os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
                # for the case it was saved in this name
                non_lora_trainables = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')

            keys = non_lora_trainables.keys()
            if any(('fga' in key) or '.atten.' in key for key in keys):
                handle_fga(model, torch.float16, device)
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            incompatible = model.load_state_dict(non_lora_trainables, strict=False)
            for key in incompatible.missing_keys:
                print(f"⚠️ Missing key when loading LLaVA weights: {key}")

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
            print("=== Loading base weights (non-LoRA) ===")
            incompatible = model.load_state_dict(non_lora_trainables, strict=False)

            # Collect issues before LoRA
            missing_before = incompatible.missing_keys
            unexpected_before = incompatible.unexpected_keys

            if missing_before:
                print(f"⚠️ Missing before LoRA: {len(missing_before)} keys")
            if unexpected_before:
                print(f"⚠️ Unexpected before LoRA: {len(unexpected_before)} keys")

            print("\n=== Loading & merging LoRA adapters ===")
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()

            # Final check: does every parameter in the model have a value?
            final_state = model.state_dict()
            missing_after = [k for k, v in final_state.items() if v is None]

            print("\n=== Final Weight Load Report ===")
            if not missing_before and not unexpected_before and not missing_after:
                print("✅ All model weights are accounted for (base + LoRA).")
            else:
                if missing_before:
                    print("\n⚠️ Missing before LoRA:")
                    for k in missing_before:
                        print(f"  - {k}")
                if unexpected_before:
                    print("\n⚠️ Unexpected before LoRA:")
                    for k in unexpected_before:
                        print(f"  - {k}")
                if missing_after:
                    print("\n⚠️ Still missing after LoRA merge:")
                    for k in missing_after:
                        print(f"  - {k}")

        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                except:
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                from llava.model.language_model.llava_llama import LlavaConfig
                cfg_pretrained = LlavaConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            # check if in local there is an mm_projector.bin in ./checkpoints
            if os.path.exists('./checkpoints/mm_projector.bin'):
                mm_projector_weights = torch.load('./checkpoints/mm_projector.bin', map_location='cpu')
            else:
                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            subkeys = [part for k in mm_projector_weights.keys() for part in k.split('.')]
            if 'fga' in subkeys or 'atten' in subkeys:
                print('Loading FGA weights...')
                handle_fga(model, torch.float16, 'cuda')

            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            # NOTE: not sure why
            # vision_tower.load_model(device_map=device_map)
            vision_tower.load_model()
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    move_all_tensors_to_device(model, device)
    return tokenizer, model, image_processor, context_len
