"""
Class to wrap LLM to enable steering.
"""

import functools
import torch
from torch import Tensor
from typing import List, Callable, Optional, Union, Sequence, Dict, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
from src.dataset import DataDict
from src.activations import Activations
import time
from src.steering_utils import single_direction_hook
from src.utils import clear_memory

class SteerableModel:
    """
    Wrapper class for loading LLMs with transformer_lens HookedTransformer,
    running generations, and extracting activations.
    """

    def __init__(
        self,
        model_name: str,
        pad_token: Optional[str] = None,       # only for use_huggingface=False
        chat_template: Optional[str] = None,   # only for use_huggingface=False
        default_padding_side: str = "left", # required, but would never be "right"
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        system_prompt = None
    ):
        """
        Initialize the model and tokenizer.

        - `default_padding_side`: where padding is applied (left for decoder-only GPT-style models).
        - `pad_token`: token used for padding (defaults to tokenizer.pad_token or tokenizer.eos_token).
        - `chat_template`: optional template for chat models; falls back to tokenizer.chat_template if available.
        """

        # Notes on Models we use
        # Llama models use a system prompt by default, though it is not entirely necessary
        # OLMo models also use a system prompt by default
        # Both of above are hard coded here
        # Only OLMo supports a default pad token, the rest just use EOS token and supply attention mask
        # Everything should work fine on all of these models!

        self.device = device
        # Load model using Hugging Face transformers
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype, # no reason to not use float16 if possible
            device_map="auto" # {"": device} if device != "cpu" else None,
        )

        # Get number of layers from model config
        self.n_layers = getattr(self.model.config, 'n_layers', 
                                getattr(self.model.config, 'num_hidden_layers', 
                                        getattr(self.model.config, 'num_layers', None)))


        # Fallback for composite configs such as Gemma 3
        if self.n_layers is None:
            text_config = getattr(self.model.config, "text_config", None)
            self.n_layers = getattr(text_config, "num_hidden_layers", None)

        if self.n_layers is None:
            raise ValueError(f"Could not determine number of layers from model config: {self.model.config}")

        # Load tokenizer separately
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      
        # make sure padding works for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token # automatically sets pad_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.tokenizer.default_padding_side = default_padding_side
        self.tokenizer.padding_side = default_padding_side

        # Llama and OLMo use these system prompts by default, though they aren't trained with this necessarily
        if "llama" in model_name.lower():
            self.system_prompt = "You are a helpful, honest and concise assistant."
        elif "olmo" in model_name.lower():
            self.system_prompt = "You are OLMo 2, a helpful and harmless AI Assistant built by the Allen Institute for AI."
        else:
            self.system_prompt = system_prompt

    # def tokenize_instructions(self, instructions: List[str]) -> torch.Tensor:
    #     """
    #     Tokenize instructions using the provided chat template.
    #     """
    #     prompts = [self.chat_template.format(instruction=instruction) for instruction in instructions]
    #     toks = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids
    #     return toks.to(self.device)

    # new version using .apply_chat_template
    # EXPANDED THIS TO HAVE MORE OPTIONS
    # 11/6 - SWITCHED TO FOLLOW TAN PAPER!!! 
    def tokenize_instructions(
        self,
        instructions: List[str], # to use with no instructions, pass list of empty strings of same size as assistant response (then pass what you want to token in assistant_responses and set apply_chat_template=False)
        system_prompt: Optional[str] = None,
        assistant_responses: Optional[List[str]] = None,
        apply_chat_template: bool = True,
        # add_generation_prompt: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of user instructions.

        Args:
            instructions: List of user instructions (prompts).
            system_prompt: Optional system prompt applied to all conversations.
            assistant_responses: Optional list of assistant responses (same length as `instructions`).
            apply_chat_template: Whether to use the tokenizer's chat template.
            add_generation_prompt: Whether to append an assistant prompt for generation (if using template).

        Returns:
            (input_ids, attention_mask): tokenized tensors ready for generation.
        """
        if apply_chat_template:
            full_prompts = []
            for i, instr in enumerate(instructions):
                conversation = []

                # Add system prompt if supplied
                if self.system_prompt is not None:
                    conversation.append({"role": "system", "content": self.system_prompt})

                # Add user instruction
                conversation.append({"role": "user", "content": instr}) # this should just be prompt

                # Add initial assistant response if provided
                # if assistant_responses and i < len(assistant_responses) and assistant_responses[i] is not None:
                #     conversation.append({"role": "assistant", "content": assistant_responses[i]})

                # messages.append(conversation)

            # toks = self.tokenizer.apply_chat_template(
            #     messages,
            #     # add_generation_prompt=add_generation_prompt,
            #     tokenize=True,
            #     padding=True,
            #     truncation=False,
            #     return_tensors="pt",
            #     return_dict=True,
            # )
                input_str = self.tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                )

                if assistant_responses is not None and i < len(assistant_responses):
                    input_str = input_str + " " + assistant_responses[i]

                full_prompts.append(input_str)
                # NEW METHOD TO COPY TAN SETTING -> THIS SHOULD BE BEST METHOD
                # consistent with evals and train data

            toks = self.tokenizer(
                full_prompts, 
                padding=True,
                truncation=False,
                return_tensors="pt",
            )

        else:
            # No chat template — simple direct tokenization
            # will concatenate system prompt, instruction, and assistant response if provided
            full_prompts = []
            for i, instr in enumerate(instructions):
                parts = []
                if system_prompt:
                    parts.append(system_prompt)
                if instr != "":
                    # To be robust to empty instructions
                    # this is used for hacky solution to having data with no questions, and pos/neg answers in answer_matching_behavior and answer_not_matching_behavior
                    parts.append(instr)
                if assistant_responses and i < len(assistant_responses) and assistant_responses[i] is not None and assistant_responses[i] != "":
                    parts.append(assistant_responses[i])
                full_prompts.append("\n".join(parts))

            toks = self.tokenizer(
                full_prompts,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )

        return toks["input_ids"].to(self.device), toks["attention_mask"].to(self.device)

    # def _get_layer_module(self, layer_idx: int):
    #     """
    #     Helper function to get the appropriate layer module for hook registration.
        
    #     Args:
    #         layer_idx: The index of the layer to access
            
    #     Returns:
    #         The PyTorch module corresponding to the specified layer
    #     """
    #     if hasattr(self.model, 'model'):  # For models with .model attribute
    #         layer_module = self.model.model.layers[layer_idx]
    #     elif hasattr(self.model, 'transformer'):  # For GPT-style models
    #         layer_module = self.model.transformer.layers[layer_idx]
    #     else:  # Direct access
    #         layer_module = self.model.layers[layer_idx]
        
    #     return layer_module
    def _get_layer_module(self, layer_idx: int):
        """
        Helper function to get the appropriate layer module for hook registration.
        
        Args:
            layer_idx: The index of the layer to access
            
        Returns:
            The PyTorch module corresponding to the specified layer
        """
        # 1. Check for Gemma 3 architecture (with nested text_model)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'text_model'):
            layer_module = self.model.model.text_model.model.layers[layer_idx]
        
        # 2. Check for alternative Gemma 3 direct wrapper
        elif hasattr(self.model, 'text_model'):
            layer_module = self.model.text_model.model.layers[layer_idx]
            
        # 3. Standard Hugging Face models (Llama, Gemma 1/2, Mistral)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layer_module = self.model.model.layers[layer_idx]
            
        # 4. GPT-style models (GPT-2, Bloom)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            layer_module = self.model.transformer.layers[layer_idx]
            
        # 5. Direct access fallback
        elif hasattr(self.model, 'layers'):
            layer_module = self.model.layers[layer_idx]
            
        else:
            raise AttributeError(f"Could not find layers attribute on model type: {type(self.model)}")
        
        return layer_module


    def _register_hooks(self, fwd_hooks: List):
        """Register forward hooks given in transformer_lens format."""
        handles = []

        if not fwd_hooks or fwd_hooks == []:
            return handles

        class MockHook:
            def __init__(self, name):
                self.name = name

        def create_huggingface_hook(transformer_lens_hook_fn, hook_name_closure):
            def huggingface_hook(module, input, output):
                # transformer_lens hook expects (activation, hook) signature
                mock_hook = MockHook(hook_name_closure)
                # Apply the transformer_lens hook function
                modified_output = transformer_lens_hook_fn(output, mock_hook)
                return modified_output
            return huggingface_hook
                                
        for hook_name, hook_fn in fwd_hooks:
            # Extract layer number from hook name (e.g., "blocks.14.hook_resid_pre" -> 14)
            if "blocks." in hook_name and "hook_resid_pre" in hook_name:
                try:
                    layer_num = int(hook_name.split("blocks.")[1].split(".")[0])
                    layer_module = self._get_layer_module(layer_num)
                    hook_handle = layer_module.register_forward_hook(create_huggingface_hook(hook_fn, hook_name))
                    handles.append(hook_handle)
                except Exception as e:
                    print(f"Warning: Could not attach hook {hook_name}: {e}")

        return handles

    def create_fwd_hook_from_vec(
            self,
            vec: Tensor,
            alpha: float,
            intervention_layers: int | List[int]
    ):
        # single_direction_hook expects steering_dir and target_class format
        # hence the use of steering_dir here
        if vec is None: # for no steer case
            return []
    
        steering_dir = {"steering_vector": vec}
    
        hook_fn = functools.partial(
            single_direction_hook,
            steering_dir=steering_dir,
            target_class="steering_vector",
            alpha = alpha
        )

        if type(intervention_layers) == int:
            intervention_layers = [intervention_layers]

        fwd_hooks = [(f"blocks.{l}.hook_resid_pre", hook_fn) for l in intervention_layers]
        
        return fwd_hooks

    @torch.no_grad()
    def generate(
        self,
        questions: List[str],
        batch_size: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        # can supply fwd_hooks directly
        fwd_hooks: Optional[List] = None,
        # or can supply steering info and it will make fwd hooks
        steering_vec: Optional[Tensor] = None,
        alpha: float = 1.0,
        intervention_layers: Optional[Union[int, List[int]]] = None,
    ):
        """
        Generate text using self.model.generate with forward hooks applied.
        """
        results = []

        # Register hooks
        if fwd_hooks is None:
            try:
                fwd_hooks = self.create_fwd_hook_from_vec(
                    steering_vec, alpha, intervention_layers
                )
            except:
                # if other things not supplied, assume no fwd_hooks
                fwd_hooks = []
       
        handles = self._register_hooks(fwd_hooks)

        for i in range(0, len(questions), batch_size):
            q_batch = questions[i:i + batch_size]

            # Tokenize prompts
            input_ids, attention_mask = self.tokenize_instructions(
                instructions=q_batch,
                apply_chat_template=True,
            )

            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

            # Use Hugging Face generate
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

            # Decode only the generated portion (excluding input)
            for j, seq in enumerate(output_ids):
                # Extract only the new tokens (after the input)
                generated_tokens = seq[input_ids.shape[1]:]
                generation = self.tokenizer.decode(generated_tokens.cpu(), skip_special_tokens=True)
                
                results.append({
                    "question": q_batch[j],
                    "generation": generation
                })

            # Free memory
            del input_ids, attention_mask, output_ids
            clear_memory()

        # Remove hooks
        for h in handles:
            h.remove()

        return results

    # sloppy return_both for now
    # NOTE: raw logits make sense for single tokens, not aggregate
    #   if we do softmax, taking the log just means we can sum logprobs for entire sequence
    #   avoiding numerical issues from multiplying 
    @torch.no_grad()
    def get_binary_logit_probs(
        self,
        questions: List[str],
        pos_answers: List[str],
        neg_answers: List[str],
        fwd_hooks: List = [],
        batch_size: int = 4,
        return_logprobs: bool = True, # true to return log softmaxes, otherwise returns logits
        return_both: bool = False # whether to return both logit_probs and logits 
        # return_both is hacky and doesn't directly integrate with eval code
    ):
        """
        Compute per-token logits for positive and negative answers and calculate logit-difference.
        
        Returns:
            logit_vals: List[Dict] with keys "pos" and "neg", each value is [seq_len] tensor of logits
        """

        logit_vals = []
        if return_both: #SLOPPY
            logprob_vals = []

        num_examples = len(questions)

        # register forward hooks
        handles = self._register_hooks(fwd_hooks)
        
        for i in range(0, num_examples, batch_size):
            # batch slices
            q_batch = questions[i:i+batch_size]
            pos_batch = pos_answers[i:i+batch_size]
            neg_batch = neg_answers[i:i+batch_size]

            # concatenate question + answer (formatted such that answer is assistant answer)
            # messages_pos = [[{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            #     for q, a in zip(q_batch, pos_batch)]
            # messages_neg = [[{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            #     for q, a in zip(q_batch, neg_batch)]

            # tokenize with padding, do not truncate

            # We want to look at log probs of an actual assistnat response in a realistic setting
            # so we use chat template here to have the answer token being the assistant response
            
            # FIXED -> should work same way as batched_activations
            # so now have as system_prompt and instruction formatted + generation prompt + answer
            input_ids_pos, attention_mask_pos = self.tokenize_instructions(
                instructions=q_batch,
                assistant_responses=pos_batch,
                apply_chat_template=True,
                # add_generation_prompt=False
            )

            input_ids_neg, attention_mask_neg = self.tokenize_instructions(
                instructions=q_batch,
                assistant_responses=neg_batch,
                apply_chat_template=True,
                # add_generation_prompt=False
            )

            # forward pass
            logits_pos = self.model(input_ids_pos, attention_mask=attention_mask_pos).logits.cpu()
            logits_neg = self.model(input_ids_neg, attention_mask=attention_mask_neg).logits.cpu()

            attention_mask_pos = attention_mask_pos.cpu()
            attention_mask_neg = attention_mask_neg.cpu()  

            # shift logits to align with target tokens (predicts token i+1)
            target_ids_pos = input_ids_pos[:, 1:].cpu()
            target_ids_neg = input_ids_neg[:, 1:].cpu()

            del input_ids_pos, input_ids_neg
            clear_memory()

            shifted_logits_pos = logits_pos[:, :-1, :]
            shifted_logits_neg = logits_neg[:, :-1, :]

            # Apply log-softmax for log probabilities
            if return_both:
                shifted_logits_pos = torch.log_softmax(shifted_logits_pos, dim=-1)
                shifted_logits_neg = torch.log_softmax(shifted_logits_neg, dim=-1)

                # Gather (log-)probabilities of the actual next tokens
                token_values_pos = torch.gather(shifted_logits_pos, 2, target_ids_pos.unsqueeze(-1)).squeeze(-1)
                token_values_neg = torch.gather(shifted_logits_neg, 2, target_ids_neg.unsqueeze(-1)).squeeze(-1)

                # Mask out padding tokens
                attn_pos = attention_mask_pos[:, 1:]
                attn_neg = attention_mask_neg[:, 1:]
                token_values_pos = token_values_pos * attn_pos
                token_values_neg = token_values_neg * attn_neg

                logprob_vals.extend([
                    {"pos": p.numpy(), "neg": n.numpy()}
                    for p, n in zip(token_values_pos.unbind(0), token_values_neg.unbind(0))])
            elif return_logprobs:
                shifted_logits_pos = torch.log_softmax(shifted_logits_pos, dim=-1)
                shifted_logits_neg = torch.log_softmax(shifted_logits_neg, dim=-1)

            # Gather (log-)probabilities of the actual next tokens
            token_values_pos = torch.gather(shifted_logits_pos, 2, target_ids_pos.unsqueeze(-1)).squeeze(-1)
            token_values_neg = torch.gather(shifted_logits_neg, 2, target_ids_neg.unsqueeze(-1)).squeeze(-1)

            # Mask out padding tokens
            attn_pos = attention_mask_pos[:, 1:]
            attn_neg = attention_mask_neg[:, 1:]
            token_values_pos = token_values_pos * attn_pos
            token_values_neg = token_values_neg * attn_neg

            # Store per-example results
            logit_vals.extend([
                {"pos": p.numpy(), "neg": n.numpy()}
                for p, n in zip(token_values_pos.unbind(0), token_values_neg.unbind(0))
            ])

        # remove hooks
        for h in handles:
            h.remove()

        # SLOPPY
        if return_both:
            return logit_vals, logprob_vals

        return logit_vals

    def get_four_class_logit_probs(
        self,
        questions: List[str],
        class_answers: Dict[int, List[str]],
        fwd_hooks: List = [],
        batch_size: int = 4,
        return_logprobs: bool = True,
    ):
        """
        Compute token-level values for four possible answer classes:
        -2, -1, +1, and +2.
    
        Args:
            questions:
                List of questions.
    
            class_answers:
                Dictionary mapping each class to its candidate answers:
    
                {
                    -2: [...],
                    -1: [...],
                     1: [...],
                     2: [...],
                }
    
            return_logprobs:
                If True, return token log-probabilities.
                If False, return raw token logits.
    
        Returns:
            A list containing one dictionary per question:
    
            [
                {
                    -2: np.ndarray,
                    -1: np.ndarray,
                     1: np.ndarray,
                     2: np.ndarray,
                },
                ...
            ]
        """
        class_labels = (-2, -1, 1, 2)
        num_examples = len(questions)
    
        missing_classes = set(class_labels) - set(class_answers)
    
        if missing_classes:
            raise ValueError(
                f"Missing answer classes: {sorted(missing_classes)}"
            )
    
        for label in class_labels:
            if len(class_answers[label]) != num_examples:
                raise ValueError(
                    f"Class {label} has {len(class_answers[label])} answers, "
                    f"but there are {num_examples} questions."
                )
    
        # One result dictionary per question.
        class_values = [
            {}
            for _ in range(num_examples)
        ]
    
        handles = self._register_hooks(fwd_hooks)
    
        try:
            for start in range(0, num_examples, batch_size):
                end = min(start + batch_size, num_examples)
                question_batch = questions[start:end]
    
                for label in class_labels:
                    answer_batch = class_answers[label][start:end]
    
                    input_ids, attention_mask = self.tokenize_instructions(
                        instructions=question_batch,
                        assistant_responses=answer_batch,
                        apply_chat_template=True,
                    )
    
                    with torch.no_grad():
                        logits = self.model(
                            input_ids,
                            attention_mask=attention_mask,
                        ).logits
    
                    # Logits at position t predict the token at position t + 1.
                    shifted_logits = logits[:, :-1, :]
                    target_ids = input_ids[:, 1:]
                    shifted_attention_mask = attention_mask[:, 1:]
    
                    if return_logprobs:
                        shifted_logits = torch.log_softmax(
                            shifted_logits,
                            dim=-1,
                        )
    
                    token_values = torch.gather(
                        shifted_logits,
                        dim=2,
                        index=target_ids.unsqueeze(-1),
                    ).squeeze(-1)
    
                    # Padding positions are represented by zeros.
                    token_values = (
                        token_values * shifted_attention_mask
                    ).detach().cpu()
    
                    for batch_index, values in enumerate(
                        token_values.unbind(0)
                    ):
                        example_index = start + batch_index
    
                        class_values[example_index][label] = values.numpy()
    
                    del (
                        input_ids,
                        attention_mask,
                        logits,
                        shifted_logits,
                        target_ids,
                        shifted_attention_mask,
                        token_values,
                    )
    
                    clear_memory()
    
        finally:
            for handle in handles:
                handle.remove()
    
        return class_values

    
    def get_generations(
            self,
            instructions: List[str],
            fwd_hooks: List = [],
            max_tokens_generated: int = 64,
            batch_size: int = 4,
        ) -> List[str]:
        """
        Generate outputs for a batch of instructions.
        Applies chat template in such a way that instructions are treated as user messages (does not support multi turn 
        conversations yet)
        """
        generations = []

        for i in tqdm(range(0, len(instructions), batch_size), desc="Generating"):
            toks, attention_mask = self.tokenize_instructions(instructions[i:i+batch_size],
                                                             apply_chat_template=True),
                                                             #add_generation_prompt=True) # put on device inside tokenize_instructions

            # toks is now a list of input_ids tensors with chat template applied using .apply_chat_template!
            generation = self._generate_with_hooks(
                toks,
                max_tokens_generated=max_tokens_generated,
                fwd_hooks=fwd_hooks,
                attention_mask=attention_mask
            )
            del toks
            generations.extend(generation)

        return generations

    @torch.no_grad()
    def batched_activations(
        self,
        layers: List[int],
        token_positions: List[Union[str, int]],
        instructions: List[str], # may be list of empty strings when data set is just positive and negative examples without shared instructions
        answers: List[str] = [],
        batch_size: int = 4,
        answer_token_pos: Optional[int] = None, # used as token position if "answer_token" is in token_positions
        apply_chat_template: bool = True # whether or not to apply chat template to instructions and answers (if False, just concatenates with \n)
    ) -> torch.Tensor:
        """
        Return activations for multiple layers and token aggregation methods.
        Instructions are assumed to be one per user prompt, with chat template applied.
        * Does not support multiple shot converesations

        Args:
            layers: list of layer indices (e.g., [5, 10, 15])
            token_positions: list of token aggregation methods, where each element is either:
                - an integer index (e.g., 3)
                - -1 meaning the last token
                - "mean" meaning average over all tokens
            instructions: list of prompts (stored as "questions" in result/eval/experiment outputs)
            batch_size: number of prompts per batch

        Returns:
            activations: Tensor of shape
                (num_layers, num_token_options, num_instructions, d_model)
        """
        num_layers = len(layers)
        num_token_pos = len(token_positions)

        # store results in list of lists
        all_results = [[[] for _ in range(num_token_pos)] for _ in range(num_layers)]

        for i in tqdm(range(0, len(instructions), batch_size), desc="Computing activations"):
            instructions_batch = instructions[i:i + batch_size]
            answers_batch = answers[i:i + batch_size]
            # if apply_chat_template is True, instructions are treated as user messages and answers as assistant messages
            # otherwise, they are simply concatenated together (this is what we did originally)
            token_ids, attention_mask = self.tokenize_instructions(
                instructions_batch,
                assistant_responses=answers_batch,
                apply_chat_template=apply_chat_template,
                # add_generation_prompt=True
            )

            # run once per batch, caching all layers we need
            # For Hugging Face models, we need to use hooks to capture activations
            cache = {}
            
            def make_hook(layer_idx):
                def hook(module, input, output):
                    cache[layer_idx] = output.cpu() # save to CPU to save GPU memory
                return hook
            
            # Register hooks for the layers we need
            hooks = []
            for layer_idx in layers:
                layer_module = self._get_layer_module(layer_idx)
                hook = layer_module.register_forward_hook(make_hook(layer_idx))
                hooks.append(hook)
            
            # Run the model
            # outputs is not directly used -> but hook functions are called in the process
            outputs = self.model(token_ids,
                                    attention_mask=attention_mask)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()

            del outputs

            attention_mask = attention_mask.cpu()
            del token_ids
            clear_memory()
            
            # for each layer
            for l_idx, layer in enumerate(layers):
                acts = cache[layer].cpu()  # shape: [B, seq_len, d_model]

                clear_memory()
                
                # for each token aggregation option
                for t_idx, pos in enumerate(token_positions):
                    if pos == "mean":
                         # attention mask shape: [B, seq_len]
                        attention_mask = attention_mask.unsqueeze(-1)  # [B, seq_len, 1]
                        masked_acts = acts * attention_mask       # zero out padded tokens
                        sel = masked_acts.sum(dim=1) / attention_mask.sum(dim=1)  # [B, d_model]
                    elif pos == -1:
                        sel = acts[:, -1, :]    # last token
                        
                        # MODIFIED TO ACTUALLY GRAB LAST TOKEN
                        # find last non-padding index per sequence
                        # last_indices = attn_mask.sum(dim=1) - 1  # [B]
                        # sel = acts[torch.arange(acts.size(0)), last_indices, :]  # [B, d_model]
                    elif pos == "answer_token":
                        sel = acts[:, answer_token_pos, :]
                    elif isinstance(pos, int):
                        sel = acts[:, pos, :]   # specific token index
                    else:
                        raise ValueError(f"Unsupported token position: {pos}")

                    all_results[l_idx][t_idx].append(sel)

                del acts
            
            del cache
            clear_memory()

        # stack into final tensor
        activations = torch.stack(
            [
                torch.stack([torch.cat(all_results[l_idx][t_idx], dim=0) # [num_instructions, hidden_dim]
                            for t_idx in range(num_token_pos)], dim=0)
                for l_idx in range(num_layers)
            ],
            dim=0
        )
        
        # shape: (num_layers, num_token_positions, num_instructions, d_model)
        return activations
    
    def get_binary_activations_on_dataset(
        self,
        dataset: Dict[str, DataDict],
        layers: List[int],
        token_positions: List[Union[str, int]],
        batch_size: int = 4,
        behaviors: Optional[List[str]] = None,
        behavior_token_mapping: Optional[Dict[str, int]] = None, # if token_pos is answer_token, this grabs separate token for each behavior (since this may depend on data format)
        activations: Optional[Activations] = None,
        apply_chat_template: bool = True # whether or not to apply chat template to instructions
    ):
        """
        Get activations for dataset of forms {behavior: {pos, neg, indices}}.
        Output is an Activations instance, containing the activations of
            the positive and negative examples for each behavior.

        Args:
            dataset: TrainDataDict structure. This can be test_data attribute from a DataSet instance.
            layers: list of layer indices to extract activations from
            token_positions: list of token positions (int or str) to extract activations at
            behaviors: list of behaviors to embed; if None, embed all behaviors in dataset.train_data
            batch_size: batch size for batched_activations
            activations: if passed in, adds new activations to this Activations object

        Returns:
            Activations instance. This will contain the activations of the data, along with the raw
            data used to compute the activations and the indices of this data in the original dataset.
        """
        if activations is None:
            activations = Activations()

        # Decide which behaviors to process
        behaviors_to_process = behaviors if behaviors is not None else list(dataset.keys())

        start_time = time.time()
        for behavior in behaviors_to_process:
            print(f"Grabbing activations for behavior: {behavior}")
            if behavior not in dataset:
                raise ValueError(f"Behavior {behavior} not found in dataset.")
            else:
                behavior_dataset = dataset[behavior]

            if "answer_token" in token_positions:
                if behavior_token_mapping is None or behavior not in behavior_token_mapping:
                    raise ValueError(f"behavior_token_mapping must be provided and contain behavior {behavior} when using 'answer_token' in token_positions.")
                answer_token_pos = behavior_token_mapping[behavior]

            # dataset option 0 is originally what we did, I think dataset option 1 is more robust
            questions = behavior_dataset["questions"] # can be list of empty strings if no questions, everything else will work fine
            pos_answers = behavior_dataset["answer_matching_behavior"]
            neg_answers = behavior_dataset["answer_not_matching_behavior"]
            indices = behavior_dataset.get("indices", list(range(len(pos_answers))))

            # apply_chat_template controls if 
            pos_acts = self.batched_activations(
                layers=layers,
                token_positions=token_positions,
                instructions=questions,
                answers=pos_answers,
                batch_size=batch_size,
                # answer_token_pos is used if "answer_token" is in token_positions (but will be saved in output as answer_token)
                answer_token_pos=answer_token_pos if "answer_token" in token_positions else None,
                apply_chat_template=apply_chat_template
            )

            neg_acts = self.batched_activations(
                layers=layers,
                token_positions=token_positions,
                instructions=questions,
                answers = neg_answers,
                batch_size=batch_size,
                answer_token_pos=answer_token_pos if "answer_token" in token_positions else None,
                apply_chat_template=apply_chat_template
            )

             # Wrap in Activations object
             # Reason for passing in layers, token_positions again is so Activations object knows what these are
             #  If we didn't have this Activations wouldn't know what layers/token_positions the activations correspond to
            activations.add_behavior(
                behavior=behavior,
                layers=layers,
                token_positions=token_positions,
                pos_acts=pos_acts,
                neg_acts=neg_acts,
                # indices=indices, commented this out, might have downstream consequences
                raw_data=behavior_dataset # this will be form of indices, pos, neg
            )

            end_time = time.time()
            print(f"Time taken for {behavior}: {end_time - start_time} seconds\n")

        return activations

    def test_hook_functions(self, prompts:List[str], layer:int, hook_fn:Callable):
        """
        Test hook function.
        steeirng vectors are specified by the hook fucntions already
        """
        fwd_hooks = [(f"blocks.{layer}.hook_resid_pre", hook_fn)]
        generations = self.get_generations(prompts, fwd_hooks=fwd_hooks, max_tokens_generated=50, batch_size=2)
        return generations

    def get_answer_letters(self, 
                           questions,
                           fwd_hooks: Optional[List] = None, # can pass in forward hooks
                           steering_vec: Optional[Tensor] = None, # or can pass in steering vector, alpha, layers to make fwd hooks here
                           alpha: Optional[float] = 1.0,
                           intervention_layers: Optional[Union[int, List[int]]] = None,
                           batch_size: int = 4,
                           letters: List[str] = ['A', 'B', 'C', 'D'] # four choices for tinymmlu
                           ):
        preds = []

        # Register hooks
        if fwd_hooks is None:
            try:
                fwd_hooks = self.create_fwd_hook_from_vec(
                    steering_vec, alpha, intervention_layers
                )
            except:
                # if other things not supplied, assume no fwd_hooks
                fwd_hooks = []
       
        handles = self._register_hooks(fwd_hooks)

        letter_ids = [self.tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]

        for i in range(0, len(questions), batch_size):
            q_batch = questions[i:i+batch_size]

            input_ids, attention_mask = self.tokenize_instructions(
                instructions=q_batch,
                apply_chat_template=True,
            )

            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).logits
                logits = logits[:, -1, :] # next token logits [batch, vocab_size]
            
            logits = logits.cpu()
            del input_ids, attention_mask
            clear_memory()

            letter_logits = logits[:, letter_ids] # shape [batch, num_letters]

            pred_idx = letter_logits.argmax(dim=-1) # shape [batch] with index of predicted letter
            preds.extend(pred_idx.tolist())

        # Remove hooks
        for h in handles:
            h.remove()

        return preds