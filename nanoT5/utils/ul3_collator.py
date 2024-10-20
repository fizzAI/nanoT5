from typing import Dict, List, Optional, Union
import torch
from torch import Tensor
from transformers import BatchEncoding, AutoTokenizer, DataCollator
import numpy as np
from dataclasses import dataclass



def multi_forward(
        model,
    input_ids_mlm=None,
    attention_mask_mlm=None,
    labels_mlm=None,
    input_ids_ntp=None,
    attention_mask_ntp=None,
    labels_ntp=None,
    calc_acc=False,
    **kwargs
):
    loss = 0.0
    outputs = {}

    # MLM Objective
    if input_ids_mlm is not None and labels_mlm is not None:
        outputs_mlm = model.forward(
            input_ids=input_ids_mlm,
            attention_mask=attention_mask_mlm,
            labels=labels_mlm,
            **kwargs
        )
        loss += outputs_mlm.loss
        # outputs['loss_mlm'] = outputs_mlm.loss
        # outputs['logits_mlm'] = outputs_mlm.logits

    # NTP Objective
    if input_ids_ntp is not None and labels_ntp is not None:
        outputs_ntp = model.forward(
            input_ids=input_ids_ntp,
            attention_mask=attention_mask_ntp,
            labels=labels_ntp,
            **kwargs
        )
        loss += outputs_ntp.loss
        # outputs['loss_ntp'] = outputs_ntp.loss
        # outputs['logits_ntp'] = outputs_ntp.logits

    # outputs['loss'] = loss

    stats = {}
    stats["loss"] = loss.detach().float().item()
    stats['loss_ntp'] = outputs_ntp.loss.detach().float().item()
    stats['loss_mlm'] = outputs_mlm.loss.detach().float().item()


    if calc_acc:
        if input_ids_mlm is not None and labels_mlm is not None:
            correct_mlm = (outputs_mlm.logits.argmax(-1) == labels_mlm).sum().item()
            accuracy_mlm = correct_mlm / labels_mlm.numel()
            stats["accuracy_mlm"] = accuracy_mlm

        if input_ids_ntp is not None and labels_ntp is not None:
            correct_ntp = (outputs_ntp.logits.argmax(-1) == labels_ntp).sum().item()
            accuracy_ntp = correct_ntp / labels_ntp.numel()
            stats["accuracy_ntp"] = accuracy_ntp

        stats["accuracy"] = (correct_mlm + correct_ntp) / (labels_mlm.numel() + labels_ntp.numel())

    return loss, stats


from typing import Dict, List, Optional, Union
import numpy as np
from transformers import BatchEncoding, AutoTokenizer
from dataclasses import dataclass


from typing import Dict, List, Optional, Union
import numpy as np
import torch  # Import torch to convert numpy arrays to tensors
from transformers import BatchEncoding, AutoTokenizer
from dataclasses import dataclass


@dataclass
class DataCollatorForT5UL3:
    """
    Data collator for T5 UL2-style training with multiple denoising objectives, including MLM and NTP.

    This collator prepares batches by applying denoising spans, creating sentinel tokens,
    handling padding and masking, and incorporating an additional objective for next token prediction.

    Args:
        tokenizer (AutoTokenizer): Tokenizer used for encoding the data.
        max_length (int): Maximum length of input sequences.
        max_labels_length (int): Maximum length of label sequences.
        batch_size (int): Number of samples per batch.
        denoiser_list (List[Dict]): List of denoiser configurations, each containing:
            - "mu": Mean noise span length.
            - "r": Noise density.
            - "max_spans": Maximum number of spans.
            - "prefix": Prefix string to prepend to inputs.
        denoiser_proportions (List[float]): Probabilities for selecting each denoiser.
        causal (bool, optional): Whether to use causal masking. Defaults to False.
        random_chunk (bool, optional): Whether to use random chunking of inputs. Defaults to True.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        max_labels_length: int,
        batch_size: int,
        denoiser_list: List[Dict],
        denoiser_proportions: List[float],
        causal: bool = False,
        random_chunk: bool = True,
    ):
        super().__init__()

        # Normalize denoiser proportions to sum to 1
        total = sum(denoiser_proportions)
        if not np.isclose(total, 1.0):
            denoiser_proportions = [x / total for x in denoiser_proportions]
        self.denoiser_proportions = denoiser_proportions
        self.denoiser_list = denoiser_list  # List of denoiser configurations

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_labels_length = max_labels_length
        self.causal = causal
        self.random_chunk = random_chunk

        # Precompute prefix token IDs for each denoiser
        self.prefixes = [
            np.array(tokenizer.encode(denoiser["prefix"], add_special_tokens=False), dtype=int)
            for denoiser in denoiser_list
        ]

        # Identify extra special token IDs (assuming they are contiguous)
        self.extra_ids = sorted(
            [tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>") for i in range(100)],
            reverse=True
        )

        # Compute optimal input and target lengths for each denoiser
        max_prefix_len = max(prefix.shape[0] for prefix in self.prefixes)
        self.denoiser_optimal_len = []
        for denoiser in denoiser_list:
            if denoiser['mu'] == 0 and denoiser['r'] == 0:
                # Special case for NTP (no noise applied)
                self.denoiser_optimal_len.append((max_length - max_prefix_len, max_length - max_prefix_len))
            else:
                if denoiser['mu'] == 0:
                    raise ValueError(f"Invalid mean_noise_span_length (mu) of 0 in denoiser configuration: {denoiser}")
                self.denoiser_optimal_len.append(
                    self.compute_input_and_target_lengths(
                        max_length - max_prefix_len,
                        denoiser["r"],
                        denoiser["mu"]
                    )
                )

    def is_special_token(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Check if the token IDs correspond to special tokens.

        Args:
            token_ids (np.ndarray): Array of token IDs.

        Returns:
            np.ndarray: Boolean array indicating special tokens.
        """
        # Vectorized check for special tokens within the range of extra_ids
        return (token_ids <= self.extra_ids[0]) & (token_ids >= self.extra_ids[-1])

    def _best_fit(
        self,
        input_ids: List[np.ndarray],
        labels: List[np.ndarray]
    ) -> (List[np.ndarray], List[np.ndarray]):
        """
        Packs input_ids and labels into batches respecting max_length and max_labels_length.

        Args:
            input_ids (List[np.ndarray]): List of input ID arrays.
            labels (List[np.ndarray]): List of label ID arrays.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Packed input and label arrays.
        """
        batch_inputs = []
        batch_labels = []
        remaining_indices = list(range(len(input_ids)))

        while remaining_indices and len(batch_inputs) < self.batch_size:
            current_input = []
            current_label = []
            current_input_length = 0
            current_label_length = 0
            current_special_tokens = 0

            for idx in remaining_indices.copy():
                input_len = input_ids[idx].shape[0]
                label_len = labels[idx].shape[0]
                num_special = self.is_special_token(input_ids[idx]).sum()

                if (
                    current_input_length + input_len <= self.max_length and
                    current_label_length + label_len <= self.max_labels_length and
                    current_special_tokens + num_special <= len(self.extra_ids)
                ):
                    current_input.append(input_ids[idx])
                    current_label.append(labels[idx])
                    current_input_length += input_len
                    current_label_length += label_len
                    current_special_tokens += num_special
                    remaining_indices.remove(idx)

            if current_input:
                batch_inputs.append(np.concatenate(current_input))
                batch_labels.append(np.concatenate(current_label))

        return batch_inputs, batch_labels

    def __call__(self, examples: List[Dict[str, Union[np.ndarray, List[int]]]]) -> BatchEncoding:
        """
        Collate a batch of examples into arrays suitable for T5 model input.

        Args:
            examples (List[Dict[str, Union[np.ndarray, List[int]]]]): List of examples, each containing 'input_ids' and optionally 'length'.

        Returns:
            BatchEncoding: Dictionary containing batched input_ids, attention_mask, and labels for MLM and NTP.
        """
        input_batch_size = len(examples)

        # Sample denoiser indices based on proportions
        denoisers_sample = np.random.choice(
            len(self.denoiser_list),
            size=input_batch_size,
            p=self.denoiser_proportions
        )

        # Ensure input_ids are integer numpy arrays and calculate length if not provided
        for example in examples:
            if isinstance(example["input_ids"], np.ndarray):
                example["input_ids"] = example["input_ids"].astype(int)
            else:
                example["input_ids"] = np.array(example["input_ids"], dtype=int)
            if "length" not in example:
                example["length"] = example["input_ids"].shape[0]

        # Truncate examples based on optimal lengths
        truncated_examples = []
        for example, denoiser_idx in zip(examples, denoisers_sample):
            max_len = self.denoiser_optimal_len[denoiser_idx][0]
            if example["length"] > max_len:
                start = 0
                if self.random_chunk:
                    start = np.random.randint(0, example["length"] - max_len + 1)
                new_input_ids = example["input_ids"][start:start + max_len]
                truncated_examples.append({
                    "input_ids": new_input_ids,
                    "length": max_len
                })
            else:
                truncated_examples.append(example)

        # Generate noise masks for each example
        spans_noise_masks = []
        for i, example in enumerate(truncated_examples):
            denoiser = self.denoiser_list[denoisers_sample[i]]
            if denoiser['r'] == 0 and denoiser['mu'] == 0:
                # For NTP, we don't apply any noise mask
                spans_noise_masks.append(np.zeros(example['length'], dtype=bool))
            else:
                spans_noise_masks.append(
                    self.random_spans_noise_mask(example["length"], denoiser)
                )

        # Create sentinel IDs for inputs and labels
        input_ids_sentinel = []
        labels_sentinel = []
        for mask in spans_noise_masks:
            if mask.any():
                input_ids_sentinel.append(self.create_sentinel_ids(mask))
                labels_sentinel.append(self.create_sentinel_ids(~mask))
            else:
                input_ids_sentinel.append(np.zeros_like(mask, dtype=int))
                labels_sentinel.append(np.zeros_like(mask, dtype=int))

        # Apply sentinel masks and prefixes
        input_ids = [
            self.filter_input_ids(
                example["input_ids"],
                sentinel_ids,
                prefixes=self.prefixes[denoisers_sample[i]]
            )
            for i, (example, sentinel_ids) in enumerate(zip(truncated_examples, input_ids_sentinel))
        ]
        labels = [
            self.filter_input_ids(
                example["input_ids"],
                sentinel_ids,
                with_eos=False
            )
            for example, sentinel_ids in zip(truncated_examples, labels_sentinel)
        ]

        # If batch size matches, use directly; else, apply best fit
        if len(input_ids) == self.batch_size:
            batch_inputs, batch_labels = input_ids, labels
        else:
            batch_inputs, batch_labels = self._best_fit(input_ids, labels)

        # Replace special tokens in labels and inputs
        for idx, label in enumerate(batch_labels):
            is_special = self.is_special_token(label)
            cumsum_special = np.cumsum(is_special)
            batch_labels[idx] = np.where(
                is_special,
                self.extra_ids[0] - cumsum_special + 1,
                label
            )

        for idx, input_id in enumerate(batch_inputs):
            is_special = self.is_special_token(input_id)
            cumsum_special = np.cumsum(is_special)
            batch_inputs[idx] = np.where(
                is_special,
                self.extra_ids[0] - cumsum_special + 1,
                input_id
            )

        # Append EOS token to labels
        eos_token = np.array([self.tokenizer.eos_token_id], dtype=int)
        labels = [np.concatenate([label, eos_token]) for label in batch_labels]

        # Pad sequences
        if self.causal:
            # Left pad inputs and right pad labels
            input_ids_padded = np.stack([
                np.concatenate([
                    np.full((self.max_length - input_id.shape[0],), self.tokenizer.pad_token_id, dtype=int),
                    input_id
                ]) if input_id.shape[0] < self.max_length else input_id[:self.max_length]
                for input_id in batch_inputs
            ])
            labels_padded = np.stack([
                np.concatenate([
                    label,
                    np.full((self.max_labels_length - label.shape[0],), self.tokenizer.pad_token_id, dtype=int)
                ]) if label.shape[0] < self.max_labels_length else label[:self.max_labels_length]
                for label in labels
            ])
        else:
            # Right pad inputs and labels
            input_ids_padded = np.stack([
                np.concatenate([
                    input_id,
                    np.full((self.max_length - input_id.shape[0],), self.tokenizer.pad_token_id, dtype=int)
                ]) if input_id.shape[0] < self.max_length else input_id[:self.max_length]
                for input_id in batch_inputs
            ])
            labels_padded = np.stack([
                np.concatenate([
                    label,
                    np.full((self.max_labels_length - label.shape[0],), self.tokenizer.pad_token_id, dtype=int)
                ]) if label.shape[0] < self.max_labels_length else label[:self.max_labels_length]
                for label in labels
            ])

        # Prepare the final batch for MLM
        batch = {
            "input_ids_mlm": input_ids_padded,
            "attention_mask_mlm": (input_ids_padded != self.tokenizer.pad_token_id).astype(int),
            "labels_mlm": np.where(labels_padded == self.tokenizer.pad_token_id, -100, labels_padded)
        }

        if self.causal:
            # For causal models, concatenate input_ids and labels
            batch["input_ids_mlm"] = np.concatenate([input_ids_padded, labels_padded], axis=-1)
            batch["attention_mask_mlm"] = (batch["input_ids_mlm"] != self.tokenizer.pad_token_id).astype(int)
            batch["labels_mlm"] = np.where(batch["input_ids_mlm"] == self.tokenizer.pad_token_id, -100, batch["input_ids_mlm"])

        # **Adding Next Token Prediction (NTP) Objective**
        # For NTP, we'll treat it as a separate task where the model predicts the next token in the sequence
        # We'll prepare separate input_ids and labels for NTP

        # Shift input_ids for NTP: labels_ntp are the next tokens
        input_ids_ntp = [example["input_ids"] for example in truncated_examples]
        labels_ntp = [input_id[1:].copy() for input_id in input_ids_ntp]
        input_ids_ntp = [input_id[:-1].copy() for input_id in input_ids_ntp]

        # Pad NTP sequences
        if self.causal:
            # Left pad NTP inputs and right pad NTP labels
            input_ids_ntp_padded = np.stack([
                np.concatenate([
                    np.full((self.max_length - input_id.shape[0],), self.tokenizer.pad_token_id, dtype=int),
                    input_id
                ]) if input_id.shape[0] < self.max_length else input_id[:self.max_length]
                for input_id in input_ids_ntp
            ])
            labels_ntp_padded = np.stack([
                np.concatenate([
                    label,
                    np.full((self.max_labels_length - label.shape[0],), self.tokenizer.pad_token_id, dtype=int)
                ]) if label.shape[0] < self.max_labels_length else label[:self.max_labels_length]
                for label in labels_ntp
            ])
        else:
            # Right pad NTP inputs and labels
            input_ids_ntp_padded = np.stack([
                np.concatenate([
                    input_id,
                    np.full((self.max_length - input_id.shape[0],), self.tokenizer.pad_token_id, dtype=int)
                ]) if input_id.shape[0] < self.max_length else input_id[:self.max_length]
                for input_id in input_ids_ntp
            ])
            labels_ntp_padded = np.stack([
                np.concatenate([
                    label,
                    np.full((self.max_labels_length - label.shape[0],), self.tokenizer.pad_token_id, dtype=int)
                ]) if label.shape[0] < self.max_labels_length else label[:self.max_labels_length]
                for label in labels_ntp
            ])

        # Prepare the final batch for NTP
        batch.update({
            "input_ids_ntp": input_ids_ntp_padded,
            "attention_mask_ntp": (input_ids_ntp_padded != self.tokenizer.pad_token_id).astype(int),
            "labels_ntp": np.where(labels_ntp_padded == self.tokenizer.pad_token_id, -100, labels_ntp_padded)
        })

        if self.causal:
            # For causal models, concatenate input_ids and labels for NTP
            batch["input_ids_ntp"] = np.concatenate([input_ids_ntp_padded, labels_ntp_padded], axis=-1)
            batch["attention_mask_ntp"] = (batch["input_ids_ntp"] != self.tokenizer.pad_token_id).astype(int)
            batch["labels_ntp"] = np.where(batch["input_ids_ntp"] == self.tokenizer.pad_token_id, -100, batch["input_ids_ntp"])

        # **Convert numpy arrays back to torch tensors**
        for key in batch:
            batch[key] = torch.tensor(batch[key], dtype=torch.long)

        return BatchEncoding(batch)

    def compute_input_and_target_lengths(
            self,
            inputs_length: int,
            noise_density: float,
            mean_noise_span_length: float
        ) -> (int, int):
            """
            Compute the optimal input and target lengths based on noise density and span length.

            Args:
                inputs_length (int): Desired length of the tokenized inputs sequence.
                noise_density (float): Density of noise in the sequence.
                mean_noise_span_length (float): Mean length of noise spans.

            Returns:
                Tuple[int, int]: Computed input and target lengths.
            """

            def _tokens_length_to_lengths(tokens_length: int) -> (int, int):
                num_noise_tokens = int(round(tokens_length * noise_density))
                num_noise_spans = max(1, int(round(num_noise_tokens / mean_noise_span_length)))
                num_nonnoise_tokens = tokens_length - num_noise_tokens

                # Inputs: non-noise tokens + sentinel tokens + EOS
                input_length = num_nonnoise_tokens + num_noise_spans + 1
                # Targets: noise tokens + sentinel tokens + EOS
                target_length = num_noise_tokens + num_noise_spans + 1

                return input_length, target_length

            tokens_length = inputs_length

            if noise_density == 0.0 and mean_noise_span_length == 0.0:
                # For NTP, input and target lengths are the same
                return inputs_length, inputs_length

            if noise_density == 0.0:
                # For NTP, we set input and target lengths accordingly
                if mean_noise_span_length == 0:
                    return (self.max_labels_length - 2, inputs_length)
                else:
                    return (self.max_labels_length - 2 + int(self.max_length // mean_noise_span_length) - 2, inputs_length)

            # Increment tokens_length until input_length <= inputs_length
            while _tokens_length_to_lengths(tokens_length + 1)[0] <= inputs_length:
                tokens_length += 1

            input_length, target_length = _tokens_length_to_lengths(tokens_length)

            # Adjust for specific noise density cases
            if noise_density == 0.5 and target_length > input_length:
                tokens_length -= 1
                input_length, target_length = _tokens_length_to_lengths(tokens_length)

            return tokens_length, target_length

    def random_spans_noise_mask(self, sequence_length: int, denoiser_params: Dict) -> np.ndarray:
        """
        Generate a noise mask with random spans based on denoiser parameters.

        Args:
            sequence_length (int): Length of the token sequence.
            denoiser_params (Dict): Parameters for denoising, including:
                - "mu": Mean noise span length.
                - "r": Noise density.
                - "max_spans": Maximum number of spans.

        Returns:
            np.ndarray: Boolean array indicating noise positions.
        """
        noise_density = denoiser_params["r"]
        mean_noise_span_length = denoiser_params["mu"]
        max_num_spans = denoiser_params.get("max_spans", 1000)  # Default to a large number if not provided

        if noise_density == 0.0 and mean_noise_span_length == 0.0:
            # For NTP, return a mask of all False (no noise)
            return np.zeros(sequence_length, dtype=bool)

        if max_num_spans == 1:
            # Force the span to start at the beginning
            prefix_span = int(round(sequence_length / mean_noise_span_length))
            masked_span = sequence_length - prefix_span
            interleaved_span_lengths = np.array([prefix_span, masked_span], dtype=int)
        else:
            num_noise_tokens = int(round(sequence_length * noise_density))
            num_noise_tokens = max(1, min(num_noise_tokens, sequence_length - 1))
            num_noise_spans = min(max_num_spans, int(round(num_noise_tokens / mean_noise_span_length)))
            num_noise_spans = max(1, num_noise_spans)
            num_nonnoise_tokens = sequence_length - num_noise_tokens

            noise_span_lengths = self._random_segmentation(num_noise_tokens, num_noise_spans)
            nonnoise_span_lengths = self._random_segmentation(num_nonnoise_tokens, num_noise_spans)

            interleaved_span_lengths = np.vstack((nonnoise_span_lengths, noise_span_lengths)).T.flatten()

        # Generate span start indicators
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros(sequence_length, dtype=bool)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = (span_num % 2 == 1)

        return is_noise

    def _random_segmentation(self, num_items: int, num_segments: int) -> np.ndarray:
        """
        Randomly partition a sequence into non-empty segments.

        Args:
            num_items (int): Number of items to partition.
            num_segments (int): Number of segments.

        Returns:
            np.ndarray: Array containing the length of each segment.
        """
        if num_segments == 1:
            return np.array([num_items], dtype=int)

        # Randomly choose split points
        split_points = np.random.choice(range(1, num_items), num_segments - 1, replace=False)
        split_points = np.sort(split_points)
        split_points = np.concatenate(([0], split_points, [num_items]))
        segment_lengths = split_points[1:] - split_points[:-1]
        return segment_lengths

    def create_sentinel_ids(self, mask_indices: np.ndarray) -> np.ndarray:
        """
        Create sentinel IDs based on mask indices.

        Args:
            mask_indices (np.ndarray): Boolean array indicating masked positions.

        Returns:
            np.ndarray: Array with sentinel IDs.
        """
        # Identify start indices of masked spans
        start_indices = mask_indices & (~np.roll(mask_indices, 1))
        start_indices[0] = mask_indices[0]

        # Assign sentinel IDs
        sentinel_ids = np.where(start_indices, np.cumsum(start_indices), 0)
        # Assign unique sentinel IDs in decreasing order
        max_sentinels = sentinel_ids.max()
        if max_sentinels > len(self.extra_ids):
            raise ValueError(f"Number of sentinels ({max_sentinels}) exceeds available extra_ids ({len(self.extra_ids)}).")
        sentinel_id_values = np.array([self.extra_ids[0] - i for i in range(1, max_sentinels + 1)])
        sentinel_ids = np.where(
            sentinel_ids != 0,
            sentinel_id_values[sentinel_ids - 1],
            0
        )
        sentinel_ids = sentinel_ids - (mask_indices.astype(int) - start_indices.astype(int))

        return sentinel_ids

    def filter_input_ids(
        self,
        input_ids: Union[np.ndarray, List[int]],
        sentinel_ids: np.ndarray,
        prefixes: Optional[np.ndarray] = None,
        with_eos: bool = True
    ) -> np.ndarray:
        """
        Apply sentinel masks and optionally prepend prefixes and append EOS tokens.

        Args:
            input_ids (Union[np.ndarray, List[int]]): Original input IDs.
            sentinel_ids (np.ndarray): Sentinel IDs to replace masked tokens.
            prefixes (Optional[np.ndarray], optional): Prefix array to prepend. Defaults to None.
            with_eos (bool, optional): Whether to append EOS token. Defaults to True.

        Returns:
            np.ndarray: Filtered input IDs.
        """
        # Convert input_ids to array if it's a list
        if isinstance(input_ids, list):
            input_ids = np.array(input_ids, dtype=int)

        # Replace masked tokens with sentinel IDs
        input_ids = np.where(sentinel_ids != 0, sentinel_ids, input_ids)

        # Remove tokens after EOS
        eos_mask = (input_ids == self.tokenizer.eos_token_id)
        if eos_mask.any():
            first_eos = np.argmax(eos_mask)
            input_ids = input_ids[:first_eos + 1]

        # Remove negative tokens (masked tokens after sentinel tokens)
        input_ids = input_ids[input_ids >= 0]

        if prefixes is not None and prefixes.size > 0:
            input_ids = np.concatenate([prefixes, input_ids], axis=0)

        if with_eos and (input_ids[-1] != self.tokenizer.eos_token_id):
            input_ids = np.concatenate([input_ids, np.array([self.tokenizer.eos_token_id], dtype=int)])

        return input_ids


