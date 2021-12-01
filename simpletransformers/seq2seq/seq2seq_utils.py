import logging
import os
import pickle
from multiprocessing import Pool
from functools import partial
from typing import Tuple
from dataclasses import dataclass
import pandas as pd
import torch
import transformers
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer
from transformers.models.bart.modeling_bart import (
    shift_tokens_right as _shift_tokens_right,
)
from torch import nn
from datasets import Features, Sequence, Value, load_dataset
from datasets import Dataset as HFDataset
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
)


logger = logging.getLogger(__name__)

if transformers.__version__ < "4.2.0":
    shift_tokens_right = (
        lambda input_ids, pad_token_id, decoder_start_token_id: _shift_tokens_right(
            input_ids, pad_token_id
        )
    )
else:
    shift_tokens_right = _shift_tokens_right


def preprocess_batch_for_hf_dataset(
    dataset, encoder_tokenizer, decoder_tokenizer, args
):
    if args.model_type == "bart":
        input_ids = encoder_tokenizer.batch_encode_plus(
            dataset["input_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )

        target_ids = encoder_tokenizer.batch_encode_plus(
            dataset["target_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )

        return {
            "source_ids": input_ids["input_ids"].squeeze(),
            "source_mask": input_ids["attention_mask"].squeeze(),
            "target_ids": target_ids["input_ids"].squeeze(),
        }
    elif args.model_type == "mbart":
        tokenized_example = encoder_tokenizer.prepare_seq2seq_batch(
            src_texts=dataset["input_text"],
            tgt_texts=dataset["target_text"],
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_length=args.max_seq_length,
            padding="max_length",  # pad_to_max_length=True won't work in this case
            return_tensors="np",
            truncation=True,
        )

        decoder_input_ids = tokenized_example["labels"].clone()
        decoder_input_ids = shift_tokens_right(
            decoder_input_ids,
            encoder_tokenizer.pad_token_id,
            encoder_tokenizer.lang_code_to_id[args.tgt_lang],
        )

        labels = tokenized_example["labels"]
        labels[labels == encoder_tokenizer.pad_token_id] = -100

        return {
            "input_ids": tokenized_example["input_ids"].squeeze(),
            "attention_mask": tokenized_example["attention_mask"].squeeze(),
            "decoder_input_ids": decoder_input_ids.squeeze(),
            "labels": labels.squeeze(),
        }
    elif args.model_type in ["rag-token", "rag-sequence"]:
        source_inputs = encoder_tokenizer(
            dataset["input_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
        try:
            target_inputs = encoder_tokenizer.generator(
                dataset["target_text"],
                max_length=args.max_seq_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
            )
        except (TypeError, ValueError) as e:
            logger.warn(e)
            logger.warn(
                """Error encountered while converting target_text.
            All target_text values have been manually cast to String as a workaround.
            This may have been caused by NaN values present in the data."""
            )
            dataset["target_text"] = [str(d) for d in dataset["target_text"]]
            target_inputs = encoder_tokenizer.generator(
                dataset["target_text"],
                max_length=args.max_seq_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
            )
        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }
    else:
        source_inputs = encoder_tokenizer(
            dataset["input_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )

        target_inputs = decoder_tokenizer(
            dataset["target_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }


def load_hf_dataset(data, encoder_tokenizer, decoder_tokenizer, args):
    if isinstance(data, str):
        dataset = load_dataset(
            "csv",
            data_files=data,
            delimiter="\t",
            download_mode="force_redownload"
            if args.reprocess_input_data
            else "reuse_dataset_if_exists",
            cache_dir=args.dataset_cache_dir,
        )
    else:
        dataset = HFDataset.from_pandas(data)

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(
            x,
            encoder_tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            args=args,
        ),
        batched=True,
    )

    if args.model_type == "bart":
        column_names = [
            "source_ids",
            "source_mask",
            "target_ids",
        ]
    elif args.model_type == "mbart":
        column_names = [
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "labels",
        ]
    else:
        column_names = [
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
        ]

    dataset.set_format(type="pt", columns=column_names)

    if isinstance(data, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset


def preprocess_data(data):
    input_text, target_text, encoder_tokenizer, decoder_tokenizer, args = data

    if args.model_type in ["rag-token", "rag-sequence"]:
        source_inputs = encoder_tokenizer(
            input_text,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        target_inputs = encoder_tokenizer.generator(
            target_text,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }
    else:
        input_text = encoder_tokenizer.encode(
            input_text,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        target_text = decoder_tokenizer.encode(
            target_text,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        return (torch.flatten(input_text), torch.flatten(target_text))


class Seq2SeqDataset(Dataset):
    def __init__(self, encoder_tokenizer, decoder_tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_name.replace("/", "_")
            + "_cached_"
            + str(args.max_seq_length)
            + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            data = [
                (input_text, target_text, encoder_tokenizer, decoder_tokenizer, args)
                for input_text, target_text in zip(
                    data["input_text"], data["target_text"]
                )
            ]

            if (mode == "train" and args.use_multiprocessing) or (
                mode == "dev" and args.use_multiprocessing_for_evaluation
            ):
                if args.multiprocessing_chunksize == -1:
                    chunksize = max(len(data) // (args.process_count * 2), 500)
                else:
                    chunksize = args.multiprocessing_chunksize

                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(preprocess_data, data, chunksize=chunksize),
                            total=len(data),
                            disable=args.silent,
                        )
                    )
            else:
                self.examples = [
                    preprocess_data(d) for d in tqdm(data, disable=args.silent)
                ]

            if not args.no_cache:
                logger.info(
                    " Saving features into cached file %s", cached_features_file
                )
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def preprocess_data_bart(data):
    input_text, target_text, tokenizer, args = data

    input_ids = tokenizer.batch_encode_plus(
        [input_text],
        max_length=args.max_seq_length,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
    )

    target_ids = tokenizer.batch_encode_plus(
        [target_text],
        max_length=args.max_seq_length,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
    )

    return {
        "source_ids": input_ids["input_ids"].squeeze(),
        "source_mask": input_ids["attention_mask"].squeeze(),
        "target_ids": target_ids["input_ids"].squeeze(),
    }


def preprocess_data_mbart(data):
    input_text, target_text, tokenizer, args = data

    tokenized_example = tokenizer.prepare_seq2seq_batch(
        src_texts=[input_text],
        tgt_texts=[target_text],
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_length=args.max_seq_length,
        padding="max_length",  # pad_to_max_length=True won't work in this case
        return_tensors="pt",
        truncation=True,
    )

    decoder_input_ids = tokenized_example["labels"].clone()
    decoder_input_ids = shift_tokens_right(
        decoder_input_ids,
        tokenizer.pad_token_id,
        tokenizer.lang_code_to_id[args.tgt_lang],
    )

    labels = tokenized_example["labels"]
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": tokenized_example["input_ids"].squeeze(),
        "attention_mask": tokenized_example["attention_mask"].squeeze(),
        "decoder_input_ids": decoder_input_ids.squeeze(),
        "labels": labels.squeeze(),
    }


class SimpleSummarizationDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        self.tokenizer = tokenizer

        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_name + "_cached_" + str(args.max_seq_length) + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            data = [
                (input_text, target_text, tokenizer, args)
                for input_text, target_text in zip(
                    data["input_text"], data["target_text"]
                )
            ]

            preprocess_fn = (
                preprocess_data_mbart
                if args.model_type == "mbart"
                else preprocess_data_bart
            )

            if (mode == "train" and args.use_multiprocessing) or (
                mode == "dev" and args.use_multiprocessing_for_evaluation
            ):
                if args.multiprocessing_chunksize == -1:
                    chunksize = max(len(data) // (args.process_count * 2), 500)
                else:
                    chunksize = args.multiprocessing_chunksize

                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(preprocess_fn, data, chunksize=chunksize),
                            total=len(data),
                            disable=args.silent,
                        )
                    )
            else:
                self.examples = [
                    preprocess_fn(d) for d in tqdm(data, disable=args.silent)
                ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def split_text(text, n=100, character=" "):
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(
    documents, split_text_n=100, split_text_character=" ", include_title=True
):
    """Split documents into passages"""
    titles, texts = [], []
    if include_title:
        for title, text in zip(documents["title"], documents["text"]):
            if text is not None:
                for passage in split_text(
                    text, n=split_text_n, character=split_text_character
                ):
                    titles.append(title if title is not None else "")
                    texts.append(passage)
    else:
        for text in documents["text"]:
            if text is not None:
                for passage in split_text(
                    text, n=split_text_n, character=split_text_character
                ):
                    titles.append("")
                    texts.append(passage)
    return {"title": titles, "text": texts}


def embed(documents, ctx_encoder, ctx_tokenizer, device):
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"],
        documents["text"],
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )["input_ids"]
    embeddings = ctx_encoder(
        input_ids.to(device=device), return_dict=True
    ).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


def generate_faiss_index_dataset(data, ctx_encoder_name, args, device):
    """
    Adapted from Huggingface example script at https://github.com/huggingface/transformers/blob/master/examples/research_projects/rag/use_own_knowledge_dataset.py
    """
    import faiss

    if isinstance(data, str):
        if args.include_title_in_knowledge_dataset:
            dataset = load_dataset(
                "csv",
                data_files=data,
                delimiter="\t",
                column_names=["title", "text"],
                cache_dir=args.dataset_cache_dir,
            )
        else:
            dataset = load_dataset(
                "csv",
                data_files=data,
                delimiter="\t",
                column_names=["text"],
                cache_dir=args.dataset_cache_dir,
            )
    else:
        dataset = HFDataset.from_pandas(data)

    dataset = dataset.map(
        partial(
            split_documents,
            split_text_n=args.split_text_n,
            split_text_character=args.split_text_character,
            include_title=args.include_title_in_knowledge_dataset,
        ),
        batched=True,
        num_proc=args.process_count,
    )

    ctx_encoder = DPRContextEncoder.from_pretrained(ctx_encoder_name).to(device=device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(ctx_encoder_name)

    new_features = Features(
        {
            "text": Value("string"),
            "title": Value("string"),
            "embeddings": Sequence(Value("float32")),
        }
    )  # optional, save as float32 instead of float64 to save space
    dataset = dataset.map(
        partial(
            embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer, device=device
        ),
        batched=True,
        batch_size=args.rag_embed_batch_size,
        features=new_features,
    )
    if isinstance(data, str):
        dataset = dataset["train"]

    if args.save_knowledge_dataset:
        output_dataset_directory = os.path.join(args.output_dir, "knowledge_dataset")
        os.makedirs(output_dataset_directory, exist_ok=True)
        dataset.save_to_disk(output_dataset_directory)

    index = faiss.IndexHNSWFlat(args.faiss_d, args.faiss_m, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)

    return dataset


def add_faiss_index_to_dataset(dataset):
    import faiss

    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)

    return dataset



@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

import torch.nn.functional as F
import torch
from torch import nn
from torch import Tensor



class EISLNatCriterion:
    def __init__(self, label_smoothing=0.1, ngram='1,2', ce_factor=0.5, ngram_factor=0.5):
        self.label_smoothing = LabelSmoother(label_smoothing)
        self.ce_factor = ce_factor
        self.ngram_factor = ngram_factor

        self.ngram = [int(n) for n in ngram.split(',')]


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )
        parser.add_argument(
            "--ngram",
            default=None,
            type=str,
            help="the ngram to consider, comma separated, e.g. \"--ngram 2,3,4,-1\" (0 means output_length, -1 means output_length-1)",
        )
        parser.add_argument(
            "--ce-factor",
            required=True,
            type=float,
            help="blend factor for cross entropy",
        )
        parser.add_argument(
            "--ngram-factor",
            required=True,
            type=float,
            help="blend factor for ngram loss",
        )


    def _compute_loss(
            self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        # def mean_ds(x: Tensor, dim=None) -> Tensor:
        #     return (
        #         x.float().mean().type_as(x)
        #         if dim is None
        #         else x.float().mean(dim).type_as(x)
        #     )

        # if masks is not None:
        #     outputs, targets = outputs[masks], targets[masks]

        # if masks is not None and not masks.any():
        #     nll_loss = torch.tensor(0)
        #     loss = nll_loss
        # else:
        #     logits = F.log_softmax(outputs, dim=-1)
        #     if targets.dim() == 1:
        #         losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

        #     else:  # soft-labels
        #         losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
        #         losses = losses.sum(-1)

        #     nll_loss = mean_ds(losses)
        #     if label_smoothing > 0:
        #         loss = (
        #                 nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
        #         )
        #     else:
        #         loss = nll_loss
        loss = self.label_smoothing(outputs, targets)
        loss = loss * factor
        return {"name": name, "loss": loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def config_ngram_list(self, output_length):
        ngram_list = set()
        for n in self.ngram:
            if n>0:
                if n<=output_length:
                    ngram_list.add(n)
            else:
                real_n = output_length+n
                if 0 <real_n:
                    ngram_list.add(real_n)
        if ngram_list:
            ngram_list = list(ngram_list)
        else:
            ngram_list = [output_length]

        return ngram_list

    def compute_EISL(self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0):
        
        ce_loss = self._compute_loss(
            outputs=outputs,
            targets=targets,
            masks=masks,
            label_smoothing=label_smoothing,
            name=name,
            factor=factor
        )
        outputs = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        log_probs = F.log_softmax(outputs, dim=-1)

        ngram_list = self.config_ngram_list(output_length=outputs.size(1))
        ngram_loss = self.batch_log_EISL_cnn(log_probs, targets, ngram_list=ngram_list)

        eisl_loss = ngram_loss * self.ngram_factor + ce_loss['loss'] * self.ce_factor

        return {"name": 'EISL-loss', "loss": eisl_loss,
                "ngram_loss": ngram_loss,
                "ce_loss": ce_loss['loss'],
                "factor": 1.0}

    def batch_log_EISL_cnn(self, decoder_outputs, target_idx, ngram_list, pad=-100,
                              weight_list=None):
        """
        decoder_outputs: [batch_size, output_len, vocab_size]
            - matrix with probabilityes  -- log probs
        target_variable: [batch_size, target_len]
            - reference batch
        ngram_list: int or List[int]
            - n-gram to consider
        pad: int
            the idx of "pad" token
        weight_list : List
            corresponding weight of ngram
        NOTE: output_len == target_len
        """

        batch_size, output_len, vocab_size = decoder_outputs.size()
        _, tgt_len = target_idx.size()

        if type(ngram_list) == int:
            ngram_list = [ngram_list]
        if ngram_list[0] <= 0:
            ngram_list[0] = output_len
        if weight_list is None:
            weight_list = [1. / len(ngram_list)] * len(ngram_list)

        decoder_outputs = torch.relu(decoder_outputs + 20) - 20  # 过滤掉过小的概率  logp = -20 ---> p = 2e-9

        # [batch_size, output_len, target_len]
        index = target_idx.unsqueeze(1).expand(-1, output_len, tgt_len)
        ignore = index.eq(pad)
        index[ignore] =0

        # [batch, output_len, target_len]
        cost_nll = decoder_outputs.gather(dim=2, index=index)

        # [batch, 1, output_len, target_len]
        cost_nll = cost_nll.unsqueeze(1)

        sum_gram = torch.empty((1),device="cuda")

        for cnt, ngram in enumerate(ngram_list):
            # out: [batch, 1, output_len, target_len]
            # eye_filter: [1, 1, ngram, ngram]
            eye_filter = torch.eye(ngram,device="cuda").view([1, 1, ngram, ngram])

            assert ngram <= decoder_outputs.size()[1]
            # term: [batch, 1, output_len - ngram + 1, target_len - ngram + 1]
            term = nn.functional.conv2d(cost_nll, eye_filter) / ngram

            # maybe dim should be 2, but sometime 1 is better
            gum_tmp = F.gumbel_softmax(term.squeeze_(1), tau=1, dim=1)

            term = term.mul(gum_tmp).sum(1).mean(1)

            sum_gram += weight_list[cnt] * term.sum()
        loss = - sum_gram / batch_size
        return loss

import numpy as np
import torch.nn as nn


class LossDropper(nn.Module):
    def __init__(
            self,
            dropc=0.4,
            min_count=10000,
            recompute=10000,
            verbose=True
    ):
        super().__init__()
        self.keepc = 1. - dropc
        self.count = 0
        self.min_count = min_count

        self.recompute = recompute
        self.last_computed = 0
        self.percentile_val = 100000000.
        self.cur_idx = 0

        self.verbose = verbose

        self.vals = np.zeros(self.recompute, dtype=np.float32)

    def forward(self, loss):
        if loss is None:
            return loss

        self.last_computed += loss.numel()
        self.count += loss.numel()
        if self.count < len(self.vals):
            self.vals[self.count - loss.numel():self.count] = loss.detach().cpu().numpy().flatten()
            self.cur_idx += loss.numel()
            return (loss < np.inf).type(loss.dtype)
        else:
            for idx, item in enumerate(loss):
                self.vals[self.cur_idx] = item
                self.cur_idx += 1
                if self.cur_idx >= len(self.vals):
                    self.cur_idx = 0
        if self.count < self.min_count:
            return (loss < np.inf).type(loss.dtype)

        if self.last_computed > self.recompute:
            self.percentile_val = np.percentile(self.vals, self.keepc * 100)
            if self.verbose:
                print('Using cutoff', self.percentile_val)
            self.last_computed = 0

        mask = (loss < self.percentile_val).type(loss.dtype)
        return mask


@dataclass
class UnlikelihoodLoss:
    def __init__(self, neg_tokn_id, pos_token_id, unlikelihood_loss_alpha_rank) -> None:
        self.neg_tokn_id = neg_tokn_id
        self.pos_token_id = pos_token_id
        self.alpha_rank = unlikelihood_loss_alpha_rank

    def __call__(self, model, inputs, model_output):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        labels = inputs.get("labels")
        sentence_labels = torch.tensor([label[0] for label in labels]).to(labels.device)
        # labels[:,0] = 0
        # labels = torch.roll(labels, -1, dims=1)
        labels[:,0] = -100
        # negatives = labels
        # negatives[torch.isin(labels, inputs.get("input_ids"))] = -100
        negatives = labels[sentence_labels == self.neg_tokn_id]
        positives = labels[sentence_labels == self.pos_token_id]

        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        negatives[negatives == -100] = 0
        negative_targets = torch.zeros_like(lprobs).scatter_(2, torch.unsqueeze(negatives, 2), 1)
        negative_targets[:, :, 0] = 0
        one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)
        custom_loss = -torch.log(one_minus_probs)*negative_targets
        neg_loss = custom_loss.sum()
        
        pos_inputs = {k:t[sentence_labels == self.pos_token_id] for k,t in inputs.items()}
        if positives.shape[0] > 0:
            pos_outputs = model(**pos_inputs)
            pos_loss = pos_outputs["loss"]
            del pos_outputs
        else:
            pos_loss = 0
        del sentence_labels, negatives, positives, labels, pos_inputs
        return pos_loss + self.alpha_rank * neg_loss
