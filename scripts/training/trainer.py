import os
import torch
import dataclasses
from typing import Any, Dict, List, Optional, Union
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    WhisperTokenizer,
    WhisperFeatureExtractor
)
import evaluate
from scripts.evaluation.normalizer import UnifiedNormalizer

@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # input_features: (batch_size, num_features, seq_length)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # labels: (batch_size, seq_length)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later by the model
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

class WhisperFineTuner:
    def __init__(self, model_id="openai/whisper-large-v3", language="Italian", task="transcribe"):
        self.processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.metric = evaluate.load("wer")
        self.normalizer = UnifiedNormalizer()
        
        # Whisper model logic for forced decoding
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # 1. Normalized WER
        pred_str_norm = [self.normalizer(s) for s in pred_str]
        label_str_norm = [self.normalizer(s) for s in label_str]
        
        # Filter for empty labels
        pred_str_norm = [s if s else " " for s in pred_str_norm]
        label_str_norm = [s if s else " " for s in label_str_norm]
        
        wer = 100 * self.metric.compute(predictions=pred_str_norm, references=label_str_norm)
        return {"wer": wer}

    def train(self, train_dataset, eval_dataset, output_dir, training_args: Dict[str, Any]):
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            **training_args
        )

        trainer = Seq2SeqTrainer(
            args=args,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        trainer.train()
        return trainer

if __name__ == "__main__":
    # Add CLI logic here or use in optuna_tuner.py
    pass
