import os
import torch
import evaluate
from datasets import load_dataset, Audio, Value
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    pipeline,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
common_voice_dir = os.path.join(script_dir, "mozilla", "ne")
audio_base_path = os.path.join(common_voice_dir, "clips")
model_name = "openai/whisper-base" # Changed to 'base' for memory efficiency

# --- 1. Load Pre-trained Whisper Model and Processor ---
processor = WhisperProcessor.from_pretrained(model_name, language="ne", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU (CUDA or MPS) if available
if torch.cuda.is_available():
    model = model.to("cuda")
    print("CUDA is available. Model moved to GPU.")
elif torch.backends.mps.is_available():
    model = model.to("mps")
    print("MPS is available. Model moved to MPS device for Apple Silicon acceleration.")
else:
    print("No GPU acceleration (CUDA/MPS) available. Model will run on CPU.")

# --- 2. Load Common Voice Dataset from Local TSV files ---
train_tsv_path = os.path.join(common_voice_dir, "train.tsv")
dev_tsv_path = os.path.join(common_voice_dir, "dev.tsv")

data_files = {
    "train": train_tsv_path,
    "validation": dev_tsv_path,
}

try:
    dataset = load_dataset(
        "csv",
        data_files=data_files,
        delimiter="\t",
        column_names=["client_id", "path", "sentence_id", "sentence", "sentence_domain", "up_votes", "down_votes", "age", "gender", "accents", "variant", "locale", "segment"],
        skiprows=1,
        cache_dir=None # ADDED: Disable caching for local files, crucial for Colab/Drive
    )
except FileNotFoundError as e:
    print(f"Error: Could not find Common Voice TSV files. Details: {e}")
    exit()

# --- Data Preprocessing: Filtering and Column Management ---
dataset = dataset.cast_column("sentence", Value("string"))

for split in ["train", "validation"]:
    dataset[split] = dataset[split].filter(
        lambda x: x["sentence"] is not None and len(x["sentence"].strip()) > 0,
        load_from_cache_file=False,
        num_proc=1
    )
    dataset[split] = dataset[split].filter(
        lambda x: x["path"] is not None and len(str(x["path"]).strip()) > 0,
        load_from_cache_file=False,
        num_proc=1
    )

def create_full_audio_path_column_batched(batch, audio_dir):
    return {"audio_path": [os.path.join(audio_dir, str(p)) for p in batch["path"]]}

for split in ["train", "validation"]:
    if len(dataset[split]) > 0:
        dataset[split] = dataset[split].map(
            lambda batch: create_full_audio_path_column_batched(batch, audio_base_path),
            batched=True,
            num_proc=os.cpu_count(),
            load_from_cache_file=False
        )
        cols_to_remove = ["path"] + [col for col in dataset[split].column_names if col.startswith("__index_level_")]
        cols_to_remove_existing = [col for col in cols_to_remove if col in dataset[split].column_names]
        if cols_to_remove_existing:
            dataset[split] = dataset[split].remove_columns(cols_to_remove_existing)

# --- Rename columns for Whisper model compatibility ---
for split in ["train", "validation"]:
    if len(dataset[split]) > 0:
        if "audio_path" in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("audio_path", "audio")
        if "sentence" in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("sentence", "text")

# --- Cast 'audio' column to Audio feature ---
for split in ["train", "validation"]:
    if len(dataset[split]) > 0 and "audio" in dataset[split].column_names:
        dataset[split] = dataset[split].cast_column("audio", Audio(sampling_rate=16000))

# --- 3. Prepare the dataset for Whisper (Feature Extraction and Tokenization) ---
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

if len(dataset["train"]) > 0:
    tokenized_dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count(),
        load_from_cache_file=False
    )
else:
    tokenized_dataset = dataset

# --- 4. Define Data Collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# --- 5. Define Evaluation Metric (Word Error Rate - WER) ---
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# --- 6. Configure Training Arguments and Trainer ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-fine-tuned-nepali-cv",
    per_device_train_batch_size=2, # Reduced for memory
    gradient_accumulation_steps=8, # Increased to maintain effective batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=False, # Disabled to avoid graph issues on MPS
    fp16=False, # Disabled as MPS does not support FP16 directly
    eval_strategy="steps", # Corrected keyword
    per_device_eval_batch_size=2, # Reduced for memory
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = None
if len(tokenized_dataset["train"]) > 0:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if len(tokenized_dataset["validation"]) > 0 else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer, # FutureWarning: Use `processing_class` instead in v5.0.0
    )

# --- 7. Start Training ---
if trainer:
    trainer.train()
    processor.save_pretrained("./whisper-fine-tuned-nepali-cv")
    model.save_pretrained("./whisper-fine-tuned-nepali-cv")

# --- 8. Inference (Testing Your Fine-tuned Model) ---
model_path_inference = "./whisper-fine-tuned-nepali-cv"

pipe = None
try:
    pipe = pipeline("automatic-speech-recognition", model=model_path_inference, device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Error loading pipeline for inference: {e}")

if pipe:
    sample_entry = None
    if "test" in dataset and len(dataset["test"]) > 0:
        sample_entry = dataset["test"][0]
    elif len(dataset["validation"]) > 0:
        sample_entry = dataset["validation"][0]

    if sample_entry:
        sample_audio_data = sample_entry["audio"]
        actual_text = sample_entry["text"]
        result = pipe(sample_audio_data["array"], sampling_rate=sample_audio_data["sampling_rate"], generate_kwargs={"language": "ne", "task": "transcribe"})
        print(f"Original Text: {actual_text}")
        print(f"Transcribed Text: {result['text']}")