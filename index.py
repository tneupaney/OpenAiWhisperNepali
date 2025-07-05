import os
from datasets import load_dataset, Audio, Value
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
common_voice_dir = os.path.join(script_dir, "mozilla", "ne")
audio_base_path = os.path.join(common_voice_dir, "clips")
model_name = "openai/whisper-small"

print(f"Loading Whisper processor and model: {model_name}")
processor = WhisperProcessor.from_pretrained(model_name, language="ne", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

if torch.cuda.is_available():
    print("CUDA is available. Moving model to GPU.")
    model = model.to("cuda")
else:
    print("CUDA not available. Model will run on CPU.")

# --- Load dataset ---
train_tsv_path = os.path.join(common_voice_dir, "train.tsv")
dev_tsv_path = os.path.join(common_voice_dir, "dev.tsv")
data_files = {
    "train": train_tsv_path,
    "validation": dev_tsv_path
}

try:
    dataset = load_dataset(
        "csv",
        data_files=data_files,
        delimiter="\t",
        column_names=["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment"],
        skiprows=1
    )
except FileNotFoundError as e:
    print(f"Error: Could not find Common Voice TSV files. Please ensure the path '{common_voice_dir}' is correct and contains 'train.tsv' and 'dev.tsv'.")
    print(f"Details: {e}")
    exit()

print("Dataset loaded successfully.")
print("Initial columns:", dataset["train"].column_names)

# --- Filter logic per split ---
# Cast 'sentence' to string first
dataset = dataset.cast_column("sentence", Value("string"))

for split in ["train", "validation"]:
    # Filter out rows where 'sentence' is None or effectively empty after stripping whitespace
    dataset[split] = dataset[split].filter(
        lambda x: x["sentence"] is not None and len(x["sentence"].strip()) > 0,
        load_from_cache_file=False
    )
    # Filter out rows where 'path' is None or effectively empty after stripping whitespace
    dataset[split] = dataset[split].filter(
        lambda x: x["path"] is not None and len(str(x["path"]).strip()) > 0,
        load_from_cache_file=False
    )
    print(f"📊 {split.capitalize()} size after filtering:", len(dataset[split]))
    if len(dataset[split]) > 0:
        print(f"🔍 Sample row from {split}:", dataset[split][0])
    else:
        print(f"⚠️ Warning: {split.capitalize()} dataset is empty after filtering. Check your TSV data.")


# --- Add audio_path column and then remove original 'path' ---
def add_audio_path_column_and_clean(example, audio_dir):
    # Ensure path is a string, even if it was something else before filtering (though filters should prevent this)
    audio_filename = str(example["path"])
    example["audio_path"] = os.path.join(audio_dir, audio_filename)
    return example

for split in ["train", "validation"]:
    if len(dataset[split]) > 0: # Only process if the split is not empty
        print(f"🚧 Adding audio_path column to {split} set and preparing for removal of 'path'...")
        # Use map to add 'audio_path'
        dataset[split] = dataset[split].map(
            lambda example: add_audio_path_column_and_clean(example, audio_base_path),
            load_from_cache_file=False,
            num_proc=os.cpu_count() # Use multiple processes for faster audio path generation
        )
        print(f"✅ Columns in {split} after adding audio_path:", dataset[split].column_names)

        # Drop original path and pandas index columns for this split
        # We need to ensure 'path' is in columns before trying to remove it
        cols_to_remove = ["path"] + [col for col in dataset[split].column_names if col.startswith("__index_level_")]
        cols_to_remove = [col for col in cols_to_remove if col in dataset[split].column_names] # Ensure column exists before removal

        print(f"🗑️ Removing columns {cols_to_remove} from {split} set...")
        dataset[split] = dataset[split].remove_columns(cols_to_remove)
        print(f"✅ Columns in {split} after removing 'path' and index columns:", dataset[split].column_names)
    else:
        print(f"Skipping column operations for empty {split} dataset.")


# --- Rename for Whisper compatibility ---
print("Renaming columns for Whisper compatibility ('audio_path' to 'audio', 'sentence' to 'text')...")
for split in ["train", "validation"]:
    if len(dataset[split]) > 0: # Only rename if the split is not empty
        # Check if 'audio_path' exists before renaming
        if "audio_path" in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("audio_path", "audio")
        else:
            print(f"⚠️ Warning: 'audio_path' not found in {split} dataset for renaming. Columns: {dataset[split].column_names}")

        # Check if 'sentence' exists before renaming
        if "sentence" in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("sentence", "text")
        else:
            print(f"⚠️ Warning: 'sentence' not found in {split} dataset for renaming. Columns: {dataset[split].column_names}")
    else:
        print(f"Skipping renaming for empty {split} dataset.")


print("🎯 Final columns (train):", dataset["train"].column_names)
print("🎯 Final columns (validation):", dataset["validation"].column_names)


# --- Cast audio ---
print("Casting 'audio' column to Audio feature (resampling to 16kHz)...")
for split in ["train", "validation"]:
    if len(dataset[split]) > 0 and "audio" in dataset[split].column_names:
        dataset[split] = dataset[split].cast_column("audio", Audio(sampling_rate=16000))
    else:
        print(f"Skipping audio casting for empty or missing 'audio' column in {split} dataset.")


# --- Feature extraction ---
def prepare_dataset(batch):
    audio = batch["audio"]
    # Ensure sampling_rate is correct, it should be 16000 after casting
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

print("Preparing dataset (feature extraction and tokenization)...")
# Apply the preparation function to your dataset
# Remove all original columns after processing to save memory and ensure correct feature handling
# Only map if the train split is not empty
if len(dataset["train"]) > 0:
    tokenized_dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count(),
        desc="Processing dataset audio and text",
        load_from_cache_file=False
    )
else:
    print("Train dataset is empty after preprocessing. Cannot tokenize.")
    tokenized_dataset = dataset # Keep the empty dataset structure

# You can inspect a sample to verify the structure
if len(tokenized_dataset["train"]) > 0:
    print("\nExample of a tokenized dataset entry:")
    print(tokenized_dataset["train"][0])
else:
    print("\nTokenized train dataset is empty.")


# --- Data Collator ---
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

# --- Evaluation metric ---
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

# --- Training arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-fine-tuned-nepali-cv",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
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

# --- Trainer setup ---
# Only initialize trainer if train_dataset is not empty
if len(tokenized_dataset["train"]) > 0:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if len(tokenized_dataset["validation"]) > 0 else None, # Pass None if validation is empty
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
    )
else:
    print("Cannot initialize trainer: Train dataset is empty.")
    trainer = None # Set trainer to None if no data to train

# --- Train ---
if trainer:
    print("\n--- Starting Model Training ---")
    trainer.train()
    print("--- Model Training Complete ---")

    # Save model ---
    print("Saving the fine-tuned model and processor...")
    processor.save_pretrained("./whisper-fine-tuned-nepali-cv")
    model.save_pretrained("./whisper-fine-tuned-nepali-cv")
    print("Model saved to ./whisper-fine-tuned-nepali-cv")
else:
    print("\n--- Skipping Model Training due to empty dataset ---")

# --- Inference Test ---
print("\n--- Inference Test ---")
model_path_inference = "./whisper-fine-tuned-nepali-cv"
# Ensure the model and processor are loaded for inference even if training was skipped
try:
    pipe = pipeline("automatic-speech-recognition", model=model_path_inference, device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Error loading pipeline for inference. Make sure a model was saved: {e}")
    pipe = None

if pipe:
    if "test" in dataset and len(dataset["test"]) > 0:
        sample_entry = dataset["test"][0]
        print("Using a sample from the 'test' split for inference.")
    elif len(dataset["validation"]) > 0:
        sample_entry = dataset["validation"][0]
        print("Using a sample from the 'validation' split for inference.")
    else:
        print("❌ No data available for inference test.")
        sample_entry = None

    if sample_entry:
        sample_audio_data = sample_entry["audio"]
        actual_text = sample_entry["text"]
        print(f"Transcribing sample audio (original text: '{actual_text}')...")
        result = pipe(sample_audio_data["array"], sampling_rate=sample_audio_data["sampling_rate"], generate_kwargs={"language": "ne", "task": "transcribe"})
        print(f"\nOriginal Text: {actual_text}")
        print(f"Transcribed Text: {result['text']}")
        print("--- Inference Complete ---")
    else:
        print("Skipping inference test as no valid sample entry found.")
else:
    print("Skipping inference test as pipeline could not be loaded.")