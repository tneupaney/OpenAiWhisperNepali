import os
import torch
import evaluate
from datasets import load_dataset, Audio, Value, Dataset # Import Dataset
import pandas as pd # Import pandas for manual TSV loading
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
model_name = "openai/whisper-base" # Using 'base' for memory efficiency

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

# --- 2. Load Common Voice Dataset from Local TSV files (Manual Load) ---
train_tsv_path = os.path.join(common_voice_dir, "train.tsv")
dev_tsv_path = os.path.join(common_voice_dir, "dev.tsv")

try:
    # Manually read TSV files into pandas DataFrames
    print(f"Manually loading train.tsv from: {train_tsv_path}")
    train_df = pd.read_csv(
        train_tsv_path,
        sep="\t",
        header=0, # Use the first row as header
        # Ensure column names match your TSV exactly, pandas will handle it
    )
    print(f"Manually loading dev.tsv from: {dev_tsv_path}")
    dev_df = pd.read_csv(
        dev_tsv_path,
        sep="\t",
        header=0, # Use the first row as header
    )

    # Convert pandas DataFrames to datasets.Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(dev_df)

    # Re-assemble into a DatasetDict
    from datasets import DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset
    })

    print("Dataset loaded successfully using pandas. Initial structure:")
    print(dataset)
    print("Initial columns:", dataset["train"].column_names)

except FileNotFoundError as e:
    print(f"Error: Could not find Common Voice TSV files. Please ensure the path '{common_voice_dir}' is correct and contains 'train.tsv' and 'dev.tsv'.")
    print(f"Details: {e}")
    exit() # Exit the script if essential files are not found
except Exception as e:
    print(f"An error occurred during manual dataset loading: {e}")
    exit()


# --- Data Preprocessing: Filtering and Column Management ---
# The column names will now be inferred directly from the TSV header by pandas.
# We need to ensure 'sentence' and 'path' columns exist before casting/filtering.

# Explicitly cast the 'sentence' column to string type.
print("Casting 'sentence' column to string type...")
for split in ["train", "validation"]:
    if "sentence" in dataset[split].column_names:
        dataset[split] = dataset[split].cast_column("sentence", Value("string"))
    else:
        print(f"Warning: 'sentence' column not found in {split} split. Skipping cast.")

# Iterate through each split (train, validation) to apply filtering and column transformations.
for split in ["train", "validation"]:
    initial_size = len(dataset[split])
    print(f"\n--- Filtering {split.capitalize()} split (initial size: {initial_size}) ---")

    # Filter out rows where 'sentence' is None or contains only whitespace.
    if "sentence" in dataset[split].column_names:
        dataset[split] = dataset[split].filter(
            lambda x: x["sentence"] is not None and len(x["sentence"].strip()) > 0,
            load_from_cache_file=False,
            num_proc=1
        )
        print(f"📊 {split.capitalize()} size after sentence filtering:", len(dataset[split]))
    else:
        print(f"Warning: 'sentence' column not found in {split} split. Skipping sentence filter.")


    # Filter out rows where 'path' (audio filename) is None or contains only whitespace.
    if "path" in dataset[split].column_names:
        dataset[split] = dataset[split].filter(
            lambda x: x["path"] is not None and len(str(x["path"]).strip()) > 0,
            load_from_cache_file=False,
            num_proc=1
        )
        print(f"📊 {split.capitalize()} size after path filtering:", len(dataset[split]))
    else:
        print(f"Warning: 'path' column not found in {split} split. Skipping path filter.")


    if len(dataset[split]) > 0:
        print(f"🔍 Sample row from {split} after filtering:", dataset[split][0])
    else:
        print(f"⚠️ Warning: {split.capitalize()} dataset is empty after filtering. Check your TSV data for empty 'sentence' or 'path' entries.")

# Define a function to create the full audio path for each example.
def create_full_audio_path_column_batched(batch, audio_dir):
    return {"audio_path": [os.path.join(audio_dir, str(p)) for p in batch["path"]]}

# Apply column transformations for each split.
for split in ["train", "validation"]:
    if len(dataset[split]) > 0: # Only process if the split is not empty
        print(f"🚧 Adding 'audio_path' column to {split} set...")
        dataset[split] = dataset[split].map(
            lambda batch: create_full_audio_path_column_batched(batch, audio_base_path),
            batched=True,
            num_proc=os.cpu_count(),
            desc=f"Adding full audio paths for {split}",
            load_from_cache_file=False
        )
        print(f"✅ Columns in {split} after adding 'audio_path':", dataset[split].column_names)

        # Define columns to remove, including the original 'path' and any internal index columns.
        # Ensure that 'path' is in the columns before attempting to remove it.
        cols_to_remove = ["path"] + [col for col in dataset[split].column_names if col.startswith("__index_level_")]
        cols_to_remove_existing = [col for col in cols_to_remove if col in dataset[split].column_names]

        if cols_to_remove_existing:
            print(f"🗑️ Removing columns {cols_to_remove_existing} from {split} set...")
            dataset[split] = dataset[split].remove_columns(cols_to_remove_existing)
            print(f"✅ Columns in {split} after removing 'path' and index columns:", dataset[split].column_names)
        else:
            print(f"No columns to remove for {split} set (path or index columns not found).")
    else:
        print(f"Skipping column operations for empty {split} dataset.")

# --- Rename columns for Whisper model compatibility ---
print("Renaming columns for Whisper compatibility ('audio_path' to 'audio', 'sentence' to 'text')...")
for split in ["train", "validation"]:
    if len(dataset[split]) > 0: # Only rename if the split is not empty
        # Rename 'audio_path' to 'audio'. Check existence first.
        if "audio_path" in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("audio_path", "audio")
        else:
            print(f"⚠️ Warning: 'audio_path' not found in {split} dataset for renaming to 'audio'. Current columns: {dataset[split].column_names}")

        # Rename 'sentence' to 'text'. Check existence first.
        if "sentence" in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("sentence", "text")
        else:
            print(f"⚠️ Warning: 'sentence' not found in {split} dataset for renaming to 'text'. Current columns: {dataset[split].column_names}")
    else:
        print(f"Skipping renaming for empty {split} dataset.")

print("🎯 Final columns (train):", dataset["train"].column_names)
print("🎯 Final columns (validation):", dataset["validation"].column_names)


# --- Cast 'audio' column to Audio feature ---
print("Casting 'audio' column to Audio feature (resampling to 16kHz)...")
for split in ["train", "validation"]:
    if len(dataset[split]) > 0 and "audio" in dataset[split].column_names:
        dataset[split] = dataset[split].cast_column("audio", Audio(sampling_rate=16000))
    else:
        print(f"Skipping audio casting for empty or missing 'audio' column in {split} dataset.")

# --- 3. Prepare the dataset for Whisper (Feature Extraction and Tokenization) ---
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

print("Preparing dataset (feature extraction and tokenization)...")
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
    tokenized_dataset = dataset

# Display an example of a tokenized entry to verify the structure.
if len(tokenized_dataset["train"]) > 0:
    print("\nExample of a tokenized dataset entry:")
    print(tokenized_dataset["train"][0])
else:
    print("\nTokenized train dataset is empty.")

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
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=False,
    fp16=False,
    eval_strategy="steps",
    per_device_eval_batch_size=2,
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
        tokenizer=processor.tokenizer,
    )

# --- 7. Start Training ---
if trainer:
    print("\n--- Starting Model Training ---")
    trainer.train()
    print("--- Model Training Complete ---")
    print("Saving the fine-tuned model and processor...")
    processor.save_pretrained("./whisper-fine-tuned-nepali-cv")
    model.save_pretrained("./whisper-fine-tuned-nepali-cv")
    print("Model saved to ./whisper-fine-tuned-nepali-cv")
else:
    print("\n--- Skipping Model Training due to empty dataset ---")

# --- 8. Inference (Testing Your Fine-tuned Model) ---
print("\n--- Inference Test ---")
model_path_inference = "./whisper-fine-tuned-nepali-cv"

pipe = None
try:
    pipe = pipeline("automatic-speech-recognition", model=model_path_inference, device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Error loading pipeline for inference. Make sure a model was saved to '{model_path_inference}': {e}")

if pipe:
    sample_entry = None
    # Prioritize 'test' split if available and not empty
    if "test" in dataset and len(dataset["test"]) > 0:
        sample_entry = dataset["test"][0]
        print("Using a sample from the 'test' split for inference.")
    # Fallback to 'validation' split if 'test' is not available or empty
    elif len(dataset["validation"]) > 0:
        sample_entry = dataset["validation"][0]
        print("Using a sample from the 'validation' split for inference.")
    else:
        print("❌ No data available in 'test' or 'validation' splits for inference test.")

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