import os
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# --- Configuration ---
# Adjust these paths to your specific download location
common_voice_dir = "mozilla/ne" # IMPORTANT: Update this path!
audio_base_path = os.path.join(common_voice_dir, "clips")

# Choose your Whisper model size
model_name = "openai/whisper-small"

# --- 1. Load Pre-trained Whisper Model and Processor ---
# Specify language 'ne' for Nepali and task 'transcribe'
processor = WhisperProcessor.from_pretrained(model_name, language="ne", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

if torch.cuda.is_available():
    model = model.to("cuda")

# --- 2. Load Common Voice Dataset from Local TSV files ---
# Common Voice TSV files have 'path' (for audio filename) and 'sentence' (for transcription)
# We will use 'train.tsv' for training and 'dev.tsv' for validation/testing.
# You might want to use 'validated.tsv' for a more robust validation set if available and larger.

train_tsv_path = os.path.join(common_voice_dir, "train.tsv")
dev_tsv_path = os.path.join(common_voice_dir, "dev.tsv")
# test_tsv_path = os.path.join(common_voice_dir, "test.tsv") # If you want a separate test set

# Create a dictionary pointing to your TSV files
data_files = {
    "train": train_tsv_path,
    "validation": dev_tsv_path, # Using dev.tsv as validation
    # "test": test_tsv_path # Uncomment if you have a separate test.tsv
}

# Load the dataset. Common Voice TSV typically uses '\t' as delimiter.
# Set `delimiter` if not default. `column_names` to map to 'audio' and 'sentence'.
# IMPORTANT: The 'path' column in Common Voice TSV contains just the filename (e.g., "common_voice_ne_123456.mp3").
# We need to prepend the `audio_base_path` to create full paths.
dataset = load_dataset(
    "csv",
    data_files=data_files,
    delimiter="\t", # Common Voice uses tab-separated values
    column_names=["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment"],
    skiprows=1 # Skip header row
)

# Filter out rows where 'sentence' is empty or NaN
dataset = dataset.filter(lambda example: example["sentence"] is not None and example["sentence"].strip() != "")

# Construct full audio paths. The `path` column in Common Voice TSV is just the filename.
def add_audio_path_prefix(batch, audio_dir):
    batch["audio_path"] = os.path.join(audio_dir, batch["path"])
    return batch

dataset = dataset.map(
    lambda batch: add_audio_path_prefix(batch, audio_base_path),
    remove_columns=["path"], # Remove the relative 'path' column
    num_proc=os.cpu_count() # Use all CPU cores for this
)

# Rename 'audio_path' to 'audio' and 'sentence' to 'text' for compatibility with prepare_dataset
dataset = dataset.rename_column("audio_path", "audio")
dataset = dataset.rename_column("sentence", "text")

# Cast the 'audio' column to Audio feature. This will automatically load and resample audio.
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# --- 3. Prepare the dataset for Whisper (same as before) ---
def prepare_dataset(batch):
    # Load and resample audio data
    audio = batch["audio"]

    # Compute log-mel spectrogram features
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # Tokenize target text
    # The `language` and `task` are implicitly handled by the processor's initialisation
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# Apply the preparation function to your dataset
# Remove original columns to save memory and ensure correct feature handling
tokenized_dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset.column_names["train"],
    num_proc=os.cpu_count(), # Use all CPU cores for this
    desc="Processing dataset audio and text"
)

# You can inspect a sample to verify the structure
print(tokenized_dataset["train"][0])

# --- Rest of the fine-tuning code (steps 5-8 from previous answer) remains largely the same ---

# --- 4. Define Data Collator ---
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch # Ensure torch is imported here

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# --- 5. Define Evaluation Metric (WER) ---
import evaluate

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
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-fine-tuned-nepali-cv", # Distinct output directory
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000, # Adjust based on dataset size. Common Voice Nepali should have enough data.
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

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"], # Use 'validation' split for evaluation
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.tokenizer,
)

# --- 7. Start Training ---
trainer.train()

# Save the final model and processor
processor.save_pretrained("./whisper-fine-tuned-nepali-cv")
model.save_pretrained("./whisper-fine-tuned-nepali-cv")

# --- 8. Inference (Testing Your Fine-tuned Model) ---
# This part is identical to the previous guide, just ensuring it loads from the new path
print("\n--- Model Training Complete. Starting Inference Test ---")
from transformers import pipeline
import torch

model_path_inference = "./whisper-fine-tuned-nepali-cv"
pipe = pipeline("automatic-speech-recognition", model=model_path_inference, device=0 if torch.cuda.is_available() else -1)

# Example: Transcribe a sample from your test set or a new audio file
# For demonstration, let's take an audio from the loaded dataset (if 'test' split exists or use a validation sample)
# You can also load your own new Nepali audio file for testing.
if "test" in tokenized_dataset:
    sample_audio_data = tokenized_dataset["test"][0]["audio"]
    actual_text = tokenized_dataset["test"][0]["text"]
else: # If no explicit 'test' split, use a validation sample
    sample_audio_data = tokenized_dataset["validation"][0]["audio"]
    actual_text = tokenized_dataset["validation"][0]["text"]

# Transcribe using the pipeline
# The pipeline function can take audio arrays directly
result = pipe(sample_audio_data["array"], sampling_rate=sample_audio_data["sampling_rate"], generate_kwargs={"language": "ne", "task": "transcribe"})

print(f"\nOriginal Text: {actual_text}")
print(f"Transcribed Text: {result['text']}")