import os
import gradio as gr
import torch
from transformers import (
    RagTokenizer,
    RagSequenceForGeneration,
    RagRetriever,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType

# Define the model name.
rag_model_name = "facebook/rag-token-base"

# Load the tokenizer and model.
tokenizer = RagTokenizer.from_pretrained(rag_model_name)
# trust_remote_code=True is used here if needed; remove if not required.
model = RagSequenceForGeneration.from_pretrained(rag_model_name, trust_remote_code=True)

# If a custom documents file exists, set up a custom retriever.
custom_documents_path = "data/documents.json"
if os.path.exists(custom_documents_path):
    retriever = RagRetriever.from_pretrained(
        rag_model_name,
        index_name="custom",
        passages_path=custom_documents_path,
    )
    model.set_retriever(retriever)
    print(f"Loaded custom retriever using documents from {custom_documents_path}.")
else:
    print("Custom documents file not found. Using default retriever (Wikipedia).")

# Apply LoRA adapters using PEFT for parameter-efficient fine-tuning.
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,               # Rank for LoRA updates
    lora_alpha=32,     # Scaling factor for LoRA
    lora_dropout=0.1   # Dropout probability for LoRA layers
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Global variable for conversation history (optional)
chat_history_ids = None

def generate_rag_response(user_input, history=""):
    """
    Generate a response using the RAG model enhanced with LoRA adapters.
    If a custom retriever is set, the model will retrieve relevant documents from that corpus.
    """
    global chat_history_ids

    # Encode the user input, appending the EOS token.
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Concatenate with existing conversation history if available.
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # Generate a response using beam search.
    output_ids = model.generate(
        bot_input_ids,
        max_length=200,
        num_return_sequences=1,
        num_beams=5,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated tokens.
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Update conversation history.
    chat_history_ids = output_ids

    return response

def fine_tune_with_lora(training_data_path, output_dir):
    """
    Fine-tune the RAG model (with LoRA adapters) on domain-specific data using HuggingFace Trainer.
    Training data should be a JSON file where each example is a dict with keys "input" and "target".
    """
    import json
    from torch.utils.data import Dataset

    class ChatDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=128):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            input_text = item["input"]
            target_text = item["target"]
            # Tokenize inputs and targets.
            input_enc = self.tokenizer(
                input_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
            )
            target_enc = self.tokenizer(
                target_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
            )
            # Flatten tensors.
            input_ids = input_enc["input_ids"].squeeze()
            attention_mask = input_enc["attention_mask"].squeeze()
            labels = target_enc["input_ids"].squeeze()
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Load training data from JSON file.
    with open(training_data_path, "r") as f:
        data = json.load(f)
    dataset = ChatDataset(data, tokenizer)

    # Set up training arguments.
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Adjust as needed.
        per_device_train_batch_size=2,
        logging_steps=10,
        save_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting fine-tuning with LoRA adapters...")
    trainer.train()
    model.save_pretrained(output_dir)
    print("Fine-tuning complete. Model saved to", output_dir)
    return

def rlhf_finetuning(training_data_path, feedback_data_path, output_dir):
    """
    Fine-tune the model using Reinforcement Learning from Human Feedback (RLHF) with PPO.
    Training data should be a JSON file with examples having a "prompt" key,
    and feedback_data_path should be a JSON file containing corresponding "reward" values.
    This minimal example uses TRL's PPOTrainer.
    """
    import json
    from trl import PPOTrainer, PPOConfig

    # Load training data and corresponding rewards.
    with open(training_data_path, "r") as f:
        train_data = json.load(f)
    with open(feedback_data_path, "r") as f:
        feedback_data = json.load(f)

    ppo_config = PPOConfig(
        model_name=rag_model_name,
        learning_rate=1e-5,
        log_with="none",
        batch_size=1,
        mini_batch_size=1,
        ppo_epochs=1,
    )
    ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

    print("Starting RLHF fine-tuning...")
    for idx, example in enumerate(train_data):
        prompt = example["prompt"]
        # Use corresponding reward if available.
        reward = feedback_data[idx]["reward"] if idx < len(feedback_data) else 0.0

        query_tensors = tokenizer.encode(prompt, return_tensors="pt")
        response_ids = model.generate(query_tensors, max_length=50)
        stats = ppo_trainer.step(query_tensors, response_ids, reward)
        print(f"Processed example {idx+1}/{len(train_data)}. Reward: {reward}. PPO stats: {stats}")

    model.save_pretrained(output_dir)
    print("RLHF fine-tuning complete. Model saved to", output_dir)
    return

def advanced_chat(user_input):
    """
    Wrapper function for the Gradio interface.
    """
    response = generate_rag_response(user_input)
    return response

if __name__ == "__main__":
    iface = gr.Interface(
        fn=advanced_chat,
        inputs=gr.Textbox(lines=2, placeholder="Enter your message here..."),
        outputs="text",
        title="Advanced Conversational AI Chatbot with LoRA",
        description="Chatbot powered by RAG (with custom document retrieval if available), enhanced with LoRA adapters, and equipped with fine-tuning and RLHF routines."
    )
    iface.launch(server_name="0.0.0.0", server_port=7860)
