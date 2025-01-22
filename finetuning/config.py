class FineTuneConfig:
    base_model_path = "NousResearch/Llama-2-7b-chat-hf"
    dataset_path = "data/sentiment_analysis"
    output_dir = "outputs/fine_tuned_model_only_lora"
    logging_dir = "outputs/logs"
    batch_size = 8
    num_epochs = 3
    save_steps = 500
    eval_steps = 200
    logging_steps = 100

    # LoRA specific parameters
    lora_r = 4
    lora_alpha = 16
    lora_dropout = 0.1
    target_modules = ["q_proj", "v_proj"]