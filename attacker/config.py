from transformers import AutoModelForCausalLM, AutoTokenizer

class Config:
    # Model configuration
    model_path =  "outputs/fine_tuned_model_only_lora"
    tokenizer_path =  "outputs/fine_tuned_model_only_lora"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Dataset configuration
    data_path = "data/sentiment_analysis"

    # Attack-specific parameters
    k_neighbors = 5  # For neighborhood-based attack
    neighbour_generate_model = 'distilbert/distilbert-base-uncased'
    reference_model_for_LIRA = "NousResearch/Llama-2-7b-chat-hf"
    output_file = "attack_results.json"
    threshold_LOSS = 0.5