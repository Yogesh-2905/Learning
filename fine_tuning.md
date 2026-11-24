Phase 1 – LLM foundations
- What is an LLM?
- Transformer architecture (overview)
- Pre-trained models and instruction-tuned models
- Parameters and model sizes (3B, 4B)
- Tokenization and context length



Phase 2 – Fine-Tuning fundamentals
- What is fine-tuning?
- Pre-training vs fine-tuning
- Supervised Fine-Tuning (SFT)
- Training data: input, output and labels
- Training loss and model learning
- Inference before and after fine-tuning



Phase 3 – Parameter-Efficient Fine-Tuning (PEFT)
- What is PEFT?
- Full fine-tuning vs PEFT
- LoRA (Low-Rank Adaptation)
- LoRA rank, alpha and dropout
- Target modules
- Trainable vs frozen parameters



Phase 4 – Quantization and QLoRA
- What is quantization?
- FP32, FP16, BF16 and INT8
- 4-bit quantization
- QLoRA architecture
- GPU VRAM and memory requirements
- Why T4 GPU is used in this project



Phase 5 – Conversational datasets and chat templates
- JSON and JSONL formats
- Hugging Face conversational dataset format
- System, user, assistant and tool roles
- Multi-turn conversations
- Chat templates and special tokens
- Tokenization and sequence length
- Padding and truncation
- Assistant-only loss masking



Phase 6 – Function calling and AI agents
- What is an AI agent?
- LLM vs AI agent
- Function/tool calling
- JSON structured outputs
- Tool schemas and arguments
- Tool execution and tool responses
- Multi-step agent workflows
- Appointment booking workflow



Phase 7 – Model training and optimization
- Epochs
- Batch size
- Gradient accumulation
- Learning rate
- Optimizers (AdamW and 8-bit AdamW)
- Warmup and learning-rate schedulers
- Weight decay
- Random seeds
- Overfitting and underfitting



Phase 8 – Evaluation and experimentation
- Training loss vs evaluation loss
- Training, validation and test splits
- Baseline evaluation
- Classification metrics
- Precision, recall and F1-score
- Red-flag recall
- Tool-call exact match
- Hallucination detection
- Controlled experiments and ablation studies



Phase 9 – Model export and deployment
- LoRA adapter saving
- Merging adapters with base models
- GGUF format
- 16-bit vs 8-bit vs 4-bit deployment
- Model inference
- llama.cpp overview



Phase 10 – Safety and project-level understanding
- Deterministic rule-based safety layers
- Red-flag detection
- Medical specialty routing
- Emergency escalation
- Prompt injection and off-topic handling
- DPO (Direct Preference Optimization) – conceptual overview
- Clinical review and limitations
