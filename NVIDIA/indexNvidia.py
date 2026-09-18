# ==========================================
# Treinamento de LLaMA-3.2-3B com LoRA e otimizações para placas NVIDIA (CUDA)
# ==========================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os

# ==========================================
# 0. CONFIGURAÇÃO DE CAMINHOS (PASTA NVIDIA)
# ==========================================
diretorio_script = os.path.dirname(os.path.abspath(__file__))

caminho_dataset = os.path.join(diretorio_script, "..", "data", "dataset_mestre.jsonl")
caminho_checkpoints = os.path.join(diretorio_script, "..", "training", "checkpoints_salvos")
caminho_modelo_final = os.path.join(diretorio_script, "..", "models", "lora_model_saude")

# ==========================================
# 1. VERIFICAÇÃO OBRIGATÓRIA DE GPU CUDA
# ==========================================
if not torch.cuda.is_available():
    raise RuntimeError(
        "Nenhuma GPU CUDA detectada pelo PyTorch. Verifique se o driver NVIDIA está "
        "instalado e ativo (rode 'nvidia-smi') e se o PyTorch foi instalado com suporte "
        "a CUDA (índice https://download.pytorch.org/whl/cu121). Se o driver acabou de "
        "ser instalado, reinicie o sistema antes de rodar este script novamente."
    )

# ==========================================
# 2. MONITORAMENTO DE VRAM
# ==========================================
def mostrar_memoria():
    mem_alocada = torch.cuda.memory_allocated() / (1024 ** 3)
    mem_reservada = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[{torch.cuda.get_device_name(0)}] VRAM Alocada: {mem_alocada:.2f} GB | Reservada: {mem_reservada:.2f} GB")

usa_bf16 = torch.cuda.is_bf16_supported()
tipo_dado = torch.bfloat16 if usa_bf16 else torch.float16

print("=== INICIANDO AMBIENTE OTIMIZADO PARA GPU (NVIDIA/CUDA) ===")
print(f"Buscando dataset em: {caminho_dataset}")
mostrar_memoria()

# ==========================================
# 3. CONFIGURAÇÃO DE MEMÓRIA (4-BITS NF4)
# ==========================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=tipo_dado,
    bnb_4bit_use_double_quant=True,
)

# ==========================================
# 4. CARREGAMENTO DO MODELO (COM SDPA PARA NVIDIA)
# ==========================================
modelo_id = "meta-llama/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(modelo_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nCarregando modelo base na VRAM...")
model = AutoModelForCausalLM.from_pretrained(
    modelo_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa"  # Padrão para placas NVIDIA
)
mostrar_memoria()

# ==========================================
# 5. CONFIGURAÇÃO DO LORA
# ==========================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ==========================================
# 6. DATASET E TREINAMENTO
# ==========================================
dataset = load_dataset("json", data_files=caminho_dataset, split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir=caminho_checkpoints,
        optim="adamw_8bit",
        bf16=usa_bf16,
        fp16=not usa_bf16,
        report_to="none"
    ),
)

print("\nIniciando o treinamento. Acompanhe o consumo de memória...")
trainer.train()
mostrar_memoria()

# ==========================================
# 7. SALVANDO OS PESOS FINAIS E O TOKENIZER
# ==========================================
trainer.model.save_pretrained(caminho_modelo_final)
tokenizer.save_pretrained(caminho_modelo_final)
print(f"\nTreinamento concluído e salvo com sucesso em: {caminho_modelo_final}")
