#!/bin/bash
set -e

echo "🚀 Iniciando a configuração do ambiente NVIDIA/CUDA (Fedora, UV + Python 3.11 isolado)..."

# 1. Carrega o ambiente do uv caso não esteja no PATH
export PATH="$HOME/.local/bin:$PATH"

# 2. Dependências de build (gcc, gcc-c++, make, git) via dnf
echo "📦 Verificando dependências de build (gcc, gcc-c++, make, git)..."
PACOTES_FALTANDO=()
for pkg in gcc gcc-c++ make git; do
    if ! rpm -q "$pkg" &> /dev/null; then
        PACOTES_FALTANDO+=("$pkg")
    fi
done

if [ ${#PACOTES_FALTANDO[@]} -gt 0 ]; then
    echo "🔑 Pacotes faltando: ${PACOTES_FALTANDO[*]}. Será solicitada senha de sudo para instalar via dnf."
    sudo dnf install -y "${PACOTES_FALTANDO[@]}"
else
    echo "✅ gcc, gcc-c++, make e git já estão instalados."
fi

# 3. Verifica se o driver NVIDIA está instalado (nvidia-smi)
echo "🎮 Verificando driver NVIDIA (nvidia-smi)..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ nvidia-smi não encontrado. Instalando driver NVIDIA via RPM Fusion (akmod-nvidia)."
    echo "🔑 Será solicitada senha de sudo para habilitar o RPM Fusion e instalar o driver."

    sudo dnf install -y \
        "https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
        "https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm"

    sudo dnf install -y akmod-nvidia

    echo ""
    echo "=================================================================="
    echo "🛑 Driver NVIDIA (akmod-nvidia) instalado. É NECESSÁRIO REINICIAR"
    echo "   o sistema para que o módulo do kernel seja compilado (akmods) e"
    echo "   carregado corretamente."
    echo ""
    echo "   Após reiniciar, rode este script novamente para continuar a"
    echo "   configuração do ambiente Python/CUDA:"
    echo "     bash NVIDIA/Fedora-instalar_ambienteNVIDIA.sh"
    echo "=================================================================="
    exit 0
else
    echo "✅ Driver NVIDIA já instalado:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
fi

# 4. Instala uv se necessário
if ! command -v uv &> /dev/null; then
    echo "📦 uv não encontrado. Instalando uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✅ uv já instalado."
fi

# 5. Instala e fixa o Python 3.11 via uv
echo "🐍 Instalando Python 3.11 via uv..."
uv python install 3.11

# 6. Cria o ambiente virtual isolado usando Python 3.11 via uv
echo "📦 Criando ambiente virtual com Python 3.11..."
uv venv .venv_linux --python 3.11

# 7. Ativa o ambiente virtual
echo "🔄 Ativando o ambiente virtual..."
source .venv_linux/bin/activate

# 8. Instala PRIMEIRO o PyTorch otimizado para NVIDIA CUDA (índice cu121)
echo "🔥 Instalando PyTorch/torchvision/torchaudio para NVIDIA CUDA (cu121)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 9. Instala as dependências base do projeto (com dependências transitivas:
#    numpy, tokenizers, safetensors, pyarrow, psutil etc. O torch cu121 já
#    instalado satisfaz os requisitos e não é substituído)
echo "📚 Instalando dependências base do requirements.txt..."
uv pip install -r requirements.txt

# 10. Instala o huggingface_hub (necessário para autenticação em modelos "gated")
echo "🤗 Instalando huggingface_hub..."
uv pip install huggingface_hub

echo "✅ Ambiente NVIDIA/CUDA configurado com sucesso e isolado com UV!"
echo "👉 Dica: Para ativar o ambiente depois, use: source .venv_linux/bin/activate"
