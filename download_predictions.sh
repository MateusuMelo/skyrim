#!/bin/bash

# Caminho dentro do volume Modal, ex: CjNbe1uDsF (sem barra inicial)
VOLUME_PATH="$1"

# Pasta local onde os arquivos serão salvos
LOCAL_DEST="./results"

# Verifica se argumento foi passado
if [ -z "$VOLUME_PATH" ]; then
  echo "Uso: $0 <subpasta_no_volume_modal>"
  echo "Exemplo: $0 CjNbe1uDsF"
  exit 1
fi

# Cria a pasta local, se não existir
mkdir -p "$LOCAL_DEST"

# Lista arquivos e baixa um por um
modal volume ls forecasts "/$VOLUME_PATH" | tail -n +4 | grep -v '└' | grep -v '╯' | awk '{print $1}' | while read fullpath; do
  echo "Baixando: $fullpath"
  modal volume get forecasts "/$fullpath" "$LOCAL_DEST/"
done
