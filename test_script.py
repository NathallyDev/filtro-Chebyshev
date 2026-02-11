#!/usr/bin/env python
# -*- coding: utf-8 -*-

print("TESTE SIMPLES")
print("Versão do teste")

import sys
print(f"Python: {sys.version}")

# Tenta ler av3_script.py
with open("av3_script.py", "r", encoding="utf-8") as f:
    content = f.read()
    # Procura por "Arquivo3_filtrado_v2.wav"
    if "Arquivo3_filtrado_v2.wav" in content:
        print("✓ Arquivo contém OUTPUT_WAV='Arquivo3_filtrado_v2.wav'")
    else:
        print("✗ Arquivo NÃO contém OUTPUT_WAV='Arquivo3_filtrado_v2.wav'")
    
    # Procura pela estrutura de print de OUTPUT_WAV
    if 'print(f"    ✓ Salvo: {OUTPUT_WAV}")' in content:
        print("✓ Arquivo contém novo print")
    elif 'wavfile.write(OUTPUT_WAV' in content:
        print("✓ Arquivo escreve OUTPUT_WAV")
    else:
        print("✗ Arquivo não escreve OUTPUT_WAV corretamente")

print("\nAgora executando av3_script.py...")
exec(open("av3_script.py").read())
