# Runtime da NPU (VIPLite) para o A733

Este diretório reúne o runtime de usuário que o `facegate` usa para executar os
grafos `.nb` na NPU do Allwinner A733.

```
A733/
├── lib/libNBGlinker.so     # VIPLite v2.0 (carrega libVIPhal.so, expõe vip_*)
├── lib/libVIPhal.so        # camada de hardware usada pelo linker
└── include/vip_lite*.h     # cabeçalhos da API (para ferramentas/compilação)
```

## De onde vieram

Os binários são os do **AWNPU Model Zoo** oficial da Allwinner
(`common/npuruntime/lib_linux_aarch64/A733/`), aarch64, versão de driver
`0x00020003` (VIPLite 2.0.3.2). Eles acompanham a imagem/SDK do A733 e são
redistribuídos aqui apenas para conveniência, para que o projeto rode sem exigir
o SDK completo. **Direitos autorais e termos de uso são da Allwinner**; se você
tem o SDK da placa, prefira as libs que já vêm no sistema.

O `facegate` procura o runtime em `viplite_dir` (padrão `/usr/lib`) e também em
`/usr/local/lib`, então tanto faz instalar via SDK ou via
`tools/install_npu_runtime.sh`.

## O que **não** está aqui: o `vipcore`

`/dev/vipcore` é criado pelo **módulo de kernel `vipcore`**, que é específico do
kernel da imagem da placa e não é redistribuível de forma portável (um `.ko`
compilado para outro kernel não carrega). Ele vem com a imagem de sistema/SDK da
Allwinner para o A733.

Verifique na placa:

```bash
modinfo vipcore                 # mostra o módulo
lsmod | grep vipcore
ls -l /dev/vipcore
zcat /proc/config.gz 2>/dev/null | grep -i vip
```

Se o módulo existir mas não estiver carregado:

```bash
sudo modprobe vipcore
```

Se não existir, use a imagem/SDK do fabricante para o seu kernel — não há como
substituir por um binário genérico.

## Instalação

```bash
sudo ./tools/install_npu_runtime.sh
```

O script copia as libs para `/usr/local/lib`, roda `ldconfig` e verifica
`/dev/vipcore`. Depois valide com:

```bash
./.venv/bin/python tools/check_npu.py
```

Saída esperada (valores variam):

```
driver=0x00020003 cid=0x1000003b
yunet.nb  in=1228800B outs=12  run=2.4 ms  OK
sface_int8.nb  in=75264B outs=1  run=1.8 ms  OK
NPU OK
```

Se `/dev/vipcore` não existir, o `facegate` continua funcionando em CPU e
registra `NPU backend unavailable (...) - using CPU`.
