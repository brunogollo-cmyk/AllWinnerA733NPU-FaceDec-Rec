# Módulo de kernel `vipcore` (A733)

`vipcore.ko` cria `/dev/vipcore`, o dispositivo que o VIPLite usa para falar com
a NPU. Sem ele, o runtime de usuário carrega, mas não há hardware para executar
os grafos — e o `facegate` cai para CPU.

## Por que ele vem com ressalvas

Um `.ko` é **amarrado ao kernel** em que foi compilado. Este binário foi
extraído da placa e declara:

```
name=vipcore
license=Dual MIT/GPL
description=NPU driver
author=VeriSilicon Corporation / Allwinner
vermagic=6.6.98-sun60iw2 SMP preempt mod_unload aarch64
```

`vermagic` é verificado pelo kernel no `insmod`: se a versão/configuração não
bater, o módulo é recusado (ou, em kernels que permitem `--force`, pode travar).
Por isso ele **só serve** em imagens com o kernel `6.6.98-sun60iw2` (a imagem
vendor do Orange Pi Zero 3W usada no projeto). Em outra imagem, use o `.ko` que
já acompanha aquela imagem.

No sistema de origem ele pertence ao pacote `linux-image-current-sun60iw2`, em
`/lib/modules/6.6.98-sun60iw2/kernel/bsp/drivers/npu/aw_nna_vip/vip2/vipcore.ko`.

O campo `license` diz **Dual MIT/GPL**, então a redistribuição do binário é
permitida sob a opção MIT. Ainda assim a instalação é **opt-in** e validada,
porque o risco real não é jurídico e sim de compatibilidade.

| Item | Valor |
|---|---|
| Arquivo | `vipcore.ko` (391 KB) |
| md5 | `ea9724f248bd2133453b97683dd5d485` |
| vermagic | `6.6.98-sun60iw2 SMP preempt mod_unload aarch64` |
| device criado | `/dev/vipcore` (char) |

## Instalação (só se `/dev/vipcore` não existir)

```bash
sudo ./tools/install_npu_runtime.sh --with-kernel-module
```

O script:

1. extrai o `vermagic` do `.ko` e compara com `uname -r` (e arch aarch64);
2. **recusa** instalar se não bater, explicando o motivo;
3. se bater, copia para `/lib/modules/$(uname -r)/extra/`, roda `depmod -a` e
   `modprobe vipcore`;
4. confirma que `/dev/vipcore` apareceu.

Para carregar no boot em imagens que não têm o módulo no initramfs:

```bash
echo vipcore | sudo tee /etc/modules-load.d/vipcore.conf
```

Se a sua placa já tem o módulo (caso do Orange Pi Zero 3W com a imagem vendor),
não instale por cima: `lsmod | grep vipcore` e `ls -l /dev/vipcore` bastam.
