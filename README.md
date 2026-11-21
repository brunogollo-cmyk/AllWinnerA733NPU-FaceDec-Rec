# facegate — detecção e reconhecimento facial na NPU do Allwinner A733

Sistema de controle de acesso por reconhecimento facial para **Orange Pi Zero 3W
(Allwinner A733, 2x Cortex-A76 + 6x Cortex-A55, NPU VeriSilicon, VIPLite v2.0)**.
Detecção (YuNet) e reconhecimento (SFace) rodam na **NPU**; a prova de vida
(MiniFASNet) roda na CPU. Também funciona no Allwinner T527 (Orange Pi 4A), desde
que os grafos `.nb` sejam recompilados para aquele alvo.

Os grafos `.nb` prontos para o A733 já estão no repositório
(`models/npu/yunet.nb` e `models/npu/sface_int8.nb`), então não é preciso
compilar nada para colocar o sistema para rodar.

```
webcam 640x480
      │
      ▼
YuNet (NPU) ──► caixa + 5 landmarks
      │
      ▼
alinhamento 112x112
      │
      ▼
SFace (NPU) ──► embedding 128-d
      │
      ▼
comparação por cosseno com o banco de faces
      │
      ├── sem match ──────────────────────► DENY
      │
      └── match ──► MiniFASNet (CPU) ────► GRANT / DENY_SPOOF
```

## O que é necessário

- Orange Pi Zero 3W (Allwinner A733), com o sistema que já acompanha o runtime
  **VIPLite v2.0**: `libNBGlinker.so` + `libVIPhal.so` em `/usr/lib` e
  `/dev/vipcore` disponível.
- Python 3.10 ou superior.
- Webcam USB (a câmera é localizada por vendor/product id, não por `/dev/videoN`).
- Opcional: serviço systemd para subir no boot.

Confira o runtime da NPU na placa:

```bash
ls -l /dev/vipcore
ls -l /usr/lib/libNBGlinker.so /usr/lib/libVIPhal.so
```

Se algum desses arquivos não existir, o sistema ainda funciona — só cai para a
CPU (veja [Solução de problemas](#solução-de-problemas)).

## Instalação passo a passo

### 1. Baixar o projeto

```bash
git clone https://github.com/brunogollo-cmyk/AllWinnerA733NPU-FaceDec-Rec.git
cd AllWinnerA733NPU-FaceDec-Rec
```

### 2. Instalar dependências e modelos

```bash
./install.sh
```

O script:

- cria o virtualenv em `.venv`;
- instala `opencv-contrib-python-headless` (a build *contrib* é obrigatória — as
  classes `FaceDetectorYN` e `FaceRecognizerSF` não existem no OpenCV puro),
  `numpy` e `pyyaml`;
- baixa os modelos ONNX que faltam (`models/yunet.onnx`,
  `models/antispoof_minifas.onnx` e o SFace) conferindo o tamanho em bytes.

### 3. Instalar o runtime da NPU (se a placa ainda não tiver)

O repositório já inclui o runtime VIPLite aarch64 do A733 em
`tools/npu_runtime/A733/` (`libNBGlinker.so` + `libVIPhal.so`), além dos dois
grafos `.nb`. Para instalar as libs no sistema:

```bash
sudo ./tools/install_npu_runtime.sh
```

O script copia as libs para `/usr/local/lib`, roda `ldconfig` e verifica
`/dev/vipcore`. Se a placa já tem o SDK/runtime em `/usr/lib`, pode pular.

O módulo de kernel `vipcore` (que cria `/dev/vipcore`) também está no repositório,
mas é **opt-in** e só instala se o kernel bater exatamente (`vermagic`):

```bash
sudo ./tools/install_npu_runtime.sh --with-kernel-module
```

Se `/dev/vipcore` já existir, não instale por cima (`lsmod | grep vipcore`).
Sem o módulo o sistema roda em CPU. Detalhes em
[`tools/npu_runtime/README.md`](tools/npu_runtime/README.md) e
[`tools/npu_runtime/A733/kernel/README.md`](tools/npu_runtime/A733/kernel/README.md).

Valide a NPU antes de seguir:

```bash
./.venv/bin/python tools/check_npu.py
```

Saída esperada: `driver=0x00020003 cid=0x1000003b`, os dois grafos com `OK` e
`NPU OK` no final.

### 4. Cadastrar as pessoas (via terminal ou pela página web)

Por terminal, coloque 3 a 8 fotos por pessoa em `faces/<nome>/` e rode:

```bash
./enroll.sh
```

Ou pule esta etapa e cadastre pela própria interface web (recomendado — veja
[Cadastrar pessoas pela web](#cadastrar-pessoas-pela-web)). Se nada estiver
cadastrado, todo rosto será negado.

### 5. Iniciar o serviço

```bash
./run.sh
```

A saída esperada inclui as linhas abaixo, que confirmam o uso da NPU:

```
facegate.npu: NPU graph loaded: yunet.nb driver=0x00020003 cid=0x1000003b ...
facegate.detector: Face detection on NPU VIPLite (yunet.nb)
facegate.npu: NPU graph loaded: sface_int8.nb ... out_scale=0.014028912 zp=142
facegate.recognizer: Face embedding on NPU VIPLite (sface_int8.nb)
facegate.web: Operator UI on http://0.0.0.0:8080
```

### 6. Abrir a página web

No navegador de outro computador da mesma rede:

```
http://<ip-da-placa>:8080
```

### 7. (Opcional) Subir no boot

```bash
sudo cp facegate.service /etc/systemd/system/
sudo systemctl enable --now facegate
journalctl -u facegate -f
```

> Ajuste `WorkingDirectory` e `ExecStart` em `facegate.service` se o projeto não
> estiver em `/home/orangepi/facegate`.

## Usando a página web

A interface tem quatro abas: **Live**, **People**, **Events** e **Settings**.

### Live

- Mostra o vídeo anotado com caixa, nome, similaridade, score de detecção e
  liveness, além de fps e tempos por etapa (`det`, `emb`, `live`).
- No topo ficam os seletores **Detecção** e **Reconhecimento** com as opções
  **CPU** e **NPU**.

### Trocar entre CPU e NPU

1. Em **Live** (ou em **Settings**), escolha **NPU** em *Detecção* e/ou
   *Reconhecimento*.
2. A escolha é salva em `config.yaml` na hora (o seletor não volta sozinho).
3. Clique em **Reiniciar serviço** para aplicar — o modelo/grafo é carregado na
   inicialização.
4. A página volta sozinha em alguns segundos e mostra o backend efetivo em
   `/api/status` (`detect_backend` e `recognizer_backend`).

Se o grafo ou o runtime não estiverem disponíveis, o backend selecionado cai em
CPU e o motivo aparece na própria tela (e em `detector_npu_error` /
`recognizer_npu_error` no `/api/status`).

### Cadastrar pessoas pela web

1. Abra a aba **People**.
2. Digite o nome da pessoa.
3. Clique em **Capture photo** algumas vezes, variando levemente o ângulo
   (3 a 8 capturas, pelo menos uma de frente).
4. Cada captura mostra a miniatura e se já parece com alguém cadastrado. Você
   pode descartar capturas individuais.
5. Clique em **Enrol**.

### Events

- Histórico de decisões (`GRANT`, `DENY`, `DENY_SPOOF`), com nome, similaridade,
  caixa e liveness.
- Filtro por decisão e botão para limpar o log.
- O mesmo log é gravado em `data/events.jsonl` (uma linha JSON por decisão).

### Settings

- Limiares de reconhecimento e de liveness, tamanho mínimo de rosto, janela de
  reutilização de embedding, resolução da câmera, etc.
- Só um conjunto restrito de campos é gravável, cada um com faixa validada no
  servidor. Alterações que exigem recarga aparecem com o aviso para reiniciar.

### Reiniciar serviço

O botão **Reiniciar serviço** libera câmera e NPU na thread principal e
reexecuta o processo; a página se reconecta sozinha. Se a interface tiver sido
atualizada, use **Ctrl+F5** para recarregar o JS/CSS.

## Desempenho medido nesta placa

Valores reais medidos no Orange Pi Zero 3W (A733), câmera 640x480 MJPG:

| Etapa | Tempo | Observação |
|---|---|---|
| Captura 640x480 MJPG | ~1 ms | JPEG decodificado no hardware |
| YuNet na NPU | **~2,4 ms por inferência** | grafo 640x640; ~14–18 ms por frame com pré/pós |
| YuNet na CPU (256px) | ~17 ms | fallback |
| SFace na NPU | **~2,5 ms por inferência** | 112x112, int8 |
| SFace na CPU | ~68 ms | fallback float32 |
| MiniFASNet (liveness) | ~10 ms | CPU, só em rostos que seriam liberados |

Na comparação com o caminho CPU: as caixas e scores da detecção NPU foram
**idênticos** numa imagem de teste; o embedding apresentou
`cosine(NPU, CPU) ≈ 0,98`.

> O `cosine ≈ 0,98` é a diferença esperada da quantização int8. O banco
> `data/facedb.json` continua válido, mas a margem de similaridade cai um pouco.
> Recadastrar a face com a NPU ativa recupera a maior margem.

## Sobre a NPU

A NPU só executa binários de rede Vivante (`.nb`), então os modelos ONNX são
convertidos com o **ACUITY** da Allwinner e compilados para o alvo `a733`
(`VIP9000NANODI_PLUS_PID0X1000003B` / `0x1000003B`).

Os dois grafos do A733 já estão no repositório:

| Arquivo | Função | Entrada | Saída |
|---|---|---|---|
| `models/npu/yunet.nb` | detecção | `1x3x640x640` uint8 (imagem) | 12 cabeças cruas (cls/obj/bbox/kps, strides 8/16/32) |
| `models/npu/sface_int8.nb` | embedding | `1x3x112x112` float16 | 128 valores (uint8 affine) |

O decodificador do YuNet (caixas, landmarks e NMS) é feito em Python
(`facegate/detector.py`).

O runtime (VIPLite) é carregado de `/usr/lib` (`libNBGlinker.so` +
`libVIPhal.so`) e usa `/dev/vipcore`. Os buffers são mapeados em memória host
alinhada, como no runtime de referência da Allwinner (`npulib`); os grafos são
preparados uma vez e reaproveitados entre frames.

Para recompilar os grafos (por exemplo, para o T527 ou para outra versão do
SFace), veja `tools/README-npu.md` — a conversão só roda em uma máquina x86_64
com a imagem Docker de ACUITY da Allwinner.

## Reutilização de embedding

`recognizer.reuse_window` (padrão `1.5 s`): o SFace só roda de novo quando o
rosto muda. Dois portões precisam passar:

- **`reuse_min_iou`** (0.5) — a caixa nova precisa sobrepor a anterior;
- **`reuse_min_appearance`** (0.9) — um descritor 16x16 em tons de cinza do
  recorte alinhado precisa ser próximo o suficiente do anterior.

O portão de aparência é o que importa para segurança: sem ele, outra pessoa que
entrasse na mesma caixa herdaria a identidade em cache. Separação medida:

```
mesma pessoa, deslocamento de 8px   cos=0.945
mesma pessoa, exposição -25%        cos=0.999
mesma pessoa, ruído do sensor       cos=0.988
negativo difícil (rosto espelhado)  cos=0.780
```

O padrão 0.9 fica no meio dessa faixa. O erro é propositalmente assimétrico: um
falso reaproveitamento libera acesso; um falso descarte custa apenas um novo
embedding.

A consulta ao banco **não** é cacheada: casar contra o banco acontece todo frame,
então cadastrar alguém ou mudar o limiar tem efeito imediato. Em frames
reaproveitados o HUD mostra `emb 0`; `/api/status` expõe `embed_reused` e
`embed_fresh`.

## Prova de vida (anti-spoofing)

O MiniFASNet V2 SE classifica cada rosto que seria liberado em *live* ou
*spoofed*. Resultados `DENY_SPOOF` aparecem em âmbar.

```yaml
access:
  require_liveness: true
  liveness_threshold: 0.3
```

**Limitação importante, sem rodeios:** o limiar **não foi validado contra
ataques reais** de foto impressa ou tela de celular. Testes com ataques
sintéticos mostraram que spoofs digitais realistas passam:

| Teste | Score | Veredito |
|---|---|---|
| Fotos de rosto genuíno | 0.50 – 0.90 | live (correto) |
| "Impressão" de baixo contraste | 0.02 | spoof (correto, mas falhou a detecção) |
| "Tela" superexposta | 0.47 | spoof (correto) |
| Replay com moiré | 0.86 | **live (ataque não detectado)** |
| JPEG de baixa qualidade | 0.94 | **live (não detectado)** |
| Reamostrado/upscalado | 0.63 | **live (não detectado)** |

Distribuição real do score num rosto genuíno: `min 0.244 | p1 0.289 |
p5 0.389 | mediana 0.822 | max 0.985`. Não existe limiar que seja ao mesmo tempo
livre de falsos `DENY_SPOOF` e acima do ataque real medido (0.47). O padrão 0.3
prioriza não trancar a pessoa para fora; aumente se o acesso for desassistido.
Para segurança de verdade, prefira desafio-resposta (virar a cabeça/piscar),
sensor de profundidade ou uma segunda câmera em outro ângulo.

## Configuração

Tudo fica em `config.yaml`; o que não estiver lá usa o padrão. Os ajustes mais
comuns:

- **`detector.use_npu` / `recognizer.use_npu`** — escolhe NPU ou CPU por função.
  Também ajustável pela interface web (aplicado no próximo reinício).
- **`recognizer.match_threshold`** (0.5) — cosseno mínimo para aceitar. Pessoas
  cadastradas ficam em 0.55–0.69; estranhos perto de zero.
- **`recognizer.reuse_window`** (1.5) — janela de reaproveitamento; `0` desliga.
- **`detector.input_size`** (256) — usado pelo caminho CPU. No caminho NPU o
  grafo é 640x640 e a imagem é redimensionada para ele.
- **`camera.usb_id`** (1b3f:2002) — identifica a webcam; use `camera.device` para
  fixar um `/dev/videoN`.

## Segurança

Não há autenticação na interface web. Quem alcançar a porta 8080 na rede pode
cadastrar/remover pessoas e mudar limiares. Isso é adequado para uma LAN
confiável e nada além. Para expor de outra forma, coloque atrás de um proxy com
autenticação ou use `web.host: 127.0.0.1` com túnel SSH.

O sistema **falha fechado**: rosto não reconhecido é `DENY`; rostos menores que
`min_face_size` (60 px) nunca são comparados.

## API HTTP

A interface web é um cliente fino destes endpoints (usáveis com `curl`):

| Método | Caminho | Função |
|---|---|---|
| GET | `/api/status` | uptime, fps, tempos, câmera, backends efetivos |
| GET | `/api/people` | pessoas cadastradas e contagem de embeddings |
| GET | `/api/events?limit=N` | eventos recentes |
| DELETE | `/api/events` | limpa o log |
| POST | `/api/capture` | captura um frame e devolve embedding + miniatura |
| POST | `/api/enroll` | `{"name":..., "features":[[128 floats], ...]}` |
| POST | `/api/people/remove` | `{"name":...}` |
| GET | `/api/config` | configuração efetiva |
| POST | `/api/config` | atualiza ajustes permitidos |
| POST | `/api/restart` | reinicia o serviço (recarrega modelos/grafos) |
| GET | `/api/snapshot` | frame anotado em JPEG |
| GET | `/stream` | vídeo MJPEG |

## Estrutura do projeto

```
facegate/
├── config.py        dataclasses + carregamento do YAML
├── camera.py        captura V4L2, resolução por USB id, reconexão
├── detector.py      YuNet (NPU VIPLite ou CPU) + decode das cabeças cruas
├── recognizer.py    SFace (NPU ou CPU) + banco de faces
├── npu.py           binding ctypes do VIPLite (buffers host alinhados)
├── tracking.py      reaproveitamento por caixa + aparência
├── liveness.py      MiniFASNet (anti-spoofing)
├── enroll.py        regras de cadastro (CLI e web)
├── service.py       fila de jobs, estado, escrita do config
├── pipeline.py      orquestração por frame e anotação
├── events.py        log JSONL + webhook opcional
├── web.py           API JSON + arquivos estáticos
├── static/          UI do operador (index.html, app.js, style.css)
└── tools/           notas de recompilação dos .nb + CLI de embedding
```

`bench.py` mede tempos reais: `./bench.sh -n 100 --sizes 224,256,320`.

## Solução de problemas

- **`NPU backend unavailable (...) - using CPU`** — confira `/dev/vipcore`, as
  libs em `/usr/lib` e o caminho em `viplite_dir`. O serviço continua em CPU.
- **`vip_create_network failed: -7`** — grafo `.nb` de outro alvo (por exemplo
  T527 `0x1000003B` vs `0x10000016`). Recompile para o alvo da placa.
- **`awtop` não mostra uso** — `awtop` lê
  `/sys/kernel/debug/viplite/core_loading`. Sem carga de NPU ele fica em 0%;
  com os grafos do A733 e NPU selecionada, ele mostra 2–3%.
- **Seleção volta para CPU** — atualize a página com **Ctrl+F5** (o JS antigo
  espelhava o backend efetivo). A seleção salva fica em `config.yaml`.
- **Reiniciar não volta** — atualize o código (`git pull`) e reinicie; o fluxo
  atual reexecuta o processo na thread principal e não segura câmera/NPU.
- Avisos `setPreferableTarget ... new graph engine` do OpenCV são esperados e
  inofensivos.

## Licença

Veja [LICENSE](LICENSE).
