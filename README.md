# Projeto 3 — Segmentação Semântica em Imagens Histológicas

## LIPAI — Laboratório de Inteligência Artificial e Processamento de Imagens

Este repositório contém os experimentos desenvolvidos no **Projeto 3 do LIPAI**, voltado à **segmentação semântica de imagens histológicas utilizando Deep Learning**.

O projeto investiga o comportamento de diferentes arquiteturas e estratégias de treinamento em dois problemas de segmentação, considerando fatores como arquitetura, inicialização dos pesos, função de perda, data augmentation e diferentes seeds.

O objetivo principal é realizar uma comparação experimental controlada entre as configurações propostas, avaliando tanto a **qualidade da segmentação** quanto aspectos relacionados ao **custo computacional**, estabilidade do treinamento e capacidade de generalização dos modelos.

---

## 1. Objetivos

Os principais objetivos do experimento são:

- comparar diferentes arquiteturas de segmentação semântica;
- avaliar treinamento **From Scratch (FS)** e **Pre-Trained All Layers (PT-ALL)**;
- comparar as funções de perda **Binary Cross-Entropy (BCE)** e **Dice Loss**;
- avaliar o impacto de **data augmentation**;
- analisar a estabilidade dos resultados utilizando diferentes seeds;
- comparar o desempenho nos dois datasets;
- analisar custo computacional por meio de número de parâmetros e GFLOPs;
- avaliar qualitativamente as máscaras produzidas;
- investigar possíveis casos de overfitting, underfitting, sobresegmentação e subsegmentação.

---

## 2. Datasets

O projeto utiliza dois conjuntos de dados histológicos.

### 2.1 Oral Epithelium — Displasia

O primeiro conjunto contém imagens histológicas do epitélio oral.

Neste dataset, a tarefa é tratada como um problema de **segmentação binária**, em que a classe de primeiro plano corresponde aos **núcleos celulares**.

Classes utilizadas:

| Valor | Classe |
|---|---|
| 0 | Fundo |
| 1 | Núcleo |

Os dados utilizados no experimento são organizados em conjuntos independentes de:

- treino;
- validação;
- teste.

A separação é definida pelos arquivos de *split* utilizados pelo projeto, garantindo que o conjunto de teste permaneça reservado para a avaliação final.

### 2.2 Tumor Tissue — Tecido Tumoral

O segundo conjunto contém imagens histológicas utilizadas para segmentação de **tecido tumoral**.

Também é utilizado como um problema de segmentação binária.

Classes utilizadas:

| Valor | Classe |
|---|---|
| 0 | Não tumor |
| 1 | Tumor |

Assim como no dataset de displasia, são utilizados conjuntos independentes de treino, validação e teste.

---

## 3. Pré-processamento das Imagens

Antes de serem fornecidas às redes neurais, imagens e máscaras passam por uma etapa de preparação para garantir compatibilidade com o tamanho de entrada utilizado pelos modelos.

Entre as etapas do pipeline estão:

- carregamento da imagem;
- carregamento da máscara correspondente;
- adequação ao tamanho de entrada da rede;
- normalização;
- aplicação opcional de data augmentation;
- conversão para tensores utilizados pelo PyTorch.

As transformações geométricas aplicadas simultaneamente à imagem e à máscara preservam o alinhamento espacial entre ambas.

---

## 4. Patching / Tiling e Redimensionamento

Um ponto importante da metodologia é a estratégia adotada para preparação espacial das imagens.

### 4.1 Estratégia adotada

Neste projeto, **não é realizado recorte adicional das imagens em uma grade de patches durante o pipeline de treinamento**.

Em vez disso, cada amostra é **redimensionada diretamente para o tamanho de entrada definido para a rede**.

Portanto, o fluxo utilizado é:

```text
Imagem original / patch fornecido pelo dataset
                  ↓
             Resize direto
                  ↓
        Tamanho de entrada da rede
                  ↓
               Modelo
```

e não:

```text
Imagem
   ↓
Divisão em grade
   ↓
Patch 1 + Patch 2 + ... + Patch N
   ↓
Modelo
```

### 4.2 Justificativa para o dataset tumoral

Essa decisão é particularmente importante no conjunto de **tecido tumoral**.

As imagens desse dataset já são disponibilizadas como regiões previamente recortadas de **640 × 640 pixels**, sem necessidade de produzir uma nova grade de patches com sobreposição.

Dessa forma, realizar um segundo processo de patching introduziria uma subdivisão adicional das amostras que não é necessária para o desenho experimental adotado.

Cada patch de 640 × 640 é, portanto, tratado como uma amostra individual e posteriormente redimensionado para o tamanho de entrada utilizado pela rede.

### 4.3 Ausência de sobreposição

Como não é realizado um novo processo de tiling, também **não é utilizada sobreposição entre patches no pipeline desenvolvido**.

Não há, portanto, necessidade de:

- definir stride entre patches;
- tratar regiões de sobreposição;
- combinar predições de patches vizinhos;
- reconstruir uma imagem maior a partir das predições.

Essa escolha simplifica o pipeline e mantém cada amostra fornecida pelo dataset como uma unidade independente de treinamento e avaliação.

### 4.4 Consequências da estratégia

O resize direto possui como principal vantagem a simplicidade e a padronização das entradas.

Entretanto, o redimensionamento pode alterar a escala espacial de estruturas presentes na imagem. Esse aspecto deve ser considerado principalmente na interpretação de estruturas pequenas e pode ser tratado como uma limitação metodológica do experimento.

---

## 5. Arquiteturas

O desenho experimental utiliza duas famílias de modelos.

### 5.1 U-Net com backbone

Foi utilizada uma arquitetura **U-Net com encoder ResNet18**.

Essa configuração permite avaliar duas estratégias distintas de inicialização e treinamento.

#### FS — From Scratch

Os pesos são inicializados sem utilizar pré-treinamento.

```text
U-Net + ResNet18
        ↓
Pesos inicializados do zero
        ↓
Treinamento no dataset histológico
```

#### PT-ALL — Pre-Trained All Layers

O encoder utiliza pesos pré-treinados e todas as camadas permanecem treináveis durante o treinamento no dataset histológico.

```text
ResNet18 pré-treinada
        ↓
U-Net
        ↓
Fine-tuning de todas as camadas
```

### 5.2 Sharp U-Net

A segunda arquitetura utilizada é a **Sharp U-Net**, arquitetura específica para segmentação.

No desenho experimental adotado, essa arquitetura é avaliada utilizando treinamento **From Scratch (FS)**.

---

## 6. Desenho Experimental

O experimento considera:

- **2 datasets**;
- **3 combinações de arquitetura/modo de treinamento**;
- **2 funções de perda**;
- **2 condições de augmentation**;
- **3 seeds**.

As combinações arquitetura/treinamento são:

1. Sharp U-Net + FS;
2. U-Net ResNet18 + FS;
3. U-Net ResNet18 + PT-ALL.

Para cada uma delas são avaliadas duas funções de perda:

- BCE;
- Dice Loss.

Também são avaliadas duas condições de augmentation:

- sem augmentation;
- com augmentation.

Além disso, cada configuração é repetida utilizando três seeds diferentes.

O desenho completo resulta em:

```text
2 datasets
×
3 configurações arquitetura/treinamento
×
2 losses
×
2 condições de augmentation
×
3 seeds

= 72 execuções
```

A utilização de múltiplas seeds permite analisar não apenas o melhor resultado isolado, mas também a **média e o desvio padrão** de cada configuração experimental.

---

## 7. Data Augmentation

O experimento compara explicitamente duas condições:

- **sem augmentation**;
- **com augmentation**.

No cenário sem augmentation, são aplicadas apenas as transformações necessárias para preparar as imagens para o modelo.

No cenário com augmentation, são utilizadas transformações adicionais com o objetivo de aumentar artificialmente a variabilidade dos dados de treinamento.

Quando uma transformação geométrica é aplicada à imagem, a mesma transformação é aplicada à máscara correspondente.

Esse cuidado evita desalinhamento entre imagem e ground truth.

A comparação permite avaliar experimentalmente se o aumento artificial da variabilidade dos dados melhora a capacidade de generalização dos modelos.

---

## 8. Funções de Perda

Duas funções de perda são avaliadas.

### Binary Cross-Entropy — BCE

A BCE mede o erro da classificação binária realizada para cada pixel.

É utilizada como uma das referências do experimento para comparar o efeito da função objetivo sobre a segmentação.

### Dice Loss

A Dice Loss é baseada na sobreposição entre a máscara predita e a máscara de referência.

Ela está diretamente relacionada ao Dice Score, métrica amplamente utilizada para avaliar segmentação.

A comparação entre BCE e Dice permite verificar se otimizar diretamente uma função relacionada à sobreposição produz vantagens para os datasets utilizados.

---

## 9. Treinamento

Todas as configurações seguem um pipeline de treinamento controlado.

Para cada execução são definidos:

- dataset;
- arquitetura;
- estratégia de treinamento;
- loss;
- augmentation;
- seed.

Durante o treinamento são acompanhadas informações como:

- training loss;
- validation loss;
- validation mDice;
- melhor época;
- melhor desempenho observado na validação.

O melhor checkpoint de cada execução é armazenado para posterior avaliação no conjunto de teste.

### Hiperparâmetros

Os valores abaixo devem refletir exatamente a configuração utilizada nos notebooks de treinamento.

| Parâmetro | Valor |
|---|---|
| Tamanho de entrada | **[preencher com o valor do notebook]** |
| Batch size | **[preencher]** |
| Número máximo de épocas | **[preencher]** |
| Otimizador | **[preencher]** |
| Learning rate | **[preencher]** |
| Threshold da máscara | **0.5** |
| Seeds | **[preencher as três seeds]** |

---

## 10. Seleção do Melhor Modelo

A seleção do melhor modelo não é realizada utilizando o conjunto de teste.

Durante o treinamento, o desempenho é acompanhado no conjunto de validação e o checkpoint correspondente ao melhor desempenho é armazenado.

Após o treinamento, o modelo selecionado é avaliado no conjunto de teste.

Essa separação reduz o risco de utilizar informações do conjunto de teste durante a escolha do modelo.

---

## 11. Métricas de Avaliação

As seguintes métricas são utilizadas:

### Dice Score

Mede a sobreposição entre a segmentação predita e o ground truth.

### Intersection over Union — IoU

Mede a razão entre a interseção e a união das regiões predita e real.

### mDice

Média do Dice entre as classes consideradas.

### mIoU

Média do IoU entre as classes.

### Precision

Avalia, entre os pixels classificados como primeiro plano, quantos realmente pertencem à classe positiva.

### Recall

Avalia quantos pixels positivos existentes no ground truth foram recuperados pelo modelo.

No projeto, a classe **foreground** representa:

- **núcleo**, no dataset de displasia;
- **tumor**, no dataset de tecido tumoral.

---

## 12. Matriz de Confusão

Além das métricas principais, a análise inclui **matrizes de confusão normalizadas por pixel** para os melhores modelos.

Para cada dataset, os pixels são classificados como:

```text
                 Predição

                0         1

Real 0          TN        FP

Real 1          FN        TP
```

No dataset de displasia:

```text
0 = Fundo
1 = Núcleo
```

No dataset tumoral:

```text
0 = Não Tumor
1 = Tumor
```

As matrizes são normalizadas pela classe real, facilitando a interpretação visual da proporção de acertos e erros independentemente da quantidade absoluta de pixels.

Essa análise auxilia principalmente na identificação de:

- falsos positivos;
- falsos negativos;
- tendência à sobresegmentação;
- tendência à subsegmentação.

---

## 13. Custo Computacional

Além da qualidade da segmentação, o projeto avalia o custo computacional das arquiteturas.

São analisados:

- número total de parâmetros;
- número de parâmetros treináveis;
- GFLOPs.

Os resultados permitem analisar a relação entre desempenho e custo computacional.

```text
Desempenho
    ↑
    │
    │       ● Modelo A
    │
    │   ● Modelo B
    │
    └────────────────────→ Custo computacional
```

Dessa forma, um modelo com mDice ligeiramente superior não é necessariamente a alternativa mais interessante caso exija um aumento muito elevado no custo computacional.

---

## 14. Análise das Curvas de Aprendizado

As curvas de treinamento são armazenadas individualmente para cada execução.

São analisadas:

- training loss;
- validation loss;
- validation mDice;
- época do melhor mDice;
- diferença final entre loss de treino e validação;
- queda de mDice após o melhor ponto.

Essas informações ajudam na investigação de comportamentos como:

### Possível overfitting

Quando o desempenho no treinamento continua melhorando enquanto a validação deixa de acompanhar essa evolução.

### Possível underfitting

Quando o modelo apresenta dificuldade de alcançar desempenho satisfatório tanto no treinamento quanto na validação.

---

## 15. Análise Qualitativa

Além das métricas numéricas, são produzidos mosaicos qualitativos contendo:

1. imagem histológica original;
2. ground truth;
3. máscara predita;
4. sobreposição entre referência e predição.

Exemplo conceitual:

```text
┌──────────────┬──────────────┬──────────────┬──────────────┐
│              │              │              │              │
│    Imagem    │ Ground Truth │   Predição   │   Overlay    │
│              │              │              │              │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

As amostras qualitativas permitem observar situações que não são completamente representadas pelas métricas agregadas, incluindo:

- boas segmentações;
- segmentações intermediárias;
- falhas severas;
- estruturas pequenas não detectadas;
- falsos positivos;
- sobresegmentação;
- subsegmentação.

---

## 16. Análise Estatística e Comparações

Os resultados das diferentes seeds são consolidados para permitir comparações utilizando:

```text
média ± desvio padrão
```

São realizadas comparações entre:

- arquiteturas;
- FS × PT-ALL;
- BCE × Dice;
- com augmentation × sem augmentation;
- Displasia × Tecido Tumoral.

Também é produzido um ranking das configurações utilizando o desempenho médio obtido nas diferentes seeds.

---

## 17. Estrutura dos Resultados

Os resultados experimentais são organizados de forma a manter separadas as métricas, curvas, figuras e checkpoints.

```text
Repositorio/
│
├── datasets/
│   ├── oral_epithelium/
│   └── tumor_tissue/
│
├── checkpoints/
│
├── notebooks/
│
└── results/
    │
    ├── resultados_displasia.csv
    ├── resultados_tumoral.csv
    ├── resultados_consolidados.csv
    ├── tabela_resumo.csv
    │
    ├── curves_json/
    │
    ├── plots_pdf/
    │
    └── qualitative/
```

### `curves_json/`

Contém os históricos de treinamento das execuções.

### `plots_pdf/`

Contém as figuras e gráficos produzidos durante a análise dos experimentos.

### `qualitative/`

Contém os mosaicos utilizados na avaliação qualitativa.

### `checkpoints/`

Contém os pesos correspondentes aos melhores modelos encontrados durante as execuções.

---

## 18. Resultados Consolidados

Cada execução é identificada por uma combinação única de:

- dataset;
- arquitetura;
- modo de treinamento;
- função de perda;
- augmentation;
- seed.

O arquivo consolidado contém **72 execuções**, correspondentes ao desenho experimental completo.

Entre as informações armazenadas estão:

- `run_id`;
- `dataset`;
- `model`;
- `training_mode`;
- `loss`;
- `augmentation`;
- `seed`;
- `dice_background_test`;
- `dice_foreground_test`;
- `mDice_test`;
- `iou_background_test`;
- `iou_foreground_test`;
- `mIoU_test`;
- `precision_foreground_test`;
- `recall_foreground_test`;
- `num_params`;
- `trainable_params`;
- `gflops`;
- `best_epoch`;
- `val_mDice_best`.

---

## 19. Reprodutibilidade

Para favorecer a reprodutibilidade dos experimentos:

- são utilizadas seeds explícitas;
- os splits de treino, validação e teste são previamente definidos;
- cada configuração recebe um `run_id`;
- os históricos de treinamento são armazenados em JSON;
- os melhores checkpoints são preservados;
- as métricas são exportadas em CSV;
- os resultados das diferentes seeds são analisados por média e desvio padrão.

Essa organização permite rastrear cada resultado até sua respectiva configuração experimental.

---

## 20. Limitações e Possíveis Melhorias Futuras

Algumas limitações devem ser consideradas na interpretação dos resultados:

- possível desbalanceamento entre fundo e primeiro plano;
- dificuldade de segmentação de estruturas pequenas;
- possibilidade de sobresegmentação e subsegmentação;
- número limitado de épocas;
- dependência do threshold utilizado na binarização;
- custo computacional das arquiteturas;
- influência do resize sobre a escala das estruturas;
- ausência de um novo processo de tiling/patching durante o treinamento;
- variabilidade associada ao conjunto de dados e às seeds utilizadas.

No caso específico do processamento espacial, o resize direto foi adotado em vez de um novo recorte em patches. Embora adequado ao desenho experimental utilizado — especialmente considerando que o dataset tumoral já é fornecido em patches de 640 × 640 —, estratégias alternativas de tiling podem produzir comportamentos diferentes.

Como extensões futuras, podem ser investigados:

- outras funções de perda;
- combinação BCE + Dice;
- Focal Loss;
- Tversky Loss;
- outras arquiteturas de segmentação;
- outros backbones;
- diferentes tamanhos de entrada;
- diferentes thresholds;
- estratégias alternativas de augmentation;
- treinamento por mais épocas;
- estratégias de early stopping;
- tiling com diferentes tamanhos de patch;
- patches com sobreposição;
- avaliação do efeito do resize sobre estruturas pequenas;
- técnicas específicas para desbalanceamento entre classes.

---

## Tecnologias Utilizadas

O projeto foi desenvolvido principalmente com:

- Python;
- PyTorch;
- Segmentation Models PyTorch;
- Albumentations;
- NumPy;
- Pandas;
- Matplotlib;
- Seaborn;
- Scikit-learn;
- Google Colab.

---

## Referências

- **Sharp U-Net** — arquitetura utilizada nos experimentos.
- **Segmentation Models PyTorch** — implementação da U-Net e backbone.
- **OralEpitheliumDB** — dataset utilizado para segmentação de núcleos.
- **Tumor Tissue Dataset** — dataset utilizado para segmentação de tecido tumoral.

As referências completas e os respectivos links podem ser encontrados na documentação oficial do Projeto 3 do LIPAI.
