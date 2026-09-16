# GlobalBrico — classificação de emails

O código de produção está em `src/`. Os dados rotulados, notebooks, treino experimental e resultados antigos estão em `research/`. `tests/` contém o comparador atual e os seus testes.

```bash
source .GB/bin/activate
python -m pip install -r requirements.txt
python -m src.connection
python -m src.extraction --pasta INBOX --limite 100
python -m src.classify_emails
```

O classificador de produção carrega `src/models/xlm_roberta_large_xnli_finetuned/`. Para experimentar com os emails já extraídos, indica `--input-dir research/extracted_emails`. A extração nova guarda JSON em `data/extracted_emails/`.
Cada execução da extração mostra quantos emails chegaram à pasta desde a última verificação. O ponto de comparação é guardado em `data/extracted_emails/.extraction_state`, por servidor, conta e pasta. `--limite` continua a controlar quantos emails são extraídos, independentemente da contagem de novos.

## Comparar abordagens

O comparador avalia o checkpoint fine-tuned, `google/flan-t5-base` e um Ollama `qwen2.5:3b`. O Ollama tem de estar ativo e o modelo disponível localmente:

```bash
ollama serve
ollama pull qwen2.5:3b
python -m tests.compare_approaches
```

Os resultados são guardados numa pasta datada em `tests/results/`: `metrics.csv` para qualidade e latência, `predictions.csv` para rever cada UID, e `run.json` para o estado da execução. O tempo de carregamento e o aquecimento são medidos separadamente da latência por email. Os métodos recebem o mesmo texto, limitado a 2 000 caracteres, e não usam as labels humanas durante a inferência.

Por defeito, a comparação usa os 26 emails de `research/ground_truth/ground_truth_emails.xlsx` e `research/extracted_emails/`, que também foram usados para treinar o checkpoint. Os resultados mostram o desempenho **nesta amostra de treino**. Para comparar generalização, usa `--excel` e `--emails` com um conjunto novo e independente. O conjunto atual tem apenas um pedido de encomenda, pelo que o recall dessa classe é muito instável.

O checkpoint fine-tuned foi treinado a partir de `joeddav/xlm-roberta-large-xnli`. O FLAN-T5 e o Ollama são instruídos a escolher uma das três categorias, sem fine-tuning adicional. O FLAN-T5 gera o nome da categoria; o Ollama usa saída JSON estruturada. Respostas fora das categorias contam como incorretas e são registadas como `INVALIDO`.

Para comparar os cinco transformers do estudo inicial antes e depois do fine-tuning, abre `research/finetune_compare_transformers.ipynb`. O notebook usa a mesma divisão treino/validação para todos, guarda checkpoints e CSVs em `research/finetuning_comparison/` e mostra métricas, ganhos e previsões por UID. Sem fine-tuning, usa semelhança entre embeddings e descrições para os encoders e entailment para o XNLI, como no estudo inicial. Como só existe um Pedido de Encomenda rotulado, essa classe fica no treino e não pode ser avaliada na validação.
