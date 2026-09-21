# GlobalBrico — classificação de emails

O código de produção está em `src/`. Os dados rotulados, notebooks, treino experimental e resultados antigos estão em `research/`. `tests/` contém o comparador atual e os seus testes.

```bash
source .GB/bin/activate
python -m pip install -r requirements.txt
python -m src.connection
python -m src.extraction --pasta INBOX --limite 100
python -m src.classify_emails
```

## Receber emails do Lovable por API

A API recebe um email em JSON, guarda-o em `data/extracted_emails/` e responde com
um objeto vazio. Não executa classificação.

Configura um token secreto diferente da password da conta de email:

```bash
export EMAIL_INGEST_TOKEN='substituir-por-um-token-longo-e-aleatorio'
export EMAIL_INGEST_OUTPUT_DIR='./data/extracted_emails'
uvicorn src.email_api:app --host 0.0.0.0 --port 8000
```

No Lovable, envia cada novo email para `POST /v1/emails` com
`Authorization: Bearer <token>` e `Content-Type: application/json`:

```json
{
  "id": "identificador-unico-do-email",
  "message_id": "<abc123@example.com>",
  "received_at": "2026-09-16T09:45:00Z",
  "subject": "Pedido de orçamento",
  "from": "cliente@example.com",
  "to": ["encomendas@globalbrico.pt"],
  "text": "Bom dia, gostaria de solicitar...",
  "html": "<p>Bom dia...</p>",
  "attachments": [
    {
      "filename": "pedido.pdf",
      "content_type": "application/pdf",
      "size_bytes": 123456,
      "url": "https://..."
    }
  ]
}
```

O endpoint responde com HTTP `202` e `{}`. O mesmo `id` pode ser reenviado sem
criar duplicados. `GET /health` permite verificar se o serviço está ativo.

O classificador de produção carrega `src/models/xlm_roberta_large_xnli_finetuned/`. Para experimentar com os emails já extraídos, indica `--input-dir research/extracted_emails`. A extração nova guarda JSON em `data/extracted_emails/`.
Cada execução da extração mostra quantos emails chegaram à pasta desde a última verificação. O ponto de comparação é guardado em `data/extracted_emails/.extraction_state`, por servidor, conta e pasta. `--limite` continua a controlar quantos emails são extraídos, independentemente da contagem de novos.

## Gestão e Anotação Incremental de Novos Emails (Excel)

Para classificar emails novos à medida que vão chegando (sem alterar a base histórica em `data/extracted_emails/`) e comparar com outra IA e com o Transformer:

```bash
# Execução padrão (lê data/incoming_emails/ e atualiza data/emails_classificacao.xlsx)
python -m src.sync_excel

# Cópia direta e automática de uma pasta externa + atualização do Excel
python -m src.sync_excel --source-dir /caminho/para/pasta_externa

# Sem executar o transformer automaticamente
python -m src.sync_excel --no-transformer
```

### Como funciona:
1. **Deteção Incremental**: Deteta apenas ficheiros `.json` novos e adiciona-os ao Excel (`data/emails_classificacao.xlsx`). Registos e anotações manuais já existentes são 100% preservados.
2. **Folha `Revisão`**:
   - `Label correta`: Lista suspensa (*dropdown*) com `Pedido de Informação`, `Pedido de Encomenda` e `SPAM` para anotação humana rápida e sem gralhas.
   - `Label Outra IA`: Registada a partir do JSON (se já existir no payload) ou para preenchimento manual da IA externa a comparar.
   - `Label Transformer` e `Score Transformer`: Previsão e probabilidade calculadas automaticamente pelo modelo fine-tuned de produção.
   - `Concordância Outra IA` e `Concordância Transformer`: Fórmulas automáticas de concordância (`Concorda`, `Diverge` ou `Pendente`).
3. **Compatibilidade com Fine-Tuning**: A folha `Revisão` é compatível com `src/finetuning.py` e `src/email_data.py`.

## Fine-tuning diário com cross-validation

Depois de adicionar os JSON a `data/globalbrico_emails/`, sincroniza o Excel, preenche
`Label correta` nos novos registos e executa:

```bash
source .GB/bin/activate
python -m src.sync_excel
# Preencher agora as novas células "Label correta" no Excel.
python -m src.finetuning
```

Só entram no Excel e no fine-tuning mensagens iniciais cujo campo `in_reply_to` está
vazio ou nulo. Respostas a conversas existentes são ignoradas.

Cada execução faz 5-fold cross-validation estratificada apenas pela label. Em cada
fold, cerca de 80% dos dados são usados para treino e 20% para teste; cada email é
avaliado fora do treino exatamente uma vez. Os cinco modelos recomeçam sempre em
`joeddav/xlm-roberta-large-xnli`. No fim, é treinado um sexto modelo com 100% dos
dados, que fica disponível para produção.

Cada execução cria `data/finetuning_results/<data-hora>/` com:

- `metrics.json`: métricas out-of-fold globais, por classe, por fold, matriz de
  confusão, parâmetros e dispersão entre folds;
- `predictions.csv`: uma previsão out-of-fold para cada email;
- `fold_metrics.csv`: métricas individuais dos cinco folds;
- `folds.json`: UIDs atribuídos a cada fold pela divisão estratificada.

O ficheiro `data/finetuning_history.csv` acumula uma linha por execução e
`data/finetuning_results/latest.json` aponta para o resultado mais recente. O
histórico inclui ainda a variação emparelhada de accuracy e macro-F1, calculada
apenas nos UIDs comuns à execução atual e à anterior; esta é a comparação mais
adequada quando a base de dados cresce entre execuções.



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
