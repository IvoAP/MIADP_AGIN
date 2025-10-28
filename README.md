# MIADP_AGIN

Ferramenta de anonimização adaptativa usando MIADP (Mutual Information Adaptive Differential Privacy) com ruído Laplace.

O objetivo é adicionar ruído proporcional à importância (informação mútua) de cada feature em relação à coluna alvo (target), mantendo utilidade dos dados enquanto protege privacidade.

## 📂 Estrutura

```
MIADP_AGIN/
├── data/              # Coloque seu dataset CSV bruto aqui (ignorado pelo git)
├── results/           # Saídas anonimizadas e relatórios (ignorado pelo git)
├── src/
│   ├── miadp_laplace.py  # Implementação do algoritmo
│   └── main.py           # CLI para anonimização
```

Coloque APENAS um arquivo `.csv` dentro de `data/` (excluindo arquivos já gerados). Se houver mais de um, use `--input` para escolher qual processar.

## 🚀 Instalação

Requer Python >= 3.13.

```powershell
pip install -e .
```

Caso não queira instalar, pode executar diretamente com Python:

```powershell
python src/main.py --help
```

## 🔑 Uso Básico

1. Coloque seu dataset em `data/` (ex: `data/heart.csv`).
2. Rode o comando informando somente a coluna target. Todas as outras colunas serão tratadas como features.

```powershell
python src/main.py --target HeartDisease --epsilon 1.0
```

Esse comando:
- Detecta automaticamente `data/heart.csv`.
- Cria `results/heart_anon.csv` como saída.
- Usa `epsilon = 1.0` (quanto menor o valor, mais ruído).

## 🛠 Opções Avançadas

| Opção | Obrigatória | Descrição |
|-------|-------------|-----------|
| `--target / -y` | Sim | Nome da coluna alvo. |
| `--epsilon / -e` | Sim | Epsilon global (>0). Menor => mais ruído. |
| `--input / -i` | Não | Caminho do CSV se houver vários em `data/`. |
| `--output / -o` | Não | Caminho de saída. Padrão: `<nome>_anon.csv` em `results/`. |
| `--task / -t` | Não | `classification`, `regression` ou `auto`. Padrão: auto. |
| `--sep` | Não | Separador do CSV (default `,`). |
| `--encoding` | Não | Encoding (default `utf-8`). |
| `--random-state` | Não | Seed para reprodutibilidade (default 42). |
| `--report` | Não | Salva relatório de MI e epsilons por feature. |

## 📈 Exemplo Completo

Gerando dataset anonimizado + relatório:

```powershell
python src/main.py --target HeartDisease --epsilon 0.7 --report data/relatorio_mi.csv
```

Saída gerada:
- `results/heart_anon.csv`
- `results/relatorio_mi.csv`

## 🧪 Como Funciona

1. Normaliza features numéricas para faixa [0,1].
2. Calcula informação mútua com a coluna target.
3. Converte MI em pesos (se MI ~ 0 usa pesos uniformes).
4. Distribui epsilon global entre as features proporcional aos pesos.
5. Aplica ruído Laplace por coluna com escala = 1/epsilon_i.
6. Reverte normalização e clipa valores dentro dos limites originais.

## ❗ Mensagens de Erro Comuns

- "Nenhum arquivo CSV encontrado": coloque o dataset em `data/`.
- "Mais de um CSV encontrado": use `--input` para escolher qual usar.
- "Coluna target 'X' não encontrada": verifique nome exato da coluna.
- "Não há colunas de features": dataset precisa ter pelo menos 1 coluna além do target.




Se precisar de ajuda com parâmetros ou interpretação do relatório, é só pedir.
