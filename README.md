# CVRP com busca tabu

O repositório reúne experimentos de busca tabu para o Problema de Roteamento de Veículos Capacitados (CVRP).

As instâncias adicionais estão em [`benchmarks/general/`](benchmarks/general/) e [`benchmarks/selected/`](benchmarks/selected/). O material de consulta está em [`references/`](references/).

## Instalação

Na raiz do repositório, use o [uv](https://docs.astral.sh/uv/getting-started/installation/) para criar o ambiente com Python 3.12 e instalar o pacote.

```sh
uv sync --locked
```

## Execução

```sh
uv run cvrp --help
uv run cvrp baseline
uv run cvrp tabu-search
```

`baseline` executa o experimento original com inversões dentro de cada rota. `tabu-search` executa a variante com movimentos entre rotas, intensificação e diversificação. Por padrão, ambos usam a instância `A-n32-k5.vrp`, incluída em [`src/cvrp_tabu_search/data/`](src/cvrp_tabu_search/data/).

Cada execução grava seus CSVs e PNGs em uma subpasta com a data e a hora locais, como `results/2026-11-22_18-34-34/`. Se o nome já estiver em uso, a nova subpasta recebe um sufixo numérico.

Use `--output-dir` em qualquer um dos comandos para substituir a pasta base `results/`. O caminho pode ser absoluto ou relativo ao diretório de trabalho. As pastas são criadas automaticamente.

Use `--tabu-tenure-min` e `--tabu-tenure-max` para configurar o intervalo de duração tabu nos dois comandos. A cada movimento, a busca sorteia uma duração em iterações dentro do intervalo inclusivo, que por padrão vai de 3 a 7. Use dois valores iguais para fixar a duração ou dois zeros para desativar o bloqueio tabu.

Use `--max-iterations`, `--stagnation-limit` e `--time-limit` para ajustar os critérios de parada dos dois comandos. O limite de tempo é opcional, aceita segundos fracionários e vale para cada repetição, contando desde o início da construção da solução inicial. A busca confere o tempo antes de iniciar outra avaliação da vizinhança, então uma avaliação em andamento pode ultrapassar o limite. Com `--time-limit 0`, o experimento retorna a solução inicial. O CSV de execuções registra o valor na coluna `time_limit_s`.

Para usar outra instância CVRP com distâncias `EUC_2D`, informe o arquivo e seu custo ótimo de referência, que serve de alvo no relatório *time-to-target*.

```sh
uv run cvrp tabu-search --instance benchmarks/general/A-n33-k5.vrp --target-cost 661 --max-iterations 100 --stagnation-limit 40 --time-limit 2.5 --tabu-tenure-min 4 --tabu-tenure-max 8 --output-dir resultados/a-n33-k5
```

A CLI também pode ser executada com `uv run python -m cvrp_tabu_search`.

Os resultados usam os prefixos `runs_`, `history_`, `convergence_` e `ttt_`. A variante com movimentos entre rotas inclui `tabu_` antes do nome da instância.

### Calibração

Na raiz do repositório, execute a grade do trabalho nas instâncias `A-n32-k5`, `A-n80-k10` e `B-n78-k10`.

```sh
uv run cvrp calibrate
```

O comando cruza três durações tabu mínimas (`1`, `3`, `6`) com três máximas (`7`, `11`, `15`). São 27 execuções com a mesma semente (`1`) e um orçamento de 180 segundos por execução, cerca de 81 minutos de busca no total. A tabela `results/<data-hora>/calibration.csv` é preenchida conforme as execuções terminam e registra a instância, os parâmetros, o tempo e o melhor custo obtido.

Use `--jobs 4` para executar até quatro buscas em processos separados. O padrão é `--jobs 1`, e a coluna `jobs` registra o limite de processos da calibração. No modo paralelo, as linhas são gravadas conforme as buscas terminam. O orçamento conta tempo de relógio. Com processos disputando CPU, cada busca pode completar menos iterações durante o mesmo orçamento.

```sh
uv run cvrp calibrate --jobs 4
```

Para testar outros níveis, passe três valores distintos em cada opção. O maior mínimo deve ser menor ou igual ao menor máximo. `--time-limit` permite um teste curto e `--output-dir` escolhe a pasta dos resultados. Se executar fora da raiz, passe o caminho da pasta com `A-n80-k10.vrp` e `B-n78-k10.vrp` via `--benchmarks-dir`.

```sh
uv run cvrp calibrate --tabu-tenure-min 2 4 6 --tabu-tenure-max 8 12 16 --time-limit 0.1 --output-dir results/piloto
```

### Time-to-target

`cvrp ttt` executa `--runs` buscas independentes (sementes 1, 2, …). Cada uma para assim que encontra uma solução de custo menor ou igual ao ótimo informado em `--target-cost`, ou quando atinge `--time-limit` (180 s por padrão). Não há limite de iterações. O gráfico segue Aiex et al. (2007): o ponto i tem coordenadas (tᵢ, (i − ½)/n), com n igual ao total de execuções.

```sh
uv run cvrp ttt --instance benchmarks/general/B-n50-k7.vrp --target-cost 741 --runs 50 --time-limit 180 --tabu-tenure-min 1 --tabu-tenure-max 15 --jobs 4
```

### Tabelas e figuras do artigo

```sh
uv run cvrp report final resultados/finais --output-dir artigo/tabelas
uv run cvrp report calibration resultados/calibracao/calibration.csv --output-dir artigo/tabelas
uv run cvrp report convergence resultados/finais/<pasta>/history_tabu_B-n50-k7.csv --optimal-cost 741 --output artigo/figuras/convergencia.png
uv run cvrp report ablation resultados/ablacao --output-dir artigo/tabelas
uv run cvrp plot-solution --instance src/cvrp_tabu_search/data/A-n32-k5.vrp --target-cost 784 --tabu-tenure-min 1 --tabu-tenure-max 11 --output-dir artigo/figuras
```

Toda pasta de resultados inclui `environment.json` (Python, sistema, processador, memória e commit) e `solution_*.json`, com as rotas da melhor execução conferidas por um verificador independente.

## Desenvolvimento

O pacote vive em [`src/cvrp_tabu_search/`](src/cvrp_tabu_search/).

- [`search/`](src/cvrp_tabu_search/search/) contém os algoritmos e as operações de busca compartilhadas.
- [`experiments/`](src/cvrp_tabu_search/experiments/) executa os experimentos. O pacote [`experiments/reports/`](src/cvrp_tabu_search/experiments/reports/) coordena a geração dos relatórios e reúne os módulos de CSV, gráficos e diretórios de saída.
- `cli.py` recebe os argumentos. `instances.py` carrega os dados e `routes.py` reúne as operações de rota.

Os testes acompanham essa organização em `tests/search/`, `tests/experiments/` e `tests/cli/`. Em `tests/cli/`, os testes dos comandos e dos pontos de entrada ficam em arquivos próprios.

Execute os testes com pytest. Use Ruff para lint e formatação e ty para checagem de tipos.

```sh
uv run pytest
uv run ruff check
uv run ruff format --check
uv run ty check
```

## Distribuição

Use `uv build` para gerar o wheel e a distribuição de código-fonte em `dist/`.
