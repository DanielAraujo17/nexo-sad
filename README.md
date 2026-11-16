# Nexo — SAD para Priorização de Chamados de Suporte de TI

Protótipo funcional de um **Sistema de Apoio à Decisão (SAD)** para apoiar o coordenador de suporte técnico a decidir quais chamados devem ser atendidos primeiro. Trabalho individual da disciplina **Sistemas de Apoio à Decisão**, Bacharelado em Sistemas de Informação — UFBA.

## 1. O que o sistema faz

O Nexo recebe chamados, estima a probabilidade de violação do SLA, calcula uma pontuação ponderada e recomenda a ordem de atendimento. O coordenador pode alterar os pesos dos critérios, registrar a resolução e consultar o histórico.

| Área | Função | Conteúdo |
|---|---|---|
| Central de decisão | Classificação + modelo de decisão | Fila recomendada, cadastro, detalhes, risco estimado, pontuação e ajuste dos pesos. |
| Histórico e análises | Sumarização + agrupamento | Indicadores dos chamados resolvidos, K-Means, perfis e últimos encerramentos. |
| Avaliação do modelo | Validação | Métricas da árvore, matriz de confusão, comparação de políticas e sensibilidade dos pesos. |

O diferencial em relação a um painel é a **recomendação de uma ordem de atendimento**, explicável por critérios e pesos configuráveis.

## 2. Ciclo de vida dos chamados e retenção do histórico

O sistema mantém uma única tabela `chamados`, com registros em dois estados:

- **Aberto:** ainda aguarda atendimento. Não possui tempo de resolução nem resultado do SLA.
- **Resolvido:** atendimento encerrado. O sistema registra o tempo decorrido e calcula se o SLA hipotético foi ultrapassado.

Na primeira inicialização de um banco vazio, são criados 200 chamados resolvidos sintéticos e 25 chamados abertos de demonstração. Essa carga inicial ocorre uma única vez; reiniciar a aplicação não recria nem substitui os registros existentes.

A política de retenção mantém os **200 chamados resolvidos mais recentes** (configurável por `HISTORICO_LIMITE`). Quando um novo chamado é encerrado, o sistema:

1. calcula o tempo entre abertura e encerramento;
2. compara esse tempo com o prazo de SLA associado à urgência;
3. preserva `idade_horas` como a idade observada na última avaliação/priorização da fila;
4. grava a duração completa em `tempo_resolucao_horas`, muda o estado para `resolvido` e registra a data de encerramento;
5. elimina apenas os chamados resolvidos mais antigos que excederem o limite.

A distinção é importante para evitar vazamento de informação: a idade usada como preditora é um retrato conhecido antes do encerramento; o tempo total e a violação do SLA são desfechos observados somente depois da resolução. Ao consultar a fila, o sistema atualiza o retrato da idade dos chamados abertos. Ao encerrar, não copia a duração final para a variável de idade.

A comparação com o SLA é feita usando a duração exata calculada no encerramento; apenas o valor exibido é arredondado para duas casas decimais. Assim, uma duração ligeiramente superior ao prazo não é indevidamente classificada como dentro do SLA por causa do arredondamento.

Existe uma migração opcional para bancos criados por versões antigas, nas quais `idade_horas` recebia a duração final. Ela **não roda por padrão**: é preciso definir `MIGRAR_IDADE_LEGADA=1` (ou chamar `inicializar(migrar_legado=True)`). Motivo: ligada em toda inicialização, ela também reescreveria snapshots legítimos (um chamado consultado na fila e encerrado segundos depois tem idade ≈ tempo total). O banco SQLite fornecido na entrega foi verificado e não possui registros nesse estado. Quando executada, a idade é redefinida para `0` (idade conhecida no cadastro); `tempo_resolucao_horas` e `violou_sla` são preservados.

**Chamados em aberto nunca são removidos pela política de retenção.** Eles permanecem na base até serem encerrados. A idade de um chamado aberto é recalculada a partir da data de abertura sempre que a fila é consultada.

O banco é persistente: localmente, usa SQLite no arquivo `instance/chamados.db`; em produção, pode usar PostgreSQL hospedado no Supabase. O aplicativo é executado no servidor Flask; o navegador acessa a interface, mas não recebe acesso direto ao banco.

## 3. Dados e funções analíticas

O ZIP entregue inclui um banco SQLite local já inicializado para demonstração (200 resolvidos + 25 abertos). Esse arquivo é ignorado pelo Git para não versionar bases locais; ao clonar o repositório sem o arquivo, a aplicação cria e popula o banco na primeira execução. A base de partida é sintética e reproduzível. As variáveis utilizadas nas análises são:

`id_chamado`, `categoria`, `urgencia`, `usuarios_afetados`, `criticidade`, `idade_horas`, `recorrencias`, `complexidade`, `tempo_resolucao_horas` e `violou_sla`.

O banco também armazena título, estado, data de abertura e data de resolução para suportar o fluxo operacional.

**Números do relatório.** Todos os valores citados no relatório (métricas, matriz de confusão, agrupamento, comparação de filas, sensibilidade e caso concreto) são recalculados por `python scripts/resultados_relatorio.py`, a partir da base sintética determinística em memória (não lê nem altera o banco local). `docs/resultados_referencia.json` guarda a saída de referência; `python scripts/resultados_relatorio.py --conferir docs/resultados_referencia.json` informa se algum valor mudou no seu ambiente. As métricas dependem das versões das bibliotecas, por isso `requirements.txt` fixa as versões testadas.

### Sumarização

Calcula quantidade de chamados resolvidos, taxa de violação do SLA, tempo médio de resolução e distribuição por categoria, urgência e complexidade.

### Agrupamento

Aplica K-Means sobre usuários afetados, criticidade, idade, recorrências e complexidade. Os grupos são perfis exploratórios, não categorias rígidas.

### Classificação

Uma Árvore de Decisão estima a probabilidade de violação do SLA. As entradas são informações disponíveis na avaliação do chamado. `tempo_resolucao_horas` e `violou_sla` são desfechos e não entram como preditores.

### Modelo de decisão

Cada chamado recebe nota entre 0 e 100 para urgência, criticidade, usuários afetados, idade em relação ao SLA, recorrências e risco estimado. A nota final é a soma das notas multiplicadas pelos pesos normalizados.

- **P1:** pontuação igual ou superior a 60.
- **P2:** pontuação igual ou superior a 40 e inferior a 60.
- **P3:** pontuação inferior a 40.

Os pesos podem ser ajustados na Central de decisão. O sistema destaca a quantidade de chamados que a equipe consegue iniciar no próximo ciclo. A decisão final permanece com o coordenador.

## 4. Tecnologias

- **Flask:** rotas e lógica web no servidor.
- **HTML, CSS, Bootstrap e JavaScript:** interface responsiva.
- **SQLAlchemy:** camada de persistência compatível com SQLite e PostgreSQL.
- **SQLite:** banco local para desenvolvimento.
- **PostgreSQL/Supabase:** banco persistente recomendado para a versão publicada.
- **Pandas, NumPy e scikit-learn:** manipulação e funções analíticas.
- **Plotly:** gráficos.
- **Gunicorn:** servidor WSGI para produção.
- **Render:** hospedagem da aplicação web.

## 5. Executar localmente no Windows

Requisitos: **Python 3.12** (versão em que os testes e os resultados do relatório foram conferidos; é também a usada no Render). Confira com `python --version`. Se o seu Python for anterior à 3.12, instale a 3.12 ou use `requirements-compat.txt` no lugar de `requirements.txt` (faixas de versão; as métricas podem variar levemente).

```powershell
# 1. Criar o ambiente virtual
python -m venv .venv

# 2. Instalar dependências usando diretamente o Python do ambiente
# (não é necessário executar Activate.ps1)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Criar o arquivo de configuração local
Copy-Item .env.example .env

# 4. Executar a aplicação
.\.venv\Scripts\python.exe app.py
```

Esse modo de execução também funciona quando a política do PowerShell bloqueia scripts de ativação do ambiente virtual.

Acesse `http://127.0.0.1:5000`. Sem `DATABASE_URL`, o aplicativo cria automaticamente `instance/chamados.db` e usa SQLite.

No Linux/macOS, ative o ambiente com `source .venv/bin/activate` e execute os mesmos comandos `pip` e `python app.py`.

## 6. Configurar Supabase (PostgreSQL)

1. Crie um projeto no Supabase.
2. No painel do projeto, abra **Connect** e copie a connection string de PostgreSQL.
3. Prefira a conexão direta para um backend persistente quando a hospedagem oferecer conectividade IPv6. Se a hospedagem for somente IPv4, use o **Session Pooler** indicado pelo próprio painel do Supabase.
4. Configure a variável `DATABASE_URL` no ambiente do aplicativo, substituindo o marcador de senha pela senha do banco. Não compartilhe nem versione essa URL.
5. Configure `FLASK_SECRET_KEY` com uma chave aleatória longa. A aplicação interrompe a inicialização em ambiente de produção se essa variável estiver ausente.
6. Mantenha `APP_ENV=production` e `HISTORICO_LIMITE=200`. Em produção, o cookie de sessão é marcado como seguro por padrão; em desenvolvimento local, essa opção fica desativada para permitir HTTP em `localhost`.
7. Na primeira execução com o banco vazio, o sistema cria a tabela e popula os registros sintéticos iniciais automaticamente.
8. Se a senha do banco tiver caracteres especiais (`@`, `:`, `/`, `#`, `%`), codifique-os na URL (por exemplo, `@` vira `%40`) ou gere uma senha apenas com letras e números.
9. Depois de publicar, abra `https://SEU-APP.onrender.com/health`: o JSON deve mostrar `"banco": "postgresql"`. Se mostrar `"sqlite"`, `DATABASE_URL` não está chegando à aplicação e os dados serão perdidos a cada reinício.
10. Projetos gratuitos do Supabase são **pausados após cerca de uma semana sem atividade** e deixam de responder até serem restaurados no painel. Acesse o sistema (ou o painel) pouco antes da avaliação para garantir que o banco esteja ativo.

O SQLAlchemy cria a tabela `chamados`. O esquema também está documentado em `docs/schema.sql`.

**Atenção:** a aplicação não inclui credenciais de banco. Cada pessoa que publicar o projeto precisa conectar seu próprio projeto Supabase e adicionar os segredos nas variáveis de ambiente da hospedagem.

## 7. Publicar no Render

O projeto inclui `render.yaml`. Também é possível configurar manualmente:

- **Runtime:** Python 3
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT app:app`
- **Health check path:** `/health`

No painel do Render, adicione as variáveis:

- `DATABASE_URL`: connection string PostgreSQL do Supabase.
- `FLASK_SECRET_KEY`: segredo longo e aleatório.
- `HISTORICO_LIMITE`: `200`.

O arquivo `render.yaml` gera automaticamente `FLASK_SECRET_KEY`, mas `DATABASE_URL` deve ser preenchida manualmente. Não coloque senhas no repositório.

Observações:

- `render.yaml` define `plan: free`. Sem essa linha, o Blueprint assume um plano pago. No plano gratuito o serviço "dorme" após cerca de 15 minutos sem acessos e a primeira requisição seguinte demora para responder; abra o link antes de gravar o vídeo ou de a avaliação começar. O disco do plano gratuito é efêmero, por isso o banco deve ser o Supabase.
- Em configuração manual (sem Blueprint), crie também a variável `PYTHON_VERSION` com o valor `3.12.7` e `APP_ENV` com o valor `production`.
- Em produção, a aplicação recusa iniciar sem `FLASK_SECRET_KEY` ou com o valor de exemplo do `.env.example`.

## 7.1 Reiniciar a demonstração

A idade dos chamados abertos é calculada em tempo real. Depois de alguns dias, os 25 chamados de demonstração ficam muito mais velhos que qualquer chamado do histórico (a idade máxima no histórico inicial é 81,5 h): o classificador passa a atribuir 100% de risco a todos e os critérios *idade* e *risco* deixam de diferenciar a fila. Antes de gravar o vídeo ou de a avaliação começar, restaure o cenário inicial:

```bash
python scripts/reiniciar_demo.py              # simulação: mostra o que seria feito, sem alterar nada
python scripts/reiniciar_demo.py --confirmar  # exporta backup em backups/ e recria os 200 + 25 registros
```

O script usa o mesmo banco da aplicação (`DATABASE_URL` ou `instance/chamados.db`). Em PostgreSQL ele exige digitar `REINICIAR`. Uma alternativa para a versão publicada é criar um novo projeto/banco Supabase vazio: a primeira inicialização recria a base inicial.

## 8. Testes

```bash
python -m pytest -q
```

A suíte tem 50 testes. Eles verificam a geração da base, ausência de vazamento de informação, modelo de pontuação, fila, criação e encerramento de chamados, persistência, regra de retenção, as páginas HTTP (inclusive com proteção CSRF ativa, como no navegador), o endpoint `/health`, as proteções de produção e os scripts auxiliares. Os testes usam SQLite em memória ou temporário, nunca o banco real, e não precisam de Supabase. Não há testes automatizados contra um PostgreSQL real: a conexão com o Supabase deve ser conferida manualmente em `/health`.

## 9. Estrutura do projeto

```text
sad-priorizacao-chamados/
├── app.py                  # Aplicação Flask e rotas
├── database.py             # Modelo SQLAlchemy, seed e retenção dos 200 resolvidos
├── config.py               # SLA, pesos, limites e parâmetros
├── gerar_dados.py          # Geração reproduzível dos dados sintéticos iniciais
├── modelo.py               # Sumarização, K-Means, árvore, decisão e validação
├── requirements.txt        # versões fixadas (Python 3.12)
├── requirements-compat.txt # faixas de versão, para Python anterior à 3.12
├── Procfile
├── render.yaml
├── .env.example
├── templates/              # Páginas HTML Jinja
├── static/css/style.css    # Identidade visual responsiva
├── scripts/
│   ├── resultados_relatorio.py  # recalcula os números do relatório (base determinística)
│   └── reiniciar_demo.py        # reinício seguro da demonstração, com backup
├── docs/
│   ├── diagrama_arquitetura.svg / .png
│   ├── resultados_referencia.json
│   └── schema.sql
├── instance/
│   └── chamados.db        # Banco SQLite local pré-carregado no ZIP (ignorado pelo Git)
└── tests/
    ├── test_modelo.py
    ├── test_database.py
    ├── test_app.py
    └── test_scripts.py
```

## 10. Limitações e premissas

- Os 200 registros históricos iniciais e os 25 chamados abertos iniciais são simulados; a validação não comprova eficácia em uma operação real.
- Chamados reais cadastrados e encerrados passam a compor o histórico. Para o cenário acadêmico, o modelo é reprocessado com os últimos 200 resolvidos após cada encerramento.
- O histórico tem retenção limitada: registros resolvidos mais antigos que o limite são eliminados. Chamados abertos não são eliminados.
- A idade do chamado aberto é calculada pela diferença entre o horário atual e a data de abertura e registrada como retrato no momento em que a fila é avaliada.
- O tempo de resolução é calculado automaticamente no encerramento. O resultado de SLA compara esse tempo com o prazo hipotético da urgência. A idade usada pelo classificador permanece sendo a última idade capturada antes do encerramento, e não a duração final.
- O rótulo `violou_sla` da base inicial é gerado pelas mesmas regras que definem as variáveis; a árvore, portanto, aprende o gerador. Parte do desempenho vem de uma regra trivial: os 12 chamados que já estavam acima do prazo no momento da avaliação violaram o SLA em 100% dos casos.
- A idade dos chamados abertos cresce em tempo real. Chamados com idade muito acima da faixa do histórico (máximo de 81,5 h) recebem risco de 100%, pois a árvore não extrapola; use `scripts/reiniciar_demo.py` antes de demonstrações (seção 7.1).
- Numa demonstração, chamados encerrados logo após o cadastro entram no histórico com tempo de resolução próximo de zero e sem violação, o que não é realista; cada encerramento também descarta o resolvido mais antigo da base. As métricas da página Avaliação podem, por isso, mudar depois de várias resoluções; os números do relatório referem-se à base inicial.
- O modelo apoia, mas não substitui, a decisão do responsável pelo suporte.
- O protótipo não implementa autenticação ou perfis de usuário. Como usa dados sintéticos, está preparado para demonstração acadêmica; não deve ser utilizado para operar chamados reais sem controles de acesso e segurança adicionais.
