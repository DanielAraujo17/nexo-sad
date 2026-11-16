"""Testes de persistência e retenção usando um SQLite em memória."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
from database import Chamado, RepositorioChamados


def novo_repo(limite=cfg.N_HISTORICO):
    repo = RepositorioChamados("sqlite://", historico_limite=limite)
    repo.inicializar(popular_demo=True)
    return repo


def payload(titulo="Teste de suporte"):
    return {
        "titulo": titulo,
        "categoria": "Rede",
        "urgencia": "Média",
        "complexidade": "Baixa",
        "usuarios_afetados": 4,
        "criticidade": 3,
        "recorrencias": 0,
    }


def test_primeira_inicializacao_popula_historico_e_fila():
    repo = novo_repo()
    counts = repo.contagens()
    assert counts["resolvidos"] == cfg.N_HISTORICO
    assert counts["abertos"] == cfg.N_ABERTOS_INICIAIS
    assert len(repo.listar_resolvidos_df()) == cfg.N_HISTORICO
    assert repo.obter(1)["status"] == "resolvido"
    repo.fechar()


def test_reinicializacao_nao_duplica_dados():
    repo = novo_repo()
    antes = repo.contagens()
    repo.inicializar(popular_demo=True)
    depois = repo.contagens()
    assert antes == depois
    repo.fechar()


def test_novo_chamado_e_salvo_com_status_aberto_e_idade_zero():
    repo = novo_repo()
    chamado_id = repo.criar(payload())
    chamado = repo.obter(chamado_id)
    assert chamado is not None
    assert chamado["status"] == "aberto"
    assert chamado["idade_horas"] < 0.01
    assert chamado["tempo_resolucao_horas"] is None
    assert chamado["violou_sla"] is None
    repo.fechar()


def test_encerramento_calcula_tempo_sla_e_move_para_historico():
    repo = novo_repo()
    chamado_id = repo.criar(payload("Chamado para encerrar"))
    resultado = repo.resolver(chamado_id)
    assert resultado is not None
    assert resultado["status"] == "resolvido"
    assert resultado["tempo_resolucao_horas"] is not None
    assert resultado["violou_sla"] in (0, 1)
    assert repo.obter(chamado_id)["status"] == "resolvido"
    repo.fechar()


def test_retencao_remove_apenas_resolvidos_mais_antigos():
    repo = novo_repo(limite=cfg.N_HISTORICO)
    abertos_antes = repo.contagens()["abertos"]
    primeiro_id = repo.listar_historico_recente(cfg.N_HISTORICO)[-1]["id_chamado"]
    ids_novos = [repo.criar(payload(f"Chamado {n}")) for n in range(2)]
    for chamado_id in ids_novos:
        assert repo.resolver(chamado_id) is not None
    counts = repo.contagens()
    assert counts["resolvidos"] == cfg.N_HISTORICO
    assert counts["abertos"] == abertos_antes
    assert repo.obter(primeiro_id) is None  # o registro resolvido mais antigo foi descartado
    assert all(repo.obter(i)["status"] == "resolvido" for i in ids_novos)
    repo.fechar()


def test_chamado_aberto_nao_e_excluido_pela_retencao():
    repo = novo_repo()
    aberto_id = repo.criar(payload("Continua aberto"))
    for n in range(3):
        chamado_id = repo.criar(payload(f"Fechar {n}"))
        repo.resolver(chamado_id)
    chamado = repo.obter(aberto_id)
    assert chamado is not None and chamado["status"] == "aberto"
    repo.fechar()


def test_sqlite_em_arquivo_preserva_os_registros_apos_reabrir(tmp_path):
    arquivo = tmp_path / "persistencia.db"
    url = f"sqlite:///{arquivo.as_posix()}"
    repo1 = RepositorioChamados(url, historico_limite=cfg.N_HISTORICO)
    repo1.inicializar()
    novo_id = repo1.criar(payload("Persistência local"))
    repo1.fechar()

    repo2 = RepositorioChamados(url, historico_limite=cfg.N_HISTORICO)
    repo2.inicializar()
    chamado = repo2.obter(novo_id)
    assert chamado is not None
    assert chamado["titulo"] == "Persistência local"
    assert repo2.contagens()["abertos"] == cfg.N_ABERTOS_INICIAIS + 1
    repo2.fechar()


def test_avaliacao_registra_idade_sem_confundir_com_tempo_total():
    """A idade usada na previsão deve ser um retrato anterior ao encerramento."""
    repo = novo_repo()
    chamado_id = repo.criar({**payload("Snapshot antes da resolução"), "urgencia": "Alta"})

    agora = datetime.now(timezone.utc)
    abertura = agora - timedelta(hours=10)
    instante_avaliacao = abertura + timedelta(hours=4)
    with repo.Session() as session:
        chamado = session.get(Chamado, chamado_id)
        chamado.aberto_em = abertura
        session.commit()

    fila = repo.listar_abertos(agora=instante_avaliacao)
    linha = fila.loc[fila["id_chamado"] == chamado_id].iloc[0]
    assert linha["idade_horas"] == 4.0
    with repo.Session() as session:
        assert session.get(Chamado, chamado_id).idade_horas == 4.0

    resultado = repo.resolver(chamado_id)
    assert resultado is not None
    assert resultado["idade_horas"] == 4.0  # preserva o snapshot de decisão
    assert resultado["tempo_resolucao_horas"] >= 9.99  # duração real desde a abertura
    assert resultado["tempo_resolucao_horas"] > resultado["idade_horas"]
    assert resultado["violou_sla"] == 1  # urgência alta: SLA hipotético de 8h
    repo.fechar()


def test_migracao_corrige_registros_legados_com_idade_igual_ao_desfecho():
    """Versões antigas gravavam a duração final também em idade_horas."""
    repo = novo_repo()
    chamado_id = repo.criar(payload("Registro legado"))
    agora = datetime.now(timezone.utc)
    with repo.Session() as session:
        chamado = session.get(Chamado, chamado_id)
        chamado.status = "resolvido"
        chamado.aberto_em = agora - timedelta(hours=5)
        chamado.resolvido_em = agora
        chamado.idade_horas = 5.0
        chamado.tempo_resolucao_horas = 5.0
        chamado.violou_sla = 0
        session.commit()

    repo.inicializar(popular_demo=False, migrar_legado=True)
    corrigido = repo.obter(chamado_id)
    assert corrigido["idade_horas"] == 0.0
    assert corrigido["tempo_resolucao_horas"] == 5.0
    assert corrigido["violou_sla"] == 0
    repo.fechar()


def test_migracao_legada_nao_roda_sozinha_nem_altera_snapshot_legitimo(monkeypatch):
    """Chamado consultado na fila e encerrado logo depois tem idade ~ tempo total, e isso é legítimo."""
    monkeypatch.delenv("MIGRAR_IDADE_LEGADA", raising=False)
    repo = novo_repo()
    chamado_id = repo.criar(payload("Consultado e encerrado em seguida"))
    repo.listar_abertos()  # registra o snapshot de idade
    resultado = repo.resolver(chamado_id)
    idade_antes = repo.obter(chamado_id)["idade_horas"]
    repo.inicializar(popular_demo=False)  # reinício da aplicação, sem opt-in
    assert repo.obter(chamado_id)["idade_horas"] == idade_antes
    assert resultado["status"] == "resolvido"
    repo.fechar()


def test_postgresql_usa_driver_psycopg_sem_prepared_statements_e_sem_conectar():
    repo = RepositorioChamados("postgresql://usuario:senha@localhost:5432/postgres")
    assert repo.dialeto == "postgresql"
    assert repo.engine.url.drivername == "postgresql+psycopg"
    repo.fechar()
    repo = RepositorioChamados("postgres://usuario:senha@localhost:5432/postgres")  # prefixo antigo
    assert repo.engine.url.drivername == "postgresql+psycopg"
    repo.fechar()


def test_violacao_sla_usa_duracao_exata_antes_do_arredondamento():
    """Uma duração pouco acima do SLA não pode virar 'dentro do prazo' ao arredondar."""
    repo = novo_repo()
    chamado_id = repo.criar({**payload("Limite de SLA"), "urgencia": "Alta"})
    instante = datetime.now(timezone.utc)
    with repo.Session() as session:
        chamado = session.get(Chamado, chamado_id)
        chamado.aberto_em = instante - timedelta(hours=8.004)
        session.commit()

    resultado = repo.resolver(chamado_id, agora=instante)
    assert resultado is not None
    assert resultado["tempo_resolucao_horas"] == 8.0
    assert resultado["violou_sla"] == 1
    repo.fechar()


def test_duracao_exatamente_igual_ao_sla_nao_e_violacao():
    """A regra do projeto considera violação somente quando o prazo é excedido."""
    repo = novo_repo()
    chamado_id = repo.criar({**payload("Exatamente no SLA"), "urgencia": "Alta"})
    instante = datetime.now(timezone.utc)
    with repo.Session() as session:
        chamado = session.get(Chamado, chamado_id)
        chamado.aberto_em = instante - timedelta(hours=8)
        session.commit()

    resultado = repo.resolver(chamado_id, agora=instante)
    assert resultado is not None
    assert resultado["violou_sla"] == 0
    repo.fechar()
