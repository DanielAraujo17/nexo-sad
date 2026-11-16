"""Persistência dos chamados em SQLite (local) ou PostgreSQL (Supabase/Render).

Uma única tabela armazena chamados abertos e resolvidos. O histórico mantém os
N chamados resolvidos mais recentes; chamados abertos nunca são removidos pela
política de retenção.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import DateTime, Float, Integer, String, create_engine, select, delete, func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

import config as cfg
from gerar_dados import gerar_abertos, gerar_chamados


class Base(DeclarativeBase):
    pass


class Chamado(Base):
    __tablename__ = "chamados"

    id_chamado: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    titulo: Mapped[str] = mapped_column(String(180), nullable=False)
    categoria: Mapped[str] = mapped_column(String(40), nullable=False)
    urgencia: Mapped[str] = mapped_column(String(20), nullable=False)
    usuarios_afetados: Mapped[int] = mapped_column(Integer, nullable=False)
    criticidade: Mapped[int] = mapped_column(Integer, nullable=False)
    # Idade conhecida na última avaliação/priorização enquanto o chamado estava aberto.
    # Não deve ser substituída pela duração total quando o chamado for encerrado.
    idade_horas: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    recorrencias: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    complexidade: Mapped[str] = mapped_column(String(20), nullable=False)
    tempo_resolucao_horas: Mapped[float | None] = mapped_column(Float, nullable=True)
    violou_sla: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    aberto_em: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    resolvido_em: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)


class RepositorioChamados:
    """Camada pequena de acesso ao banco, compatível com SQLite e PostgreSQL."""

    def __init__(self, database_url: str | None = None, historico_limite: int | None = None):
        self.database_url = database_url or os.getenv("DATABASE_URL") or self._sqlite_default_url()
        self.historico_limite = max(1, int(historico_limite or os.getenv("HISTORICO_LIMITE", cfg.N_HISTORICO)))
        url = self._normalizar_url(self.database_url)
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if url.endswith(":memory:") or url == "sqlite://":
                kwargs["poolclass"] = StaticPool
        elif url.startswith("postgresql"):
            # Desativa prepared statements do psycopg 3. Sem isso, poolers em modo
            # transação (ex.: porta 6543 do Supabase) podem gerar erros intermitentes.
            kwargs["connect_args"] = {"prepare_threshold": None}
        self.engine: Engine = create_engine(url, **kwargs)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    @property
    def dialeto(self) -> str:
        """Nome do banco em uso ('sqlite' ou 'postgresql'); nunca expõe a URL nem credenciais."""
        return self.engine.dialect.name

    @staticmethod
    def _sqlite_default_url() -> str:
        pasta = Path(__file__).resolve().parent / "instance"
        pasta.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(pasta / 'chamados.db').as_posix()}"

    @staticmethod
    def _normalizar_url(url: str) -> str:
        # O driver psycopg 3 é usado no PostgreSQL. URLs de Supabase geralmente
        # começam com postgresql://; SQLAlchemy recebe postgresql+psycopg://.
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+psycopg://", 1)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    def inicializar(self, popular_demo: bool = True, migrar_legado: bool | None = None) -> None:
        """Cria a tabela, popula a demonstração (só se vazia) e, opcionalmente, migra legado.

        ``migrar_legado`` é opt-in (padrão: variável MIGRAR_IDADE_LEGADA, desligada). A
        correção só serve a bancos criados por versões antigas e, ligada em toda
        inicialização, também reescreveria snapshots legítimos de idade (por exemplo,
        chamado consultado na fila e encerrado segundos depois).
        """
        if migrar_legado is None:
            migrar_legado = os.getenv("MIGRAR_IDADE_LEGADA", "").strip().lower() in {"1", "true", "yes", "on", "sim"}
        Base.metadata.create_all(self.engine)
        with self.Session() as session:
            total = session.scalar(select(func.count()).select_from(Chamado)) or 0
            if popular_demo and total == 0:
                self._popular_demo(session)
                session.flush()

            if migrar_legado:
                # Migração para registros criados por versões anteriores: o encerramento
                # gravava a duração final também em idade_horas. Como não há como recuperar
                # a idade observada na decisão, usamos 0 (idade conhecida no cadastro) em
                # vez de manter um valor que revela o desfecho.
                self._corrigir_idade_pos_encerramento_legada(session)
            session.commit()

    @staticmethod
    def _corrigir_idade_pos_encerramento_legada(session) -> int:
        registros = session.scalars(
            select(Chamado).where(
                Chamado.status == "resolvido",
                Chamado.tempo_resolucao_horas.is_not(None),
                Chamado.tempo_resolucao_horas > 0.01,
                func.abs(Chamado.idade_horas - Chamado.tempo_resolucao_horas) < 0.005,
            )
        ).all()
        for chamado in registros:
            chamado.idade_horas = 0.0
        return len(registros)

    def _popular_demo(self, session) -> None:
        """Cria histórico sintético inicial e fila aberta apenas no primeiro uso."""
        agora = datetime.now(timezone.utc)
        historico = gerar_chamados(n=self.historico_limite)
        rows: list[Chamado] = []
        total = len(historico)
        for idx, row in historico.iterrows():
            # Datas ilustrativas distribuídas ao longo dos meses anteriores.
            resolvido_em = agora - timedelta(days=(total - int(idx)) * 1.0)
            tempo = float(row["tempo_resolucao_horas"])
            aberto_em = resolvido_em - timedelta(hours=tempo)
            rows.append(Chamado(
                titulo=f"Chamado histórico de {str(row['categoria']).lower()}",
                categoria=str(row["categoria"]),
                urgencia=str(row["urgencia"]),
                usuarios_afetados=int(row["usuarios_afetados"]),
                criticidade=int(row["criticidade"]),
                idade_horas=float(row["idade_horas"]),
                recorrencias=int(row["recorrencias"]),
                complexidade=str(row["complexidade"]),
                tempo_resolucao_horas=tempo,
                violou_sla=int(row["violou_sla"]),
                status="resolvido",
                aberto_em=aberto_em,
                resolvido_em=resolvido_em,
            ))
        session.add_all(rows)
        session.flush()

        abertos = gerar_abertos(n=cfg.N_ABERTOS_INICIAIS)
        for row in abertos.to_dict(orient="records"):
            idade = float(row["idade_horas"])
            session.add(Chamado(
                titulo=str(row["titulo"]),
                categoria=str(row["categoria"]),
                urgencia=str(row["urgencia"]),
                usuarios_afetados=int(row["usuarios_afetados"]),
                criticidade=int(row["criticidade"]),
                idade_horas=idade,
                recorrencias=int(row["recorrencias"]),
                complexidade=str(row["complexidade"]),
                tempo_resolucao_horas=None,
                violou_sla=None,
                status="aberto",
                aberto_em=agora - timedelta(hours=idade),
                resolvido_em=None,
            ))

    @staticmethod
    def _utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _to_dict(self, chamado: Chamado, agora: datetime | None = None) -> dict[str, Any]:
        idade = chamado.idade_horas
        if chamado.status == "aberto":
            agora = agora or datetime.now(timezone.utc)
            idade = max(0.0, (agora - self._utc(chamado.aberto_em)).total_seconds() / 3600)
        return {
            "id_chamado": chamado.id_chamado,
            "titulo": chamado.titulo,
            "categoria": chamado.categoria,
            "urgencia": chamado.urgencia,
            "usuarios_afetados": chamado.usuarios_afetados,
            "criticidade": chamado.criticidade,
            "idade_horas": round(float(idade), 2),
            "recorrencias": chamado.recorrencias,
            "complexidade": chamado.complexidade,
            "tempo_resolucao_horas": chamado.tempo_resolucao_horas,
            "violou_sla": chamado.violou_sla,
            "status": chamado.status,
            "aberto_em": chamado.aberto_em,
            "resolvido_em": chamado.resolvido_em,
        }

    def listar_abertos(self, agora: datetime | None = None) -> pd.DataFrame:
        """Lista chamados abertos e registra a idade observada nesta avaliação.

        A idade é uma variável disponível no momento em que a fila é analisada.
        Guardamos esse retrato para que, no encerramento, o histórico preserve a
        idade conhecida na última priorização, separada do tempo total até resolver.
        O parâmetro ``agora`` facilita testes determinísticos.
        """
        momento = self._utc(agora or datetime.now(timezone.utc))
        with self.Session() as session:
            chamados = session.scalars(
                select(Chamado).where(Chamado.status == "aberto").order_by(Chamado.aberto_em.asc())
            ).all()
            rows = []
            for chamado in chamados:
                idade_observada = max(
                    0.0,
                    (momento - self._utc(chamado.aberto_em)).total_seconds() / 3600,
                )
                chamado.idade_horas = round(idade_observada, 2)
                rows.append(self._to_dict(chamado, agora=momento))
            session.commit()
        cols = ["id_chamado", "titulo", "categoria", "urgencia", "usuarios_afetados",
                "criticidade", "idade_horas", "recorrencias", "complexidade"]
        return pd.DataFrame(rows, columns=cols)

    def listar_resolvidos_df(self) -> pd.DataFrame:
        """Devolve os N resolvidos mais recentes, em ordem cronológica crescente."""
        with self.Session() as session:
            chamados = session.scalars(
                select(Chamado)
                .where(Chamado.status == "resolvido")
                .order_by(Chamado.resolvido_em.desc(), Chamado.id_chamado.desc())
                .limit(self.historico_limite)
            ).all()
            rows = [self._to_dict(c) for c in reversed(chamados)]
        columns = ["id_chamado", "categoria", "urgencia", "usuarios_afetados", "criticidade",
                   "idade_horas", "recorrencias", "complexidade", "tempo_resolucao_horas", "violou_sla"]
        return pd.DataFrame([{k: r[k] for k in columns} for r in rows], columns=columns)

    def listar_historico_recente(self, limite: int | None = None) -> list[dict[str, Any]]:
        with self.Session() as session:
            chamados = session.scalars(
                select(Chamado).where(Chamado.status == "resolvido")
                .order_by(Chamado.resolvido_em.desc(), Chamado.id_chamado.desc())
                .limit(limite or self.historico_limite)
            ).all()
            return [self._to_dict(c) for c in chamados]

    def obter(self, id_chamado: int) -> dict[str, Any] | None:
        with self.Session() as session:
            chamado = session.get(Chamado, int(id_chamado))
            return self._to_dict(chamado) if chamado else None

    def criar(self, dados: dict[str, Any]) -> int:
        categoria = str(dados["categoria"])
        urgencia = str(dados["urgencia"])
        complexidade = str(dados["complexidade"])
        if categoria not in cfg.CATEGORIAS:
            raise ValueError("Categoria inválida.")
        if urgencia not in cfg.ORDEM_URGENCIA:
            raise ValueError("Urgência inválida.")
        if complexidade not in cfg.ORDEM_COMPLEXIDADE:
            raise ValueError("Complexidade inválida.")
        agora = datetime.now(timezone.utc)
        chamado = Chamado(
            titulo=(str(dados.get("titulo", "")).strip()[:180] or f"Solicitação de {categoria.lower()}"),
            categoria=categoria,
            urgencia=urgencia,
            usuarios_afetados=max(1, min(200, int(dados["usuarios_afetados"]))),
            criticidade=max(1, min(5, int(dados["criticidade"]))),
            idade_horas=0.0,
            recorrencias=max(0, min(10, int(dados["recorrencias"]))),
            complexidade=complexidade,
            tempo_resolucao_horas=None,
            violou_sla=None,
            status="aberto",
            aberto_em=agora,
            resolvido_em=None,
        )
        with self.Session() as session:
            session.add(chamado)
            session.commit()
            return int(chamado.id_chamado)

    def resolver(self, id_chamado: int, agora: datetime | None = None) -> dict[str, Any] | None:
        """Encerra um chamado sem sobrescrever a idade registrada na última avaliação.

        ``tempo_resolucao_horas`` mede abertura -> encerramento e é usado para
        calcular o rótulo de SLA. ``idade_horas`` permanece como o retrato da
        idade observado antes do desfecho, evitando vazamento de informação.
        """
        agora = self._utc(agora or datetime.now(timezone.utc))
        with self.Session() as session:
            chamado = session.get(Chamado, int(id_chamado))
            if chamado is None or chamado.status != "aberto":
                return None
            duracao_exata = max(0.0, (agora - self._utc(chamado.aberto_em)).total_seconds() / 3600)
            duracao_registrada = round(duracao_exata, 2)
            # Preserva idade_horas: é a idade registrada na última avaliação da fila.
            # A duração completa fica exclusivamente em tempo_resolucao_horas.
            chamado.tempo_resolucao_horas = duracao_registrada
            # A comparação usa a duração sem arredondamento para não ocultar
            # violações muito próximas do limite do SLA (ex.: 8,004 h para SLA de 8 h).
            chamado.violou_sla = int(duracao_exata > cfg.SLA_HORAS[chamado.urgencia])
            chamado.status = "resolvido"
            chamado.resolvido_em = agora
            session.flush()

            # Retém somente os N resolvidos mais recentes. Chamados abertos não são tocados.
            ids_manter = select(Chamado.id_chamado).where(Chamado.status == "resolvido") \
                .order_by(Chamado.resolvido_em.desc(), Chamado.id_chamado.desc()) \
                .limit(self.historico_limite)
            session.execute(
                delete(Chamado).where(
                    Chamado.status == "resolvido",
                    Chamado.id_chamado.not_in(ids_manter),
                )
            )
            session.commit()
            return self._to_dict(chamado, agora=agora)

    def contagens(self) -> dict[str, int]:
        with self.Session() as session:
            abertos = session.scalar(select(func.count()).select_from(Chamado).where(Chamado.status == "aberto")) or 0
            resolvidos = session.scalar(select(func.count()).select_from(Chamado).where(Chamado.status == "resolvido")) or 0
        return {"abertos": int(abertos), "resolvidos": int(resolvidos), "historico_limite": self.historico_limite}

    def fechar(self) -> None:
        self.engine.dispose()
