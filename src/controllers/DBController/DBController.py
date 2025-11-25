import os
from datetime import datetime
from typing import Iterable, List, Tuple, Optional, Callable, cast

from sqlalchemy import (
    create_engine,
    Integer,
    BigInteger,
    String,
    Text,
    Float,
    select,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.exc import SQLAlchemyError

from src.models.Types import Label
from src.controllers.IController import IController


class Base(DeclarativeBase):
    pass


class TrainSample(Base):
    __tablename__ = "train_data"
    text: Mapped[str] = mapped_column(Text, primary_key=True)
    label: Mapped[str] = mapped_column(String(8), primary_key=True)


class ValSample(Base):
    __tablename__ = "val_data"
    text: Mapped[str] = mapped_column(Text, primary_key=True)
    label: Mapped[str] = mapped_column(String(8), primary_key=True)


class TestSample(Base):
    __tablename__ = "test_data"
    text: Mapped[str] = mapped_column(Text, primary_key=True)
    label: Mapped[str] = mapped_column(String(8), primary_key=True)


class TrainMetric(Base):
    __tablename__ = "train_metrics"
    run_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    epoch: Mapped[int] = mapped_column(Integer, primary_key=True)
    train_loss: Mapped[float | None] = mapped_column(Float)
    val_loss: Mapped[float | None] = mapped_column(Float)
    train_auc: Mapped[float | None] = mapped_column(Float)
    val_auc: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        Index("idx_train_metrics_run", "run_id"),
        Index("idx_train_metrics_epoch", "epoch"),
    )


class DBController(IController):
    def __init__(self) -> None:
        self._engine = None

    def _connect(self):
        if self._engine is not None:
            return
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        db = os.getenv("POSTGRES_DB", "orgloc")
        user = os.getenv("POSTGRES_USER", "orgloc")
        pwd = os.getenv("POSTGRES_PASSWORD", "orgloc")
        url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
        self._engine = create_engine(url, pool_pre_ping=True, future=True)

    def _session(self) -> Iterable[Session]:
        self._connect()
        s = Session(self._engine, expire_on_commit=False)
        try:
            yield s
        finally:
            s.close()

    def _safe(
        self,
        fn: Callable[
            [
                Session,
            ],
            List[tuple[str, Label]],
        ],
        default: Optional[List[tuple[str, Label]]] = None,
    ) -> List[tuple[str, Label]]:
        if default is None:
            default = []
        try:
            for s in self._session():
                return fn(s)
            return default
        except (SQLAlchemyError, Exception) as e:
            print(f"[DBController] _safe error: {type(e).__name__}: {e}")
            return default

    def get_train_dataset(self, limit: Optional[int] = None) -> List[tuple[str, Label]]:
        def _q(s: Session) -> List[tuple[str, Label]]:
            stmt = select(TrainSample.text, TrainSample.label)
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = s.execute(stmt).all()
            return [(str(t), cast(Label, str(l))) for (t, l) in rows]

        data = self._safe(_q)
        return data

    def get_val_dataset(self, limit: Optional[int] = None) -> List[tuple[str, Label]]:
        def _q(s: Session) -> List[tuple[str, Label]]:
            stmt = select(ValSample.text, ValSample.label)
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = s.execute(stmt).all()
            return [(str(t), cast(Label, str(l))) for (t, l) in rows]

        data = self._safe(_q)
        return data

    def get_test_dataset(self, limit: Optional[int] = None) -> List[tuple[str, Label]]:
        def _q(s: Session) -> List[tuple[str, Label]]:
            stmt = select(TestSample.text, TestSample.label)
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = s.execute(stmt).all()
            return [(str(t), cast(Label, str(l))) for (t, l) in rows]

        data = self._safe(_q)
        return data

    def get_datasets(
        self,
        train_limit: Optional[int] = None,
        val_limit: Optional[int] = None,
    ) -> Tuple[List[tuple[str, Label]], List[tuple[str, Label]]]:
        return self.get_train_dataset(train_limit), self.get_val_dataset(val_limit)

    def log_training_metric(
        self,
        run_id: int,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_auc: Optional[float] = None,
        val_auc: Optional[float] = None,
    ) -> None:
        def _q(s: Session) -> None:
            s.add(
                TrainMetric(
                    run_id=run_id,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_auc=train_auc,
                    val_auc=val_auc,
                )
            )
            s.commit()

        try:
            for s in self._session():
                return _q(s)
        except (SQLAlchemyError, Exception) as e:
            print(f"[DBController] _safe error: {type(e).__name__}: {e}")

    def create_training_run(self) -> int:
        return int(datetime.now().timestamp() * 1000)
