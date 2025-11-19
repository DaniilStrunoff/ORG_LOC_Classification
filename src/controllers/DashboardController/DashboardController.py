import os
import requests
from typing import Any, Mapping, Sequence, Union, TypeAlias
from src.controllers.IController import IController


JSON: TypeAlias = Union[
    str,
    int,
    float,
    bool,
    None,
    Mapping[str, "JSON"],
    Sequence["JSON"],
]


class DashboardController(IController):
    def __init__(self) -> None:
        self.grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
        self.user = os.getenv("GRAFANA_TECH_USER_LOGIN", "")
        self.password = os.getenv("GRAFANA_TECH_USER_PASSWORD", "")
        self.ds_uid = os.getenv("GRAFANA_DS_UID", "")
        self.ds_name = os.getenv("GRAFANA_DS_NAME", "")
        if not (self.user and self.password):
            raise RuntimeError(
                "Provide GRAFANA_TECH_USER_LOGIN/GRAFANA_TECH_USER_PASSWORD"
            )

    def _req(self, method: str, path: str, **kw: Any) -> requests.Response:
        url = f"{self.grafana_url}{path}"
        headers = kw.pop("headers", {})
        headers["Content-Type"] = "application/json"
        auth = (self.user, self.password)
        return requests.request(
            method, url, headers=headers, auth=auth, timeout=20, **kw
        )

    def create_training_dashboard(
        self, run_id: int, title: str | None = None
    ) -> dict[str, Any]:
        title = title or f"Training Run {run_id}"
        xychart_sql = (
            "SELECT "
            "  epoch::double precision AS epoch, "
            "  train_loss::double precision AS train_loss, "
            "  val_loss::double precision   AS val_loss, "
            "  train_auc::double precision  AS train_auc, "
            "  val_auc::double precision    AS val_auc "
            f"FROM train_metrics WHERE run_id = {run_id} "
            "ORDER BY epoch"
        )

        train_loss_last_sql = (
            f"SELECT train_loss::double precision AS train_loss "
            f"FROM train_metrics WHERE run_id = {run_id} AND train_loss IS NOT NULL "
            "ORDER BY epoch DESC LIMIT 1"
        )
        val_loss_last_sql = (
            f"SELECT val_loss::double precision AS val_loss "
            f"FROM train_metrics WHERE run_id = {run_id} AND val_loss IS NOT NULL "
            "ORDER BY epoch DESC LIMIT 1"
        )
        train_auc_last_sql = (
            f"SELECT train_auc::double precision AS train_auc "
            f"FROM train_metrics WHERE run_id = {run_id} AND train_auc IS NOT NULL "
            "ORDER BY epoch DESC LIMIT 1"
        )
        val_auc_last_sql = (
            f"SELECT val_auc::double precision AS val_auc "
            f"FROM train_metrics WHERE run_id = {run_id} AND val_auc IS NOT NULL "
            "ORDER BY epoch DESC LIMIT 1"
        )

        ds = {"uid": self.ds_uid, "type": "grafana-postgresql-datasource"}
        panels: JSON = [
            {
                "datasource": ds,
                "type": "xychart",
                "title": f"Loss / AUC by Epoch (run_id={run_id})",
                "gridPos": {"h": 12, "w": 24, "x": 0, "y": 0},
                "targets": [
                    {
                        "datasource": ds,
                        "format": "table",
                        "rawSql": xychart_sql,
                        "refId": "A",
                    }
                ],
                "fieldConfig": {
                    "defaults": {},
                    "overrides": [
                        {
                            "matcher": {"id": "byName", "options": "train_loss"},
                            "properties": [
                                {"id": "custom.axisPlacement", "value": "left"},
                                {"id": "custom.axisLabel", "value": "Loss"},
                                {"id": "min", "value": 0},
                            ],
                        },
                        {
                            "matcher": {"id": "byName", "options": "val_loss"},
                            "properties": [
                                {"id": "custom.axisPlacement", "value": "left"},
                                {"id": "custom.axisLabel", "value": "Loss"},
                                {"id": "min", "value": 0},
                            ],
                        },
                        {
                            "matcher": {"id": "byName", "options": "train_auc"},
                            "properties": [
                                {"id": "custom.axisPlacement", "value": "right"},
                                {"id": "custom.axisLabel", "value": "AUC"},
                                {"id": "min", "value": 0},
                                {"id": "max", "value": 1},
                                {"id": "unit", "value": "percentunit"},
                            ],
                        },
                        {
                            "matcher": {"id": "byName", "options": "val_auc"},
                            "properties": [
                                {"id": "custom.axisPlacement", "value": "right"},
                                {"id": "custom.axisLabel", "value": "AUC"},
                                {"id": "min", "value": 0},
                                {"id": "max", "value": 1},
                                {"id": "unit", "value": "percentunit"},
                            ],
                        },
                    ],
                },
                "options": {
                    "xField": "epoch",
                    "yFields": ["train_loss", "val_loss", "train_auc", "val_auc"],
                    "seriesMapping": "table",
                    "series": [
                        {"name": "train_loss", "yAxis": "left"},
                        {"name": "val_loss", "yAxis": "left"},
                        {"name": "train_auc", "yAxis": "right"},
                        {"name": "val_auc", "yAxis": "right"},
                    ],
                    "legend": {
                        "displayMode": "list",
                        "placement": "right",
                        "showLegend": True,
                        "calcs": [],
                    },
                    "tooltip": {"mode": "single", "sort": "none"},
                },
            },
            {
                "datasource": ds,
                "type": "stat",
                "title": "Last Train Loss",
                "gridPos": {"h": 6, "w": 6, "x": 0, "y": 12},
                "targets": [
                    {
                        "datasource": ds,
                        "format": "table",
                        "rawSql": train_loss_last_sql,
                        "refId": "B",
                    }
                ],
                "options": {
                    "reduceOptions": {"calcs": ["lastNotNull"], "values": False}
                },
                "fieldConfig": {"defaults": {"min": 0}, "overrides": []},
            },
            {
                "datasource": ds,
                "type": "stat",
                "title": "Last Val Loss",
                "gridPos": {"h": 6, "w": 6, "x": 6, "y": 12},
                "targets": [
                    {
                        "datasource": ds,
                        "format": "table",
                        "rawSql": val_loss_last_sql,
                        "refId": "C",
                    }
                ],
                "options": {
                    "reduceOptions": {"calcs": ["lastNotNull"], "values": False}
                },
                "fieldConfig": {"defaults": {"min": 0}, "overrides": []},
            },
            {
                "datasource": ds,
                "type": "stat",
                "title": "Last Train AUC",
                "gridPos": {"h": 6, "w": 6, "x": 12, "y": 12},
                "targets": [
                    {
                        "datasource": ds,
                        "format": "table",
                        "rawSql": train_auc_last_sql,
                        "refId": "D",
                    }
                ],
                "options": {
                    "reduceOptions": {"calcs": ["lastNotNull"], "values": False}
                },
                "fieldConfig": {
                    "defaults": {"min": 0, "max": 1, "unit": "percentunit"},
                    "overrides": [],
                },
            },
            {
                "datasource": ds,
                "type": "stat",
                "title": "Last Val AUC",
                "gridPos": {"h": 6, "w": 6, "x": 18, "y": 12},
                "targets": [
                    {
                        "datasource": ds,
                        "format": "table",
                        "rawSql": val_auc_last_sql,
                        "refId": "E",
                    }
                ],
                "options": {
                    "reduceOptions": {"calcs": ["lastNotNull"], "values": False}
                },
                "fieldConfig": {
                    "defaults": {"min": 0, "max": 1, "unit": "percentunit"},
                    "overrides": [],
                },
            },
        ]

        payload: JSON = {
            "dashboard": {
                "id": None,
                "uid": None,
                "title": title,
                "timezone": "browser",
                "refresh": "5s",
                "time": {"from": "now-1h", "to": "now"},
                "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m", "5m"]},
                "panels": panels,
                "schemaVersion": 39,
                "version": 1,
            },
            "overwrite": True,
        }

        r = self._req("POST", "/api/dashboards/db", json=payload)
        r.raise_for_status()
        return r.json()
