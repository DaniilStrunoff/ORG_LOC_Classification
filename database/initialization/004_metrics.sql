CREATE TABLE IF NOT EXISTS train_metrics (
  run_id BIGINT NOT NULL,
  epoch  INT    NOT NULL,

  train_loss DOUBLE PRECISION,
  val_loss   DOUBLE PRECISION,
  train_auc  DOUBLE PRECISION,
  val_auc    DOUBLE PRECISION,

  PRIMARY KEY (run_id, epoch)
);
