import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC


FEATURE_COLS = [
    "raw_turbidity",
    "raw_temperature",
    "floc_count",
    "max_floc_area",
    "min_floc_area",
    "floc_density",
]

CLASS_NAMES = ["normal(0)", "excessive(1)", "insufficient(2)"]


def composite_metric(y_true, y_pred):
    """综合评分：兼顾总体准确率、宏平均F1和平衡准确率。"""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return (acc + macro_f1 + bal_acc) / 3.0


COMPOSITE_SCORER = make_scorer(composite_metric)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FlocMLPNet(nn.Module):
    def __init__(self, input_dim=6, hidden1=32, hidden2=16, dropout=0.2, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FlocCNN1DNet(nn.Module):
    def __init__(self, input_len=6, c1=16, c2=32, dropout=0.2, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c2, num_classes),
        )

    def forward(self, x):
        # x: [B, 6] -> [B, 1, 6]
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


def load_data(train_csv: str, val_csv: str, test_csv: str):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    x_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values

    x_val = val_df[FEATURE_COLS].values
    y_val = val_df["label"].values

    x_test = test_df[FEATURE_COLS].values
    y_test = test_df["label"].values
    return x_train, y_train, x_val, y_val, x_test, y_test


def build_models(random_state: int = 42):
    models = {}

    # 1) SVM (RBF)
    models["SVM_RBF"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    C=3.0,
                    gamma="scale",
                    kernel="rbf",
                    class_weight={0: 1.4, 1: 1.0, 2: 1.0},
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # 2) RandomForest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight={0: 1.4, 1: 1.0, 2: 1.0},
        random_state=random_state,
        n_jobs=-1,
    )

    # 3) LightGBM (optional)
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight={0: 1.4, 1: 1.0, 2: 1.0},
            random_state=random_state,
        )
    except ImportError:
        warnings.warn("LightGBM 未安装，已跳过。可执行: pip install lightgbm")

    # 4) XGBoost (optional)
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
            tree_method="hist",
        )
    except ImportError:
        warnings.warn("XGBoost 未安装，已跳过。可执行: pip install xgboost")

    # 5) CatBoost (optional)
    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=700,
            learning_rate=0.03,
            depth=6,
            loss_function="MultiClass",
            verbose=False,
            random_seed=random_state,
            class_weights=(1.4, 1.0, 1.0),
        )
    except ImportError:
        warnings.warn("CatBoost 未安装，已跳过。可执行: pip install catboost")

    return models


def get_deep_configs(model_name):
    if model_name == "MLP_Torch":
        return [
            {"hidden1": 32, "hidden2": 16, "dropout": 0.2, "lr": 1e-3, "batch_size": 64, "weight_decay": 1e-4},
            {"hidden1": 64, "hidden2": 32, "dropout": 0.2, "lr": 8e-4, "batch_size": 64, "weight_decay": 1e-4},
            {"hidden1": 64, "hidden2": 16, "dropout": 0.3, "lr": 1e-3, "batch_size": 128, "weight_decay": 5e-4},
        ]

    if model_name == "CNN1D_Torch":
        return [
            {"c1": 16, "c2": 32, "dropout": 0.2, "lr": 1e-3, "batch_size": 64, "weight_decay": 1e-4},
            {"c1": 32, "c2": 64, "dropout": 0.2, "lr": 8e-4, "batch_size": 64, "weight_decay": 1e-4},
            {"c1": 16, "c2": 64, "dropout": 0.3, "lr": 1e-3, "batch_size": 128, "weight_decay": 5e-4},
        ]

    return []


def build_deep_model(model_name, cfg):
    if model_name == "MLP_Torch":
        return FlocMLPNet(
            input_dim=6,
            hidden1=cfg["hidden1"],
            hidden2=cfg["hidden2"],
            dropout=cfg["dropout"],
            num_classes=3,
        )
    if model_name == "CNN1D_Torch":
        return FlocCNN1DNet(
            input_len=6,
            c1=cfg["c1"],
            c2=cfg["c2"],
            dropout=cfg["dropout"],
            num_classes=3,
        )
    raise ValueError(f"未知深度学习模型: {model_name}")


def train_deep_once(model_name, cfg, x_train, y_train, x_val, y_val, epochs, device, seed):
    set_seed(seed)
    model = build_deep_model(model_name, cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_x_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    best_comp = -1.0
    best_epoch = 1
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(val_x_tensor).argmax(dim=1).cpu().numpy()
        comp = composite_metric(y_val, pred)

        if comp > best_comp:
            best_comp = comp
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_comp, best_epoch, best_state


def fit_predict_deep_model(
    model_name,
    x_train,
    y_train,
    x_val,
    y_val,
    x_tune,
    y_tune,
    x_test,
    seed,
    tune=True,
    deep_epochs=25,
):
    # 深度学习统一使用标准化特征
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_tune_s = scaler.transform(x_tune)
    x_test_s = scaler.transform(x_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_candidates = get_deep_configs(model_name)
    if not cfg_candidates:
        raise RuntimeError(f"{model_name} 没有可用配置")

    if not tune:
        best_cfg = cfg_candidates[0]
        best_epoch = deep_epochs
    else:
        best_cv = -1.0
        best_cfg = cfg_candidates[0]
        best_epoch = deep_epochs
        for idx, cfg in enumerate(cfg_candidates):
            cv_score, epoch_found, _ = train_deep_once(
                model_name,
                cfg,
                x_train_s,
                y_train,
                x_val_s,
                y_val,
                epochs=deep_epochs,
                device=device,
                seed=seed + idx,
            )
            if cv_score > best_cv:
                best_cv = cv_score
                best_cfg = cfg
                best_epoch = epoch_found

    # 用选中的配置在 train+val 上重训，再测 test
    set_seed(seed)
    final_model = build_deep_model(model_name, best_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=best_cfg["lr"],
        weight_decay=best_cfg["weight_decay"],
    )

    tune_ds = TensorDataset(
        torch.tensor(x_tune_s, dtype=torch.float32),
        torch.tensor(y_tune, dtype=torch.long),
    )
    tune_loader = DataLoader(tune_ds, batch_size=best_cfg["batch_size"], shuffle=True)

    for _ in range(best_epoch):
        final_model.train()
        for xb, yb in tune_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = final_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    final_model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test_s, dtype=torch.float32, device=device)
        preds = final_model(x_test_tensor).argmax(dim=1).cpu().numpy()

    best_params = dict(best_cfg)
    best_params["epochs"] = int(best_epoch)
    return preds, best_params


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    comp = composite_metric(y_true, y_pred)
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "composite_score": comp,
        "recall_0_normal": recall_per_class[0],
        "recall_1_excessive": recall_per_class[1],
        "recall_2_insufficient": recall_per_class[2],
        "confusion_matrix": cm,
    }


def print_single_model_report(name, y_true, y_pred, metrics):
    print("=" * 78)
    print(f"模型: {name}")
    print("-" * 78)
    print(
        f"Composite: {metrics['composite_score']:.4f} | Accuracy: {metrics['accuracy']:.4f} | "
        f"Macro-F1: {metrics['macro_f1']:.4f} | Balanced-Acc: {metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Recall(excessive=1): {metrics['recall_1_excessive']:.4f} | "
        f"Recall(insufficient=2): {metrics['recall_2_insufficient']:.4f}"
    )
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print("混淆矩阵 (行=真实, 列=预测):")
    print(metrics["confusion_matrix"])


def get_param_distributions(model_name):
    if model_name == "SVM_RBF":
        return {
            "clf__C": [1.0, 3.0, 8.0, 15.0],
            "clf__gamma": ["scale", 0.1, 0.03, 0.01],
            "clf__class_weight": [
                "balanced",
                {0: 1.2, 1: 1.0, 2: 1.0},
                {0: 1.0, 1: 1.0, 2: 1.0},
            ],
        }

    if model_name == "RandomForest":
        return {
            "n_estimators": [300, 500, 700],
            "max_depth": [None, 10, 16, 24],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [
                "balanced",
                {0: 1.2, 1: 1.0, 2: 1.0},
                {0: 1.0, 1: 1.0, 2: 1.0},
            ],
        }

    if model_name == "LightGBM":
        return {
            "n_estimators": [300, 600, 900],
            "learning_rate": [0.01, 0.03, 0.05],
            "num_leaves": [31, 63, 127],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "class_weight": [
                "balanced",
                {0: 1.2, 1: 1.0, 2: 1.0},
                {0: 1.0, 1: 1.0, 2: 1.0},
            ],
        }

    if model_name == "XGBoost":
        return {
            "n_estimators": [400, 700, 1000],
            "learning_rate": [0.01, 0.03, 0.05],
            "max_depth": [4, 5, 7],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }

    if model_name == "CatBoost":
        return {
            "iterations": [400, 700, 1000],
            "learning_rate": [0.01, 0.03, 0.05],
            "depth": [5, 6, 8],
            "class_weights": [
                (1.2, 1.0, 1.0),
                (1.0, 1.0, 1.0),
                (1.4, 1.0, 1.0),
            ],
        }

    return None


def tune_model(model_name, model, x_tune, y_tune, random_state=42, n_iter=16, cv_folds=3):
    param_distributions = get_param_distributions(model_name)
    if not param_distributions:
        return model, None, None

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=COMPOSITE_SCORER,
        n_jobs=-1,
        cv=cv,
        random_state=random_state,
        refit=True,
        verbose=0,
    )
    try:
        search.fit(x_tune, y_tune)
        return search.best_estimator_, search.best_params_, search.best_score_
    except RuntimeError as ex:
        warnings.warn(f"{model_name} 调参失败，回退到默认参数训练。原因: {ex}")
        model.fit(x_tune, y_tune)
        return model, None, None


def train_and_compare(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    random_state: int = 42,
    tune: bool = True,
    n_iter: int = 16,
    cv_folds: int = 3,
    deep_epochs: int = 25,
):
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(train_csv, val_csv, test_csv)

    # 调参阶段：train + val 统一做交叉验证，目标是综合三类表现。
    x_tune = np.concatenate([x_train, x_val], axis=0)
    y_tune = np.concatenate([y_train, y_val], axis=0)

    models = build_models(random_state=random_state)
    if not models:
        raise RuntimeError("没有可用模型。请先安装至少一个算法库后重试。")

    deep_model_names = ["MLP_Torch", "CNN1D_Torch"]

    all_results = []
    for name, model in models.items():
        print(f"\n正在训练: {name}")
        best_params = None
        best_cv_score = None

        if tune:
            model, best_params, best_cv_score = tune_model(
                name,
                model,
                x_tune,
                y_tune,
                random_state=random_state,
                n_iter=n_iter,
                cv_folds=cv_folds,
            )
            if best_cv_score is not None:
                print(f"调参完成 | CV最佳综合分: {best_cv_score:.4f}")
            else:
                print("调参失败，已回退默认参数")
            print(f"最佳参数: {best_params}")
        else:
            model.fit(x_tune, y_tune)

        preds = model.predict(x_test)

        metrics = evaluate(y_test, preds)
        print_single_model_report(name, y_test, preds, metrics)

        all_results.append(
            {
                "model": name,
                "composite_score": metrics["composite_score"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "recall_0_normal": metrics["recall_0_normal"],
                "recall_1_excessive": metrics["recall_1_excessive"],
                "recall_2_insufficient": metrics["recall_2_insufficient"],
                "cv_best_composite": best_cv_score if best_cv_score is not None else np.nan,
                "best_params": str(best_params) if best_params is not None else "-",
            }
        )

    for deep_name in deep_model_names:
        print(f"\n正在训练: {deep_name}")
        preds, best_params = fit_predict_deep_model(
            deep_name,
            x_train,
            y_train,
            x_val,
            y_val,
            x_tune,
            y_tune,
            x_test,
            seed=random_state,
            tune=tune,
            deep_epochs=deep_epochs,
        )
        metrics = evaluate(y_test, preds)
        print("调参完成 | 验证集最优配置已用于重训")
        print(f"最佳参数: {best_params}")
        print_single_model_report(deep_name, y_test, preds, metrics)

        all_results.append(
            {
                "model": deep_name,
                "composite_score": metrics["composite_score"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "recall_0_normal": metrics["recall_0_normal"],
                "recall_1_excessive": metrics["recall_1_excessive"],
                "recall_2_insufficient": metrics["recall_2_insufficient"],
                "cv_best_composite": np.nan,
                "best_params": str(best_params),
            }
        )

    result_df = pd.DataFrame(all_results)
    result_df.sort_values(
        by=["composite_score", "macro_f1", "balanced_accuracy", "accuracy"],
        ascending=False,
        inplace=True,
    )

    print("\n" + "#" * 78)
    print("最终排名（优先: Composite -> Macro-F1 -> Balanced-Acc -> Accuracy）")
    print("#" * 78)
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_path = os.path.join(os.path.dirname(test_csv), "model_compare_results.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n结果已保存: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="多模型对比：以全类别综合指标进行自动调参")
    parser.add_argument(
        "--train_csv",
        type=str,
        default=r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_splits\train_set.csv",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_splits\val_set.csv",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=r"C:\Users\94508\Desktop\zds\graduation_design\experiment\dataset_splits\test_set.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_tune", action="store_true", help="关闭调参，仅用默认参数训练")
    parser.add_argument("--n_iter", type=int, default=16, help="每个模型的随机搜索次数")
    parser.add_argument("--cv_folds", type=int, default=3, help="调参交叉验证折数")
    parser.add_argument("--deep_epochs", type=int, default=25, help="深度学习模型训练轮数")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_compare(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        random_state=args.seed,
        tune=not args.no_tune,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        deep_epochs=args.deep_epochs,
    )
