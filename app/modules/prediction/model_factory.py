"""Model factory with Strategy pattern for multiple ML algorithms."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder

from app.modules.prediction.schemas import FeatureImpact

# Lazy imports for ML libraries to avoid startup issues with missing deps (e.g., libomp)
if TYPE_CHECKING:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression


class OutcomePredictor(ABC):
    """Abstract base class for call outcome prediction models."""

    def __init__(self) -> None:
        self.model: Any = None
        self.feature_names: list[str] = []
        self.label_encoder: LabelEncoder = LabelEncoder()
        self._categorical_cols: list[str] = [
            "agent_id",
            "org_id",
            "time_of_day",
            "day_of_week",
            "call_purpose",
        ]
        self._label_encoders: dict[str, LabelEncoder] = {}

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying ML model."""
        ...

    def train(self, X: pl.DataFrame, y: pl.Series) -> None:
        """
        Train the model on feature data.

        Args:
            X: Feature DataFrame (polars).
            y: Target labels (polars Series).
        """
        self.feature_names = X.columns
        self.model = self._create_model()

        # Encode categorical columns
        X_encoded = self._encode_features(X, fit=True)

        # Encode target labels
        y_np = self.label_encoder.fit_transform(y.to_numpy())

        # Train model
        self.model.fit(X_encoded, y_np)

    def predict(self, X: pl.DataFrame) -> tuple[list[str], list[float]]:
        """
        Predict outcomes for feature data.

        Args:
            X: Feature DataFrame (polars).

        Returns:
            Tuple of (predicted labels, probability of positive class).
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_encoded = self._encode_features(X, fit=False)

        # Get predictions
        y_pred = self.model.predict(X_encoded)
        labels = self.label_encoder.inverse_transform(y_pred).tolist()

        # Get probabilities (probability of failure/negative outcome)
        probas = self.model.predict_proba(X_encoded)
        # Assume class 0 is 'failure' - adjust based on label encoding
        failure_idx = 0 if "failure" in self.label_encoder.classes_[:1] else 1
        failure_probs = probas[:, failure_idx].tolist()

        return labels, failure_probs

    @abstractmethod
    def get_top_factors(
        self,
        X_row: dict[str, Any],
        n: int = 3,
    ) -> list[FeatureImpact]:
        """
        Get top features driving prediction for a single sample.

        Args:
            X_row: Single row of features as dict.
            n: Number of top factors to return.

        Returns:
            List of FeatureImpact with feature name, impact, and value.
        """
        ...

    def _encode_features(self, X: pl.DataFrame, fit: bool = False) -> np.ndarray:
        """Encode categorical features to numeric."""
        X_pd = X.to_pandas()

        for col in self._categorical_cols:
            if col in X_pd.columns:
                if fit:
                    self._label_encoders[col] = LabelEncoder()
                    X_pd[col] = self._label_encoders[col].fit_transform(
                        X_pd[col].astype(str)
                    )
                else:
                    # Handle unseen categories gracefully
                    le = self._label_encoders.get(col)
                    if le is not None:
                        X_pd[col] = X_pd[col].astype(str).map(
                            lambda x, le=le: (
                                le.transform([x])[0]
                                if x in le.classes_
                                else -1
                            )
                        )

        return X_pd.values.astype(np.float64)

    def save(self, path: Path) -> None:
        """Save model and encoders to disk."""
        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "label_encoder": self.label_encoder,
            "label_encoders": self._label_encoders,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, path)

    def load(self, path: Path) -> None:
        """Load model and encoders from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.label_encoder = data["label_encoder"]
        self._label_encoders = data["label_encoders"]


class LogisticRegressionAdapter(OutcomePredictor):
    """Logistic Regression baseline model."""

    def _create_model(self) -> "LogisticRegression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )

    def get_top_factors(
        self,
        X_row: dict[str, Any],
        n: int = 3,
    ) -> list[FeatureImpact]:
        """Get top factors using coefficient weights."""
        if self.model is None:
            return []

        coefs = self.model.coef_[0]
        impacts = []

        for i, fname in enumerate(self.feature_names):
            value = X_row.get(fname, 0.0)
            if isinstance(value, str):
                # Encode categorical value
                le = self._label_encoders.get(fname)
                value = le.transform([value])[0] if le and value in le.classes_ else 0.0
            impact = float(coefs[i]) * float(value)
            impacts.append(
                FeatureImpact(feature=fname, impact=impact, value=float(value))
            )

        # Sort by absolute impact and return top n
        impacts.sort(key=lambda x: abs(x.impact), reverse=True)
        return impacts[:n]


class RandomForestAdapter(OutcomePredictor):
    """Random Forest robust default model."""

    def _create_model(self) -> "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

    def get_top_factors(
        self,
        X_row: dict[str, Any],
        n: int = 3,
    ) -> list[FeatureImpact]:
        """Get top factors using feature importances."""
        if self.model is None:
            return []

        importances = self.model.feature_importances_
        impacts = []

        for i, fname in enumerate(self.feature_names):
            value = X_row.get(fname, 0.0)
            if isinstance(value, str):
                le = self._label_encoders.get(fname)
                value = le.transform([value])[0] if le and value in le.classes_ else 0.0
            impacts.append(
                FeatureImpact(
                    feature=fname,
                    impact=float(importances[i]),
                    value=float(value),
                )
            )

        impacts.sort(key=lambda x: abs(x.impact), reverse=True)
        return impacts[:n]


class CatBoostAdapter(OutcomePredictor):
    """CatBoost model with native categorical handling."""

    def _create_model(self) -> "CatBoostClassifier":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False,
            auto_class_weights="Balanced",
        )

    def get_top_factors(
        self,
        X_row: dict[str, Any],
        n: int = 3,
    ) -> list[FeatureImpact]:
        """Get top factors using feature importances."""
        if self.model is None:
            return []

        importances = self.model.get_feature_importance()
        impacts = []

        for i, fname in enumerate(self.feature_names):
            value = X_row.get(fname, 0.0)
            if isinstance(value, str):
                le = self._label_encoders.get(fname)
                value = le.transform([value])[0] if le and value in le.classes_ else 0.0
            impacts.append(
                FeatureImpact(
                    feature=fname,
                    impact=float(importances[i]),
                    value=float(value),
                )
            )

        impacts.sort(key=lambda x: abs(x.impact), reverse=True)
        return impacts[:n]


class LightGBMAdapter(OutcomePredictor):
    """LightGBM for fastest real-time inference."""

    def _create_model(self) -> "LGBMClassifier":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            class_weight="balanced",
            verbose=-1,
        )

    def get_top_factors(
        self,
        X_row: dict[str, Any],
        n: int = 3,
    ) -> list[FeatureImpact]:
        """Get top factors using feature importances."""
        if self.model is None:
            return []

        importances = self.model.feature_importances_
        impacts = []

        for i, fname in enumerate(self.feature_names):
            value = X_row.get(fname, 0.0)
            if isinstance(value, str):
                le = self._label_encoders.get(fname)
                value = le.transform([value])[0] if le and value in le.classes_ else 0.0
            impacts.append(
                FeatureImpact(
                    feature=fname,
                    impact=float(importances[i]),
                    value=float(value),
                )
            )

        impacts.sort(key=lambda x: abs(x.impact), reverse=True)
        return impacts[:n]


class ModelFactory:
    """Factory for creating outcome prediction models."""

    _algorithms: dict[str, type[OutcomePredictor]] = {
        "lr": LogisticRegressionAdapter,
        "logistic": LogisticRegressionAdapter,
        "rf": RandomForestAdapter,
        "random_forest": RandomForestAdapter,
        "catboost": CatBoostAdapter,
        "cb": CatBoostAdapter,
        "lightgbm": LightGBMAdapter,
        "lgbm": LightGBMAdapter,
    }

    @classmethod
    def get_model(cls, algorithm: str) -> OutcomePredictor:
        """
        Get a predictor instance by algorithm name.

        Args:
            algorithm: Algorithm identifier ('lr', 'rf', 'catboost', 'lightgbm').

        Returns:
            Configured OutcomePredictor instance.

        Raises:
            ValueError: If algorithm not recognized.
        """
        algorithm = algorithm.lower()
        model_class = cls._algorithms.get(algorithm)

        if model_class is None:
            valid = list(set(cls._algorithms.keys()))
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Valid options: {valid}"
            )

        return model_class()

    @classmethod
    def list_algorithms(cls) -> list[str]:
        """Return list of primary algorithm names."""
        return ["lr", "rf", "catboost", "lightgbm"]
