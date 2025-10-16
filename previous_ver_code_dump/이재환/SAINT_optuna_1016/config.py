import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional

@dataclass
class ModelConfig:
    # SAINT specific parameters
    embedding_size: int = 32
    transformer_depth: int = 6
    attention_heads: int = 8
    attention_dropout: float = 0.1
    ff_dropout: float = 0.1
    optimizer: str = "AdamW"
    scheduler: str = "StepLR"
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # Common parameters
    lr: float = 0.0001
    epochs: int = 100
    batch_size: int = 256
    task: str = "regression"
    pretrain: bool = False
    verbose: int = 1
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    # XGBoost specific parameters (for compatibility - not used in SAINT)
    xgb_max_depth: int = 6
    xgb_min_child_weight: float = 1.0
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_gamma: float = 0.0
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0
    xgb_scale_pos_weight: float = 1.0
    xgb_tree_method: str = "hist"
    xgb_device: str = "cpu"
    xgb_eval_metric: str = "rmse"

@dataclass
class LLMConfig:
    api_key: str = ""
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: str = "You are an AI assistant analyzing machine learning model results."

@dataclass
class UIConfig:
    show_plots_popup: bool = True
    save_plots_image: bool = True
    plots_save_dir: str = "./plots"
    enable_streamlit_chat: bool = True

@dataclass
class DataConfig:
    dataset_dir: str = "./dataset"
    input_dynamic_size: bool = True
    target_columns: List[str] = field(default_factory=lambda: ["target"])
    feature_columns: List[str] = field(default_factory=lambda: [])

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def save(self, filepath: str = "config.json"):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str = "config.json"):
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            config = cls()
            config.model = ModelConfig(**data.get('model', {}))
            config.llm = LLMConfig(**data.get('llm', {}))
            config.ui = UIConfig(**data.get('ui', {}))
            config.data = DataConfig(**data.get('data', {}))
            return config

        return cls()

def get_config() -> Config:
    return Config.load()