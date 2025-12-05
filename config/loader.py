"""
CONFIGURATION LOADER

Loads configuration from:
1. Environment variables (for cloud deployment)
2. YAML files (for local development)

Environment variables take precedence over YAML files.
This allows secure credential management in cloud environments.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger

import yaml


@dataclass
class KISConfig:
    """KIS API configuration."""
    app_key: str
    app_secret: str
    account_number: str
    account_product_code: str = "01"
    hts_id: str = ""


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str
    chat_id: int
    enabled: bool = True


@dataclass
class SystemConfig:
    """Overall system configuration."""
    mode: str = "paper"  # "paper" or "live"
    debug: bool = False
    web_port: int = 8080
    web_enabled: bool = True
    telegram_enabled: bool = True


class ConfigLoader:
    """
    Loads configuration from environment variables and YAML files.
    
    Priority:
    1. Environment variables (highest)
    2. credentials.yaml
    3. settings.yaml
    4. Default values (lowest)
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._yaml_settings: Dict[str, Any] = {}
        self._yaml_credentials: Dict[str, Any] = {}
        self._load_yaml_files()
    
    def _load_yaml_files(self):
        """Load YAML configuration files if they exist."""
        settings_path = self.config_dir / "settings.yaml"
        credentials_path = self.config_dir / "credentials.yaml"
        
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    self._yaml_settings = yaml.safe_load(f) or {}
                logger.debug("Loaded settings.yaml")
            except Exception as e:
                logger.warning(f"Could not load settings.yaml: {e}")
        
        if credentials_path.exists():
            try:
                with open(credentials_path, 'r') as f:
                    self._yaml_credentials = yaml.safe_load(f) or {}
                logger.debug("Loaded credentials.yaml")
            except Exception as e:
                logger.warning(f"Could not load credentials.yaml: {e}")
    
    def _get_env_or_yaml(
        self, 
        env_key: str, 
        yaml_path: list, 
        default: Any = None,
        credentials: bool = False
    ) -> Any:
        """
        Get value from environment variable or YAML file.
        
        Args:
            env_key: Environment variable name
            yaml_path: Path to value in YAML (e.g., ["kis", "paper", "app_key"])
            default: Default value if not found
            credentials: If True, look in credentials.yaml, else settings.yaml
        """
        # Check environment variable first
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value
        
        # Check YAML file
        yaml_data = self._yaml_credentials if credentials else self._yaml_settings
        
        try:
            value = yaml_data
            for key in yaml_path:
                value = value[key]
            
            # Don't return placeholder values
            if isinstance(value, str) and "YOUR_" in value:
                return default
            
            return value
        except (KeyError, TypeError):
            return default
    
    def get_trading_mode(self) -> str:
        """Get trading mode (paper/live)."""
        return self._get_env_or_yaml(
            "TRADING_MODE",
            ["mode"],
            default="paper"
        )
    
    def get_kis_config(self, mode: str = None) -> Optional[KISConfig]:
        """
        Get KIS API configuration.
        
        Args:
            mode: "paper" or "real". If None, uses current trading mode.
        """
        if mode is None:
            mode = self.get_trading_mode()
        
        yaml_mode = "paper" if mode == "paper" else "real"
        env_prefix = "KIS_PAPER_" if mode == "paper" else "KIS_"
        
        app_key = self._get_env_or_yaml(
            f"{env_prefix}APP_KEY",
            ["kis", yaml_mode, "app_key"],
            credentials=True
        )
        
        app_secret = self._get_env_or_yaml(
            f"{env_prefix}APP_SECRET",
            ["kis", yaml_mode, "app_secret"],
            credentials=True
        )
        
        account_number = self._get_env_or_yaml(
            f"{env_prefix}ACCOUNT_NUMBER",
            ["kis", yaml_mode, "account_number"],
            credentials=True
        )
        
        account_product_code = self._get_env_or_yaml(
            f"{env_prefix}ACCOUNT_PRODUCT_CODE",
            ["kis", yaml_mode, "account_product_code"],
            default="01",
            credentials=True
        )
        
        hts_id = self._get_env_or_yaml(
            "KIS_HTS_ID",
            ["kis", "hts_id"],
            default="",
            credentials=True
        )
        
        # Validate required fields
        if not all([app_key, app_secret, account_number]):
            logger.warning(f"KIS {mode} credentials not fully configured")
            return None
        
        return KISConfig(
            app_key=app_key,
            app_secret=app_secret,
            account_number=account_number,
            account_product_code=account_product_code,
            hts_id=hts_id
        )
    
    def get_telegram_config(self) -> Optional[TelegramConfig]:
        """Get Telegram bot configuration."""
        bot_token = self._get_env_or_yaml(
            "TELEGRAM_BOT_TOKEN",
            ["notifications", "telegram", "bot_token"],
            credentials=True
        )
        
        chat_id = self._get_env_or_yaml(
            "TELEGRAM_CHAT_ID",
            ["notifications", "telegram", "chat_id"],
            credentials=True
        )
        
        if not bot_token or not chat_id:
            logger.warning("Telegram not configured")
            return None
        
        try:
            chat_id = int(chat_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid Telegram chat_id: {chat_id}")
            return None
        
        return TelegramConfig(
            bot_token=bot_token,
            chat_id=chat_id,
            enabled=True
        )
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        return SystemConfig(
            mode=self.get_trading_mode(),
            debug=self._get_env_or_yaml("DEBUG", ["debug"], default=False) in [True, "true", "1"],
            web_port=int(self._get_env_or_yaml("PORT", ["web_port"], default=8080)),
            web_enabled=self._get_env_or_yaml("WEB_ENABLED", ["web_enabled"], default=True) in [True, "true", "1", None],
            telegram_enabled=self._get_env_or_yaml("TELEGRAM_ENABLED", ["telegram_enabled"], default=True) in [True, "true", "1", None]
        )
    
    def print_config_status(self):
        """Print configuration status for debugging."""
        print("\n" + "=" * 50)
        print("CONFIGURATION STATUS")
        print("=" * 50)
        
        # System
        sys_config = self.get_system_config()
        print(f"\nSystem:")
        print(f"  Mode: {sys_config.mode}")
        print(f"  Web Dashboard: {'Enabled' if sys_config.web_enabled else 'Disabled'}")
        print(f"  Telegram Bot: {'Enabled' if sys_config.telegram_enabled else 'Disabled'}")
        
        # KIS
        kis_config = self.get_kis_config()
        print(f"\nKIS API ({sys_config.mode}):")
        if kis_config:
            print(f"  App Key: {kis_config.app_key[:8]}..." if kis_config.app_key else "  App Key: Not set")
            print(f"  Account: {kis_config.account_number}")
            print(f"  ✓ Configured")
        else:
            print(f"  ✗ Not configured")
        
        # Telegram
        tg_config = self.get_telegram_config()
        print(f"\nTelegram:")
        if tg_config:
            print(f"  Bot Token: {tg_config.bot_token[:10]}...")
            print(f"  Chat ID: {tg_config.chat_id}")
            print(f"  ✓ Configured")
        else:
            print(f"  ✗ Not configured")
        
        print("\n" + "=" * 50)


# Global config loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Get the global configuration loader."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


# Convenience functions
def get_kis_credentials(mode: str = None) -> Optional[KISConfig]:
    """Get KIS API credentials."""
    return get_config().get_kis_config(mode)


def get_telegram_credentials() -> Optional[TelegramConfig]:
    """Get Telegram credentials."""
    return get_config().get_telegram_config()


def get_trading_mode() -> str:
    """Get current trading mode."""
    return get_config().get_trading_mode()


if __name__ == "__main__":
    # Test configuration loading
    config = ConfigLoader()
    config.print_config_status()
