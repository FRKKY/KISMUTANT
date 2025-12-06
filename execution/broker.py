"""
KIS BROKER - Korea Investment & Securities API Integration

This module handles all communication with the KIS trading API.
It provides a clean interface for the rest of the system to:
- Fetch market data
- Place and manage orders
- Query account information
"""

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from loguru import logger

from core.events import get_event_bus, emit_system_event, EventType


# API Endpoints
BASE_URL_REAL = "https://openapi.koreainvestment.com:9443"
BASE_URL_PAPER = "https://openapivts.koreainvestment.com:29443"


@dataclass
class KISCredentials:
    """KIS API credentials."""
    app_key: str
    app_secret: str
    account_number: str
    account_product_code: str = "01"
    hts_id: str = ""
    
    @classmethod
    def from_yaml(cls, path: str, mode: str = "paper") -> 'KISCredentials':
        """Load credentials from YAML file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        kis_config = config.get("kis", {}).get(mode, {})
        return cls(
            app_key=kis_config.get("app_key", ""),
            app_secret=kis_config.get("app_secret", ""),
            account_number=kis_config.get("account_number", ""),
            account_product_code=kis_config.get("account_product_code", "01"),
            hts_id=config.get("kis", {}).get("hts_id", "")
        )
    
    @classmethod
    def from_environment(cls, mode: str = "paper") -> 'KISCredentials':
        """Load credentials from environment variables (for cloud deployment)."""
        try:
            from config.loader import get_kis_credentials
            config = get_kis_credentials(mode)
            if config:
                return cls(
                    app_key=config.app_key,
                    app_secret=config.app_secret,
                    account_number=config.account_number,
                    account_product_code=config.account_product_code,
                    hts_id=config.hts_id
                )
        except ImportError:
            pass
        
        # Fallback to direct environment variables
        prefix = "KIS_PAPER_" if mode == "paper" else "KIS_"
        return cls(
            app_key=os.environ.get(f"{prefix}APP_KEY", ""),
            app_secret=os.environ.get(f"{prefix}APP_SECRET", ""),
            account_number=os.environ.get(f"{prefix}ACCOUNT_NUMBER", ""),
            account_product_code=os.environ.get(f"{prefix}ACCOUNT_PRODUCT_CODE", "01"),
            hts_id=os.environ.get("KIS_HTS_ID", "")
        )


@dataclass
class Token:
    """OAuth access token."""
    access_token: str
    token_type: str
    expires_at: datetime
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired or about to expire."""
        return datetime.now() >= self.expires_at - timedelta(minutes=10)


class KISBroker:
    """
    KIS API client for the Living Trading System.
    
    Handles authentication, data fetching, and order management.
    """
    
    def __init__(
        self,
        credentials: Optional[KISCredentials] = None,
        mode: str = "paper",
        credentials_path: str = "config/credentials.yaml"
    ):
        """
        Initialize KIS broker client.
        
        Args:
            credentials: Pre-loaded credentials (optional)
            mode: "paper" for paper trading, "real" for live trading
            credentials_path: Path to credentials YAML file
        """
        self.mode = mode
        self.base_url = BASE_URL_PAPER if mode == "paper" else BASE_URL_REAL
        
        # Load credentials - try environment first (for cloud), then YAML (for local)
        if credentials:
            self.credentials = credentials
        else:
            # Try environment variables first (cloud deployment)
            self.credentials = KISCredentials.from_environment(mode)
            
            # If not configured via environment, try YAML file (local development)
            if not self.credentials.app_key:
                try:
                    self.credentials = KISCredentials.from_yaml(credentials_path, mode)
                except FileNotFoundError:
                    logger.warning(f"No credentials found for {mode} mode")
        
        # Token management
        self._token: Optional[Token] = None
        self._token_path = Path(f"config/.token_{mode}.json")
        
        # HTTP client
        self._client = httpx.Client(timeout=30.0)
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests
        
        logger.info(f"KISBroker initialized in {mode} mode")
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _load_cached_token(self) -> Optional[Token]:
        """Load token from cache file."""
        if not self._token_path.exists():
            return None
        
        try:
            with open(self._token_path, 'r') as f:
                data = json.load(f)
            
            token = Token(
                access_token=data["access_token"],
                token_type=data.get("token_type", "Bearer"),
                expires_at=datetime.fromisoformat(data["expires_at"])
            )
            
            if not token.is_expired:
                return token
                
        except Exception as e:
            logger.warning(f"Failed to load cached token: {e}")
        
        return None
    
    def _save_token(self, token: Token) -> None:
        """Save token to cache file."""
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self._token_path, 'w') as f:
            json.dump({
                "access_token": token.access_token,
                "token_type": token.token_type,
                "expires_at": token.expires_at.isoformat()
            }, f)
    
    def authenticate(self) -> Token:
        """
        Authenticate with KIS API and get access token.
        
        Returns cached token if still valid, otherwise requests new one.
        """
        # Try cached token first
        cached = self._load_cached_token()
        if cached:
            self._token = cached
            logger.info("Using cached authentication token")
            return self._token
        
        # Request new token
        logger.info("Requesting new authentication token...")
        
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.credentials.app_key,
            "appsecret": self.credentials.app_secret
        }
        
        response = self._client.post(url, headers=headers, json=body)
        
        if response.status_code != 200:
            raise RuntimeError(f"Authentication failed: {response.text}")
        
        data = response.json()
        
        self._token = Token(
            access_token=data["access_token"],
            token_type=data.get("token_type", "Bearer"),
            expires_at=datetime.now() + timedelta(seconds=data.get("expires_in", 86400))
        )
        
        self._save_token(self._token)
        logger.info("Authentication successful")
        
        return self._token
    
    def _get_headers(self, tr_id: str, hash_key: str = None) -> Dict[str, str]:
        """Build request headers."""
        if not self._token or self._token.is_expired:
            self.authenticate()
        
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._token.access_token}",
            "appkey": self.credentials.app_key,
            "appsecret": self.credentials.app_secret,
            "tr_id": tr_id,
            "custtype": "P"  # Personal customer
        }
        
        if hash_key:
            headers["hashkey"] = hash_key
        
        return headers
    
    def _get_hash_key(self, body: Dict[str, Any]) -> str:
        """Generate hash key for order requests."""
        url = f"{self.base_url}/uapi/hashkey"
        headers = {
            "content-type": "application/json",
            "appkey": self.credentials.app_key,
            "appsecret": self.credentials.app_secret
        }
        
        response = self._client.post(url, headers=headers, json=body)
        
        if response.status_code != 200:
            raise RuntimeError(f"Hash key generation failed: {response.text}")
        
        return response.json()["HASH"]
    
    def _request(
        self,
        method: str,
        endpoint: str,
        tr_id: str,
        params: Dict[str, Any] = None,
        body: Dict[str, Any] = None,
        needs_hash: bool = False
    ) -> Dict[str, Any]:
        """Make authenticated API request."""
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        hash_key = None
        if needs_hash and body:
            hash_key = self._get_hash_key(body)
        
        headers = self._get_headers(tr_id, hash_key)
        
        if method.upper() == "GET":
            response = self._client.get(url, headers=headers, params=params)
        else:
            response = self._client.post(url, headers=headers, json=body)
        
        if response.status_code != 200:
            logger.error(f"API request failed: {response.status_code} - {response.text}")
            raise RuntimeError(f"API request failed: {response.text}")
        
        data = response.json()
        
        # Check for API-level errors
        if data.get("rt_cd") != "0":
            error_msg = data.get("msg1", "Unknown error")
            logger.error(f"API error: {error_msg}")
            raise RuntimeError(f"API error: {error_msg}")
        
        return data
    
    # === MARKET DATA ===
    
    def get_price(
        self,
        symbol: str,
        market: str = "J"  # J=KOSPI, Q=KOSDAQ
    ) -> Dict[str, Any]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., "005930" for Samsung)
            market: Market code (J=KOSPI, Q=KOSDAQ)
        
        Returns:
            Dict with price data including open, high, low, close, volume
        """
        # TR_ID for domestic stock price inquiry
        tr_id = "FHKST01010100"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": market,
            "FID_INPUT_ISCD": symbol
        }
        
        data = self._request(
            method="GET",
            endpoint="/uapi/domestic-stock/v1/quotations/inquire-price",
            tr_id=tr_id,
            params=params
        )
        
        output = data.get("output", {})
        
        return {
            "symbol": symbol,
            "name": output.get("hts_kor_isnm", ""),
            "current_price": float(output.get("stck_prpr", 0)),
            "change": float(output.get("prdy_vrss", 0)),
            "change_pct": float(output.get("prdy_ctrt", 0)),
            "open": float(output.get("stck_oprc", 0)),
            "high": float(output.get("stck_hgpr", 0)),
            "low": float(output.get("stck_lwpr", 0)),
            "volume": int(output.get("acml_vol", 0)),
            "trade_value": float(output.get("acml_tr_pbmn", 0)),
            "market_cap": float(output.get("hts_avls", 0)),
            "per": float(output.get("per", 0)) if output.get("per") else None,
            "pbr": float(output.get("pbr", 0)) if output.get("pbr") else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_daily_ohlcv(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        period: str = "D",  # D=Daily, W=Weekly, M=Monthly
        market: str = "J"
    ) -> List[Dict[str, Any]]:
        """
        Get historical OHLCV data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            period: Time period (D/W/M)
            market: Market code
        
        Returns:
            List of OHLCV bars
        """
        tr_id = "FHKST01010400"  # Daily price history
        
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        
        params = {
            "FID_COND_MRKT_DIV_CODE": market,
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": start_date,
            "FID_INPUT_DATE_2": end_date,
            "FID_PERIOD_DIV_CODE": period,
            "FID_ORG_ADJ_PRC": "0"  # 0=adjusted price, 1=unadjusted
        }
        
        data = self._request(
            method="GET",
            endpoint="/uapi/domestic-stock/v1/quotations/inquire-daily-price",
            tr_id=tr_id,
            params=params
        )
        
        bars = []
        for item in data.get("output", []):
            if not item.get("stck_bsop_date"):
                continue
            bars.append({
                "date": item["stck_bsop_date"],
                "open": float(item.get("stck_oprc", 0)),
                "high": float(item.get("stck_hgpr", 0)),
                "low": float(item.get("stck_lwpr", 0)),
                "close": float(item.get("stck_clpr", 0)),
                "volume": int(item.get("acml_vol", 0)),
                "change_pct": float(item.get("prdy_ctrt", 0))
            })
        
        return sorted(bars, key=lambda x: x["date"])
    
    def get_market_index(self, index_code: str = "0001") -> Dict[str, Any]:
        """
        Get market index data.
        
        Args:
            index_code: Index code (0001=KOSPI, 1001=KOSDAQ)
        """
        tr_id = "FHPUP02100000"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "U",
            "FID_INPUT_ISCD": index_code
        }
        
        data = self._request(
            method="GET",
            endpoint="/uapi/domestic-stock/v1/quotations/inquire-index-price",
            tr_id=tr_id,
            params=params
        )
        
        output = data.get("output", {})
        
        return {
            "index_code": index_code,
            "current": float(output.get("bstp_nmix_prpr", 0)),
            "change": float(output.get("bstp_nmix_prdy_vrss", 0)),
            "change_pct": float(output.get("bstp_nmix_prdy_ctrt", 0)),
            "volume": int(output.get("acml_vol", 0)),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_stock_list(self, market: str = "J") -> List[Dict[str, Any]]:
        """
        Get list of all stocks in a market.
        
        Note: KIS API doesn't have a direct endpoint for this.
        This would need to fetch from a master file or alternative source.
        """
        # For now, return empty - will implement via master file download
        logger.warning("get_stock_list requires master file implementation")
        return []
    
    # === ACCOUNT ===
    
    def get_balance(self) -> Dict[str, Any]:
        """
        Get account balance and positions.
        
        Returns:
            Dict with cash balance and current positions
        """
        tr_id = "VTTC8434R" if self.mode == "paper" else "TTTC8434R"
        
        # Handle account number format
        account_num = self.credentials.account_number
        product_code = self.credentials.account_product_code
        
        if "-" in account_num:
            parts = account_num.split("-")
            account_num = parts[0]
            product_code = parts[1] if len(parts) > 1 else product_code
        
        params = {
            "CANO": account_num,
            "ACNT_PRDT_CD": product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        
        data = self._request(
            method="GET",
            endpoint="/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id=tr_id,
            params=params
        )
        
        # Parse positions
        positions = []
        for item in data.get("output1", []):
            if int(item.get("hldg_qty", 0)) > 0:
                positions.append({
                    "symbol": item.get("pdno", ""),
                    "name": item.get("prdt_name", ""),
                    "quantity": int(item.get("hldg_qty", 0)),
                    "avg_cost": float(item.get("pchs_avg_pric", 0)),
                    "current_price": float(item.get("prpr", 0)),
                    "market_value": float(item.get("evlu_amt", 0)),
                    "unrealized_pnl": float(item.get("evlu_pfls_amt", 0)),
                    "unrealized_pnl_pct": float(item.get("evlu_pfls_rt", 0))
                })
        
        # Parse account summary
        summary = data.get("output2", [{}])[0] if data.get("output2") else {}
        
        return {
            "cash": float(summary.get("dnca_tot_amt", 0)),
            "total_equity": float(summary.get("tot_evlu_amt", 0)),
            "positions_value": float(summary.get("scts_evlu_amt", 0)),
            "unrealized_pnl": float(summary.get("evlu_pfls_smtl_amt", 0)),
            "positions": positions,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_order_history(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> List[Dict[str, Any]]:
        """Get order history."""
        tr_id = "VTTC8001R" if self.mode == "paper" else "TTTC8001R"
        
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        
        params = {
            "CANO": self.credentials.account_number,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "INQR_STRT_DT": start_date,
            "INQR_END_DT": end_date,
            "SLL_BUY_DVSN_CD": "00",  # All orders
            "INQR_DVSN": "00",
            "PDNO": "",
            "CCLD_DVSN": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        
        data = self._request(
            method="GET",
            endpoint="/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
            tr_id=tr_id,
            params=params
        )
        
        orders = []
        for item in data.get("output1", []):
            orders.append({
                "order_id": item.get("odno", ""),
                "symbol": item.get("pdno", ""),
                "name": item.get("prdt_name", ""),
                "side": "buy" if item.get("sll_buy_dvsn_cd") == "02" else "sell",
                "quantity": int(item.get("ord_qty", 0)),
                "price": float(item.get("ord_unpr", 0)),
                "filled_quantity": int(item.get("tot_ccld_qty", 0)),
                "filled_price": float(item.get("avg_prvs", 0)),
                "status": item.get("ord_dvsn_name", ""),
                "order_time": item.get("ord_tmd", ""),
                "order_date": item.get("ord_dt", "")
            })
        
        return orders
    
    # === ORDER MANAGEMENT ===
    
    def place_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        quantity: int,
        price: float = None,
        order_type: str = "limit"  # "limit" or "market"
    ) -> Dict[str, Any]:
        """
        Place an order.
        
        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            quantity: Number of shares
            price: Limit price (required for limit orders)
            order_type: "limit" or "market"
        
        Returns:
            Order response with order ID
        """
        # Determine TR_ID based on mode and side
        if self.mode == "paper":
            tr_id = "VTTC0802U" if side == "buy" else "VTTC0801U"
        else:
            tr_id = "TTTC0802U" if side == "buy" else "TTTC0801U"
        
        # Order type code
        if order_type == "market":
            ord_dvsn = "01"  # Market order
            order_price = "0"
        else:
            ord_dvsn = "00"  # Limit order
            order_price = str(int(price))
        
        body = {
            "CANO": self.credentials.account_number,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "PDNO": symbol,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": order_price
        }
        
        data = self._request(
            method="POST",
            endpoint="/uapi/domestic-stock/v1/trading/order-cash",
            tr_id=tr_id,
            body=body,
            needs_hash=True
        )
        
        output = data.get("output", {})
        
        result = {
            "success": True,
            "order_id": output.get("ODNO", ""),
            "order_time": output.get("ORD_TMD", ""),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "message": data.get("msg1", "")
        }
        
        logger.info(f"Order placed: {side.upper()} {quantity} {symbol} - Order ID: {result['order_id']}")
        
        return result
    
    def cancel_order(
        self,
        order_id: str,
        symbol: str,
        quantity: int,
        order_branch: str = "06010"
    ) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: Original order ID
            symbol: Stock symbol
            quantity: Quantity to cancel
            order_branch: Order branch number
        """
        tr_id = "VTTC0803U" if self.mode == "paper" else "TTTC0803U"
        
        body = {
            "CANO": self.credentials.account_number,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "KRX_FWDG_ORD_ORGNO": order_branch,
            "ORGN_ODNO": order_id,
            "ORD_DVSN": "00",
            "RVSE_CNCL_DVSN_CD": "02",  # 02 = Cancel
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y"
        }
        
        data = self._request(
            method="POST",
            endpoint="/uapi/domestic-stock/v1/trading/order-rvsecncl",
            tr_id=tr_id,
            body=body,
            needs_hash=True
        )
        
        output = data.get("output", {})
        
        result = {
            "success": True,
            "original_order_id": order_id,
            "cancel_order_id": output.get("ODNO", ""),
            "message": data.get("msg1", "")
        }
        
        logger.info(f"Order cancelled: {order_id}")
        
        return result
    
    def modify_order(
        self,
        order_id: str,
        symbol: str,
        quantity: int,
        new_price: float,
        order_branch: str = "06010"
    ) -> Dict[str, Any]:
        """
        Modify an existing order's price.
        
        Args:
            order_id: Original order ID
            symbol: Stock symbol
            quantity: Quantity
            new_price: New limit price
            order_branch: Order branch number
        """
        tr_id = "VTTC0803U" if self.mode == "paper" else "TTTC0803U"
        
        body = {
            "CANO": self.credentials.account_number,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "KRX_FWDG_ORD_ORGNO": order_branch,
            "ORGN_ODNO": order_id,
            "ORD_DVSN": "00",
            "RVSE_CNCL_DVSN_CD": "01",  # 01 = Modify
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(int(new_price)),
            "QTY_ALL_ORD_YN": "Y"
        }
        
        data = self._request(
            method="POST",
            endpoint="/uapi/domestic-stock/v1/trading/order-rvsecncl",
            tr_id=tr_id,
            body=body,
            needs_hash=True
        )
        
        output = data.get("output", {})
        
        result = {
            "success": True,
            "original_order_id": order_id,
            "modified_order_id": output.get("ODNO", ""),
            "new_price": new_price,
            "message": data.get("msg1", "")
        }
        
        logger.info(f"Order modified: {order_id} -> new price: {new_price}")
        
        return result
    
    # === ETF/ETN SPECIFIC ===
    
    def get_etf_price(self, symbol: str) -> Dict[str, Any]:
        """Get ETF-specific price data including NAV."""
        # ETF uses same endpoint as stocks but with different parsing
        base_data = self.get_price(symbol, market="J")
        
        # Add ETF-specific fields if available
        # Note: Some ETF data might need different TR_IDs
        
        return base_data
    
    # === UTILITY ===
    
    def test_connection(self) -> bool:
        """Test API connectivity."""
        try:
            self.authenticate()
            # Try a simple market data request
            self.get_market_index("0001")
            logger.info("API connection test successful")
            return True
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
    
    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
        logger.info("KISBroker connection closed")


# === CONVENIENCE FUNCTIONS ===

_broker_instance: Optional[KISBroker] = None


def get_broker(mode: str = None) -> KISBroker:
    """Get or create the global broker instance."""
    global _broker_instance
    
    if _broker_instance is None:
        # Load mode from settings if not specified
        if mode is None:
            try:
                with open("config/settings.yaml", 'r') as f:
                    settings = yaml.safe_load(f)
                mode = settings.get("mode", "paper")
            except:
                mode = "paper"
        
        _broker_instance = KISBroker(mode=mode)
    
    return _broker_instance


def close_broker() -> None:
    """Close the global broker instance."""
    global _broker_instance
    if _broker_instance:
        _broker_instance.close()
        _broker_instance = None
