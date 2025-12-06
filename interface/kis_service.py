"""
KIS DATA SERVICE - Connects Dashboard to Real KIS API
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from loguru import logger


class KISDataService:
    """Fetches real data from KIS API. Falls back to empty data if unavailable."""
    
    def __init__(self):
        self._broker = None
        self._connected = False
        self._last_error = None
        self._init_broker()
    
    def _init_broker(self):
        """Initialize KIS broker connection."""
        try:
            app_key = os.environ.get("KIS_PAPER_APP_KEY", "")
            if not app_key:
                logger.info("KIS credentials not configured - using mock data")
                return
            
            from execution.broker import KISBroker
            self._broker = KISBroker(mode="paper")
            
            if self._broker.test_connection():
                self._connected = True
                logger.info("✓ KIS API connected successfully")
            else:
                logger.warning("KIS API connection test failed")
                self._last_error = "Connection test failed"
                
        except Exception as e:
            logger.error(f"Failed to initialize KIS broker: {e}")
            self._last_error = str(e)
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def get_system_status(self) -> Dict[str, Any]:
        return {
            "mode": "paper",
            "state": "running" if self._connected else "disconnected",
            "kis_connected": self._connected,
            "last_error": self._last_error,
            "last_update": datetime.now().strftime("%H:%M:%S")
        }
    
    async def get_portfolio(self) -> Dict[str, Any]:
        if not self._connected:
            return self._mock_portfolio()
        
        try:
            balance = self._broker.get_balance()
            cash = balance.get("cash", 0)
            positions = balance.get("positions", [])
            positions_value = sum(p.get("market_value", 0) for p in positions)
            total_equity = cash + positions_value
            unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
            unrealized_pnl_pct = (unrealized_pnl / total_equity * 100) if total_equity > 0 else 0
            
            return {
                "total_equity": total_equity,
                "cash": cash,
                "positions_value": positions_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
                "daily_pnl": 0,
                "daily_pnl_pct": 0,
                "weekly_pnl_pct": 0,
                "monthly_pnl_pct": 0,
                "drawdown": 0,
                "max_drawdown": 0,
                "high_water_mark": total_equity,
                "is_real_data": True
            }
        except Exception as e:
            logger.error(f"Failed to get portfolio: {e}")
            self._last_error = str(e)
            return self._mock_portfolio()
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        if not self._connected:
            return []
        
        try:
            balance = self._broker.get_balance()
            positions = balance.get("positions", [])
            
            result = []
            for pos in positions:
                result.append({
                    "symbol": pos.get("symbol", ""),
                    "name": pos.get("name", "Unknown"),
                    "quantity": pos.get("quantity", 0),
                    "avg_cost": pos.get("avg_cost", 0),
                    "current_price": pos.get("current_price", 0),
                    "market_value": pos.get("market_value", 0),
                    "pnl": pos.get("unrealized_pnl", 0),
                    "pnl_pct": pos.get("unrealized_pnl_pct", 0),
                    "weight": 0,
                    "hypothesis": "manual",
                    "is_real_data": True
                })
            return result
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def get_hypotheses(self) -> List[Dict[str, Any]]:
        return []
    
    async def get_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._connected:
            return []
        
        try:
            orders = self._broker.get_order_history()
            result = []
            for order in orders[:limit]:
                if order.get("filled_quantity", 0) > 0:
                    result.append({
                        "id": order.get("order_id", ""),
                        "timestamp": f"{order.get('order_date', '')} {order.get('order_time', '')}",
                        "symbol": order.get("symbol", ""),
                        "name": order.get("name", "Unknown"),
                        "side": order.get("side", ""),
                        "quantity": order.get("filled_quantity", 0),
                        "price": order.get("filled_price", 0),
                        "value": order.get("filled_quantity", 0) * order.get("filled_price", 0),
                        "hypothesis": "manual",
                        "status": "filled",
                        "is_real_data": True
                    })
            return result
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []
    
    async def get_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        return []
    
    def _mock_portfolio(self) -> Dict[str, Any]:
        return {
            "total_equity": 0, "cash": 0, "positions_value": 0,
            "unrealized_pnl": 0, "unrealized_pnl_pct": 0,
            "daily_pnl": 0, "daily_pnl_pct": 0,
            "weekly_pnl_pct": 0, "monthly_pnl_pct": 0,
            "drawdown": 0, "max_drawdown": 0, "high_water_mark": 0,
            "is_real_data": False
        }


_service: Optional[KISDataService] = None

def get_kis_service() -> KISDataService:
    global _service
    if _service is None:
        _service = KISDataService()
    return _service
