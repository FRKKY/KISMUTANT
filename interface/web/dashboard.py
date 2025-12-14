"""
WEB DASHBOARD - Browser Interface for Living Trading System

A comprehensive web dashboard for monitoring and controlling the trading system.

Features:
- Real-time portfolio overview with equity curve
- Position management with hypothesis attribution
- Enhanced hypothesis monitoring with detailed metrics
- Trade history and analytics
- Decision journal viewer
- Market regime indicator
- Capital allocation visualization
- Alerts panel
- Research pipeline status
- Performance analytics

Built with FastAPI + HTMX + Chart.js for a responsive interface.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
from collections import defaultdict

from loguru import logger

try:
    from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect, Query
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. Run: pip install fastapi uvicorn jinja2")

# Optional rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATELIMIT_AVAILABLE = True
except ImportError:
    RATELIMIT_AVAILABLE = False


# === APPLICATION SETUP ===

app = FastAPI(
    title="Living Trading System",
    description="Web dashboard for the Living Trading System",
    version="0.2.0"
)

# Setup rate limiting if available
if RATELIMIT_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Track startup time for health check
_startup_time = datetime.now()

# In-memory alert storage (recent alerts)
_alerts: List[Dict[str, Any]] = []
MAX_ALERTS = 50

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Setup Jinja2 templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []


# === ALERT SYSTEM ===

def add_alert(alert_type: str, title: str, message: str, severity: str = "info"):
    """Add an alert to the system."""
    global _alerts
    alert = {
        "id": len(_alerts),
        "type": alert_type,
        "title": title,
        "message": message,
        "severity": severity,  # info, warning, danger, success
        "timestamp": datetime.now().isoformat(),
        "read": False
    }
    _alerts.insert(0, alert)
    if len(_alerts) > MAX_ALERTS:
        _alerts = _alerts[:MAX_ALERTS]
    return alert


# === DATA PROVIDERS ===

class DataProvider:
    """Provides data to the dashboard. Connects to system components."""

    _service = None

    @classmethod
    def _get_service(cls):
        """Get KIS service (lazy initialization)."""
        if cls._service is None:
            try:
                from interface.kis_service import get_kis_service
                cls._service = get_kis_service()
            except Exception as e:
                logger.debug(f"Failed to get KIS service: {e}")
                cls._service = None
        return cls._service

    @staticmethod
    async def get_system_status() -> Dict[str, Any]:
        """Get current system status."""
        service = DataProvider._get_service()
        if service:
            return await service.get_system_status()
        return {
            "mode": "paper",
            "state": "running",
            "kis_connected": False,
            "last_update": datetime.now().strftime("%H:%M:%S")
        }

    @staticmethod
    async def get_portfolio() -> Dict[str, Any]:
        """Get portfolio summary."""
        service = DataProvider._get_service()
        if service and service.is_connected:
            return await service.get_portfolio()
        return {
            "total_equity": 0, "cash": 0, "positions_value": 0,
            "unrealized_pnl": 0, "unrealized_pnl_pct": 0,
            "daily_pnl": 0, "daily_pnl_pct": 0,
            "weekly_pnl_pct": 0, "monthly_pnl_pct": 0,
            "drawdown": 0, "max_drawdown": 0, "high_water_mark": 0
        }

    @staticmethod
    async def get_positions() -> List[Dict[str, Any]]:
        """Get current positions."""
        service = DataProvider._get_service()
        if service and service.is_connected:
            return await service.get_positions()
        return []

    @staticmethod
    async def get_hypotheses() -> List[Dict[str, Any]]:
        """Get all hypotheses with detailed metrics."""
        hypotheses = []

        try:
            from hypothesis import get_registry, get_promoter
            registry = get_registry()

            # Try to get promoter for promotion candidates
            try:
                promoter = get_promoter()
                candidates = {c["hypothesis_id"]: c["priority_score"]
                             for c in promoter.get_promotion_candidates()}
            except:
                candidates = {}

            for hyp in registry.get_all():
                # Get metrics based on state
                metrics = hyp.backtest_metrics or {}
                if hyp.state.value == "paper_trading" and hyp.paper_metrics:
                    metrics = hyp.paper_metrics
                elif hyp.state.value == "live" and hyp.live_metrics:
                    metrics = hyp.live_metrics

                # Handle metrics as dict or object
                if hasattr(metrics, 'to_dict'):
                    metrics = metrics.to_dict()
                elif not isinstance(metrics, dict):
                    metrics = {}

                hypotheses.append({
                    "id": hyp.hypothesis_id,
                    "id_short": hyp.hypothesis_id[:8],
                    "name": hyp.name,
                    "description": hyp.description[:100] if hyp.description else "",
                    "status": hyp.state.value if hasattr(hyp.state, 'value') else str(hyp.state),
                    "strategy_type": hyp.strategy_type.value if hasattr(hyp, 'strategy_type') and hasattr(hyp.strategy_type, 'value') else "unknown",
                    "symbols": hyp.symbols if hasattr(hyp, 'symbols') else [],
                    "allocation": hyp.capital_pct * 100 if hasattr(hyp, 'capital_pct') else 0,
                    "win_rate": metrics.get("win_rate", 0) * 100 if isinstance(metrics.get("win_rate", 0), float) and metrics.get("win_rate", 0) <= 1 else metrics.get("win_rate", 0),
                    "sharpe": metrics.get("sharpe_ratio", 0),
                    "sortino": metrics.get("sortino_ratio", 0),
                    "max_drawdown": metrics.get("max_drawdown", 0) * 100,
                    "profit_factor": metrics.get("profit_factor", 0),
                    "total_trades": metrics.get("total_trades", 0),
                    "total_return": metrics.get("total_return", 0) * 100,
                    "pnl": metrics.get("total_pnl", 0) if "total_pnl" in metrics else 0,
                    "promotion_score": candidates.get(hyp.hypothesis_id, 0),
                    "days_in_state": (datetime.now() - hyp.incubation_start).days if hyp.incubation_start else 0,
                    "source": "registry"
                })
        except Exception as e:
            logger.debug(f"Could not fetch from registry: {e}")

        try:
            from research import get_knowledge_base
            kb = get_knowledge_base()

            for hyp_id, hyp in kb._hypotheses.items():
                if not any(h["id"] == hyp_id for h in hypotheses):
                    hypotheses.append({
                        "id": hyp_id,
                        "id_short": hyp_id[:8],
                        "name": hyp.name,
                        "description": hyp.description[:100] if hasattr(hyp, 'description') else "",
                        "status": "research",
                        "strategy_type": hyp.strategy_type if hasattr(hyp, 'strategy_type') else "unknown",
                        "symbols": [],
                        "allocation": 0,
                        "win_rate": 0,
                        "sharpe": 0,
                        "sortino": 0,
                        "max_drawdown": 0,
                        "profit_factor": 0,
                        "total_trades": 0,
                        "total_return": 0,
                        "pnl": 0,
                        "promotion_score": 0,
                        "days_in_state": 0,
                        "source": "research"
                    })
        except Exception as e:
            logger.debug(f"Could not fetch from knowledge base: {e}")

        return hypotheses

    @staticmethod
    async def get_trades(limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trades."""
        service = DataProvider._get_service()
        if service and service.is_connected:
            return await service.get_trades(limit)
        return []

    @staticmethod
    async def get_decisions(limit: int = 50, decision_type: str = None) -> List[Dict[str, Any]]:
        """Get recent decisions from the journal."""
        try:
            from memory.journal import get_journal, DecisionType
            journal = get_journal()

            dtype = None
            if decision_type:
                try:
                    dtype = DecisionType(decision_type)
                except:
                    pass

            decisions = journal.get_decisions(decision_type=dtype, limit=limit)
            return decisions
        except Exception as e:
            logger.debug(f"Could not fetch decisions: {e}")
            return []

    @staticmethod
    async def get_equity_curve(days: int = 30) -> Dict[str, Any]:
        """Get equity curve data for charting."""
        try:
            from memory.models import get_database, PortfolioSnapshot
            db = get_database()
            session = db.get_session()

            start_date = datetime.now() - timedelta(days=days)
            snapshots = session.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.timestamp >= start_date
            ).order_by(PortfolioSnapshot.timestamp).all()

            session.close()

            if snapshots:
                dates = [s.timestamp.strftime("%Y-%m-%d") for s in snapshots]
                equity = [s.total_equity for s in snapshots]

                # Calculate daily returns
                returns = [0]
                for i in range(1, len(equity)):
                    if equity[i-1] > 0:
                        returns.append((equity[i] - equity[i-1]) / equity[i-1] * 100)
                    else:
                        returns.append(0)

                return {
                    "dates": dates,
                    "equity": equity,
                    "returns": returns,
                    "benchmark": []  # Could add KOSPI benchmark
                }
        except Exception as e:
            logger.debug(f"Could not fetch equity curve: {e}")

        # Return sample data structure
        return {"dates": [], "equity": [], "returns": [], "benchmark": []}

    @staticmethod
    async def get_market_regime() -> Dict[str, Any]:
        """Get current market regime classification."""
        try:
            from learning import get_regime_detector
            detector = get_regime_detector()
            state = detector.export_state()

            return {
                "regime": state.get("current_regime", "unknown"),
                "confidence": state.get("regime_confidence", 0),
                "duration_days": state.get("regime_duration", 0),
                "volatility": state.get("volatility_state", "normal"),
                "trend": state.get("trend_state", "neutral"),
                "regime_history": state.get("regime_history", [])[-10:]  # Last 10
            }
        except Exception as e:
            logger.debug(f"Could not fetch market regime: {e}")

        return {
            "regime": "unknown",
            "confidence": 0,
            "duration_days": 0,
            "volatility": "normal",
            "trend": "neutral",
            "regime_history": []
        }

    @staticmethod
    async def get_capital_allocation() -> Dict[str, Any]:
        """Get capital allocation breakdown."""
        try:
            from portfolio import get_capital_allocator
            allocator = get_capital_allocator()
            stats = allocator.get_stats()

            # Get allocations by strategy
            allocations = []
            for strategy_id, alloc in stats.get("allocations", {}).items():
                allocations.append({
                    "strategy_id": strategy_id[:8],
                    "allocated": alloc.get("allocated", 0),
                    "deployed": alloc.get("deployed", 0),
                    "pnl": alloc.get("realized_pnl", 0)
                })

            return {
                "total_capital": stats.get("total_capital", 0),
                "available_capital": stats.get("available_capital", 0),
                "deployed_capital": stats.get("deployed_capital", 0),
                "reserved_capital": stats.get("reserved_capital", 0),
                "daily_risk_used": stats.get("daily_risk_used", 0),
                "max_daily_risk": stats.get("max_daily_risk", 0.02),
                "allocations": allocations
            }
        except Exception as e:
            logger.debug(f"Could not fetch capital allocation: {e}")

        return {
            "total_capital": 0,
            "available_capital": 0,
            "deployed_capital": 0,
            "reserved_capital": 0,
            "daily_risk_used": 0,
            "max_daily_risk": 0.02,
            "allocations": []
        }

    @staticmethod
    async def get_performance_analytics() -> Dict[str, Any]:
        """Get performance analytics data."""
        try:
            from learning import get_analyzer
            analyzer = get_analyzer()
            stats = analyzer.get_stats()

            return {
                "total_trades": stats.get("total_trades", 0),
                "strategies_analyzed": stats.get("strategies_analyzed", 0),
                "by_strategy": stats.get("by_strategy", {}),
                "overall": {
                    "win_rate": stats.get("overall_win_rate", 0),
                    "avg_return": stats.get("avg_return", 0),
                    "best_trade": stats.get("best_trade", 0),
                    "worst_trade": stats.get("worst_trade", 0),
                    "avg_holding_period": stats.get("avg_holding_period", 0)
                }
            }
        except Exception as e:
            logger.debug(f"Could not fetch performance analytics: {e}")

        return {
            "total_trades": 0,
            "strategies_analyzed": 0,
            "by_strategy": {},
            "overall": {
                "win_rate": 0,
                "avg_return": 0,
                "best_trade": 0,
                "worst_trade": 0,
                "avg_holding_period": 0
            }
        }

    @staticmethod
    async def get_research_status() -> Dict[str, Any]:
        """Get research pipeline status."""
        result = {
            "papers": [],
            "ideas": [],
            "hypotheses_generated": 0,
            "knowledge_stats": {},
            "fetcher_stats": {},
            "extractor_stats": {},
            "last_fetch": None,
            "pipeline_status": "idle"
        }

        try:
            from research import get_knowledge_base, get_fetcher, get_extractor
            kb = get_knowledge_base()
            fetcher = get_fetcher()
            extractor = get_extractor()

            # Get recent papers
            for paper_id, paper in list(kb._papers.items())[:15]:
                result["papers"].append({
                    "id": paper_id,
                    "title": paper.title[:70] + "..." if len(paper.title) > 70 else paper.title,
                    "authors": paper.authors[:2],
                    "relevance": paper.relevance_score,
                    "keywords": paper.trading_keywords[:4],
                    "source": paper.source.value if hasattr(paper.source, 'value') else str(paper.source),
                    "published": paper.published_date.strftime("%Y-%m-%d") if paper.published_date else "N/A"
                })

            # Get recent ideas
            for idea_id, idea in list(kb._ideas.items())[:15]:
                result["ideas"].append({
                    "id": idea_id,
                    "title": idea.title[:50] + "..." if len(idea.title) > 50 else idea.title,
                    "type": idea.idea_type.value if hasattr(idea.idea_type, 'value') else str(idea.idea_type),
                    "confidence": idea.confidence.value if hasattr(idea.confidence, 'value') else str(idea.confidence),
                    "indicators": idea.indicators[:3] if idea.indicators else [],
                    "asset_class": idea.asset_class if hasattr(idea, 'asset_class') else None
                })

            # Stats
            result["knowledge_stats"] = kb.get_stats()
            result["fetcher_stats"] = fetcher.get_stats()
            result["extractor_stats"] = extractor.get_stats()
            result["hypotheses_generated"] = len(kb._hypotheses)

        except Exception as e:
            logger.debug(f"Could not fetch research data: {e}")

        return result


# === API ROUTES ===

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    status = await DataProvider.get_system_status()
    portfolio = await DataProvider.get_portfolio()
    positions = await DataProvider.get_positions()
    hypotheses = await DataProvider.get_hypotheses()
    trades = await DataProvider.get_trades(10)
    decisions = await DataProvider.get_decisions(10)
    regime = await DataProvider.get_market_regime()
    allocation = await DataProvider.get_capital_allocation()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "status": status,
        "portfolio": portfolio,
        "positions": positions,
        "hypotheses": hypotheses,
        "trades": trades,
        "decisions": decisions,
        "regime": regime,
        "allocation": allocation,
        "alerts": _alerts[:10]
    })


@app.get("/api/status")
async def api_status():
    """Get system status."""
    return await DataProvider.get_system_status()


@app.get("/api/portfolio")
async def api_portfolio():
    """Get portfolio data."""
    return await DataProvider.get_portfolio()


@app.get("/api/positions")
async def api_positions():
    """Get positions."""
    return await DataProvider.get_positions()


@app.get("/api/hypotheses")
async def api_hypotheses():
    """Get hypotheses with detailed metrics."""
    return await DataProvider.get_hypotheses()


@app.get("/api/hypothesis/{hypothesis_id}")
async def api_hypothesis_detail(hypothesis_id: str):
    """Get detailed info for a specific hypothesis."""
    hypotheses = await DataProvider.get_hypotheses()
    for h in hypotheses:
        if h["id"] == hypothesis_id or h["id_short"] == hypothesis_id:
            return h
    raise HTTPException(status_code=404, detail="Hypothesis not found")


@app.get("/api/trades")
async def api_trades(limit: int = 20):
    """Get recent trades."""
    return await DataProvider.get_trades(limit)


@app.get("/api/decisions")
async def api_decisions(limit: int = 50, decision_type: str = None):
    """Get recent decisions from the journal."""
    return await DataProvider.get_decisions(limit, decision_type)


@app.get("/api/equity-curve")
async def api_equity_curve(days: int = 30):
    """Get equity curve data for charting."""
    return await DataProvider.get_equity_curve(days)


@app.get("/api/market-regime")
async def api_market_regime():
    """Get current market regime."""
    return await DataProvider.get_market_regime()


@app.get("/api/capital-allocation")
async def api_capital_allocation():
    """Get capital allocation breakdown."""
    return await DataProvider.get_capital_allocation()


@app.get("/api/performance-analytics")
async def api_performance_analytics():
    """Get performance analytics."""
    return await DataProvider.get_performance_analytics()


@app.get("/api/research")
async def api_research():
    """Get research module status."""
    return await DataProvider.get_research_status()


@app.get("/api/alerts")
async def api_alerts(limit: int = 20):
    """Get recent alerts."""
    return _alerts[:limit]


@app.post("/api/alerts/read/{alert_id}")
async def api_mark_alert_read(alert_id: int):
    """Mark an alert as read."""
    for alert in _alerts:
        if alert["id"] == alert_id:
            alert["read"] = True
            return {"success": True}
    return {"success": False}


@app.get("/api/system")
async def api_system():
    """Get full system status including orchestrator state."""
    result = {
        "orchestrator": None,
        "market": {"is_open": False, "next_open": None},
        "components": {},
        "promoter": None,
        "validator": None
    }

    try:
        from orchestrator import get_orchestrator
        from core.clock import get_clock, is_market_open

        clock = get_clock()
        result["market"] = {
            "is_open": is_market_open(),
            "current_time_kst": clock.now().strftime("%Y-%m-%d %H:%M:%S KST"),
            "time_to_open": str(clock.time_to_open()) if clock.time_to_open() else None
        }

        try:
            orch = get_orchestrator()
            result["orchestrator"] = orch.get_status()
        except:
            pass

        try:
            from hypothesis import get_promoter
            promoter = get_promoter()
            result["promoter"] = promoter.get_stats()
        except:
            pass

        try:
            from execution import get_validator
            validator = get_validator()
            result["validator"] = validator.get_stats()
        except:
            pass

    except Exception as e:
        logger.debug(f"Could not fetch system status: {e}")

    return result


# === HEALTH CHECK ENDPOINTS ===

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - _startup_time).total_seconds(),
        "components": {}
    }

    try:
        from memory.models import get_database
        db = get_database()
        session = db.get_session()
        session.execute("SELECT 1")
        session.close()
        health["components"]["database"] = "ok"
    except Exception as e:
        health["components"]["database"] = f"error: {str(e)[:50]}"
        health["status"] = "degraded"

    try:
        from core.clock import get_clock, is_market_open
        clock = get_clock()
        health["components"]["clock"] = "ok"
        health["market_open"] = is_market_open()
    except Exception as e:
        health["components"]["clock"] = f"error: {str(e)[:50]}"

    try:
        from research import get_knowledge_base
        kb = get_knowledge_base()
        health["components"]["knowledge_base"] = "ok"
    except Exception as e:
        health["components"]["knowledge_base"] = f"error: {str(e)[:50]}"

    errors = [v for v in health["components"].values() if v != "ok"]
    if len(errors) > len(health["components"]) / 2:
        health["status"] = "unhealthy"
    elif errors:
        health["status"] = "degraded"

    return health


@app.get("/health/live")
async def liveness_check():
    """Simple liveness probe."""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness probe."""
    try:
        from memory.models import get_database
        db = get_database()
        session = db.get_session()
        session.execute("SELECT 1")
        session.close()
        return {"status": "ready"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": str(e)[:100]}
        )


# === CONTROL ENDPOINTS ===

@app.post("/api/control/pause")
async def control_pause():
    """Pause the trading system."""
    logger.info("Pause requested via web dashboard")
    try:
        from orchestrator import get_orchestrator
        orch = get_orchestrator()
        await orch.pause("web_dashboard")
        add_alert("system", "System Paused", "Trading system paused via dashboard", "warning")
        return {"success": True, "message": "System paused"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/control/resume")
async def control_resume():
    """Resume the trading system."""
    logger.info("Resume requested via web dashboard")
    try:
        from orchestrator import get_orchestrator
        orch = get_orchestrator()
        await orch.resume()
        add_alert("system", "System Resumed", "Trading system resumed via dashboard", "success")
        return {"success": True, "message": "System resumed"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/control/stop")
async def control_stop():
    """Emergency stop."""
    logger.warning("Emergency stop requested via web dashboard")
    try:
        from orchestrator import get_orchestrator
        orch = get_orchestrator()
        await orch.stop()
        add_alert("system", "EMERGENCY STOP", "Trading system stopped via dashboard", "danger")
        return {"success": True, "message": "Emergency stop executed"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/hypothesis/{hypothesis_id}/pause")
async def hypothesis_pause(hypothesis_id: str):
    """Pause a specific hypothesis."""
    logger.info(f"Hypothesis {hypothesis_id} pause requested")
    try:
        from hypothesis import get_registry
        registry = get_registry()
        success, msg = registry.pause(hypothesis_id, "paused_via_dashboard")
        if success:
            add_alert("hypothesis", "Hypothesis Paused", f"Strategy {hypothesis_id[:8]} paused", "warning")
        return {"success": success, "message": msg}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/hypothesis/{hypothesis_id}/retire")
async def hypothesis_retire(hypothesis_id: str):
    """Retire a specific hypothesis."""
    logger.info(f"Hypothesis {hypothesis_id} retire requested")
    try:
        from hypothesis import get_registry
        registry = get_registry()
        success, msg = registry.retire(hypothesis_id, "retired_via_dashboard")
        if success:
            add_alert("hypothesis", "Hypothesis Retired", f"Strategy {hypothesis_id[:8]} retired", "info")
        return {"success": success, "message": msg}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/hypothesis/{hypothesis_id}/promote")
async def hypothesis_promote(hypothesis_id: str):
    """Force promote a hypothesis."""
    logger.info(f"Hypothesis {hypothesis_id} promote requested")
    try:
        from hypothesis import get_registry
        registry = get_registry()
        success, msg = registry.promote(hypothesis_id, force=True)
        if success:
            add_alert("hypothesis", "Hypothesis Promoted", f"Strategy {hypothesis_id[:8]} promoted", "success")
        return {"success": success, "message": msg}
    except Exception as e:
        return {"success": False, "message": str(e)}


# === WEBSOCKET FOR REAL-TIME UPDATES ===

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            await asyncio.sleep(5)

            data = {
                "type": "update",
                "portfolio": await DataProvider.get_portfolio(),
                "status": await DataProvider.get_system_status(),
                "regime": await DataProvider.get_market_regime(),
                "alerts": _alerts[:5],
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_json(data)

    except WebSocketDisconnect:
        active_connections.remove(websocket)


async def broadcast_update(data: Dict[str, Any]):
    """Broadcast update to all connected clients."""
    for connection in active_connections:
        try:
            await connection.send_json(data)
        except:
            pass


# === RUNNER ===

def run_dashboard(host: str = "0.0.0.0", port: int = None):
    """Run the web dashboard."""
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Run: pip install fastapi uvicorn jinja2")
        return

    if port is None:
        port = int(os.environ.get("PORT", 8080))

    create_default_templates()

    print(f"\n🌐 Starting web dashboard at http://localhost:{port}")
    print("   Press Ctrl+C to stop\n")

    uvicorn.run(app, host=host, port=port)


def create_default_templates():
    """Create default HTML templates."""
    dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Living Trading System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .positive { color: #10b981; }
        .negative { color: #ef4444; }
        .card { @apply bg-white rounded-lg shadow p-4; }
        .tab-active { border-bottom: 2px solid #3b82f6; color: #3b82f6; }
        .regime-bull { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
        .regime-bear { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
        .regime-sideways { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
        .regime-unknown { background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%); }
        .alert-info { border-left: 4px solid #3b82f6; }
        .alert-warning { border-left: 4px solid #f59e0b; }
        .alert-danger { border-left: 4px solid #ef4444; }
        .alert-success { border-left: 4px solid #10b981; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <!-- Header -->
    <header class="bg-gray-900 text-white p-4 shadow-lg sticky top-0 z-50">
        <div class="container mx-auto flex justify-between items-center">
            <div class="flex items-center gap-4">
                <h1 class="text-xl font-bold">🤖 Living Trading System</h1>
                <span class="px-3 py-1 rounded-full text-sm
                    {% if status.mode == 'paper' %}bg-yellow-500{% else %}bg-green-500{% endif %}">
                    {{ status.mode | upper }}
                </span>
                <span class="px-3 py-1 rounded-full text-sm
                    {% if status.state == 'running' %}bg-green-500{% elif status.state == 'paused' %}bg-yellow-500{% else %}bg-red-500{% endif %}">
                    {{ status.state | upper }}
                </span>
                <!-- Market Regime Badge -->
                <span class="px-3 py-1 rounded-full text-sm text-white
                    regime-{{ regime.regime | default('unknown') }}">
                    📊 {{ regime.regime | upper | default('UNKNOWN') }}
                    {% if regime.confidence > 0 %}({{ "{:.0f}".format(regime.confidence * 100) }}%){% endif %}
                </span>
            </div>
            <div class="flex gap-2">
                <button onclick="pauseSystem()" class="px-4 py-2 bg-yellow-600 rounded hover:bg-yellow-700">
                    ⏸️ Pause
                </button>
                <button onclick="resumeSystem()" class="px-4 py-2 bg-green-600 rounded hover:bg-green-700">
                    ▶️ Resume
                </button>
                <button onclick="stopSystem()" class="px-4 py-2 bg-red-600 rounded hover:bg-red-700">
                    ⏹️ Stop
                </button>
            </div>
        </div>
    </header>

    <main class="container mx-auto p-4">
        <!-- Alerts Banner -->
        {% if alerts %}
        <div class="mb-4" id="alerts-container">
            {% for alert in alerts[:3] if not alert.read %}
            <div class="bg-white rounded-lg shadow p-3 mb-2 alert-{{ alert.severity }} flex justify-between items-center">
                <div>
                    <span class="font-bold">{{ alert.title }}</span>
                    <span class="text-gray-600 ml-2">{{ alert.message }}</span>
                    <span class="text-xs text-gray-400 ml-2">{{ alert.timestamp }}</span>
                </div>
                <button onclick="dismissAlert({{ alert.id }})" class="text-gray-400 hover:text-gray-600">✕</button>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <!-- Portfolio Overview Cards -->
        <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
            <div class="bg-white rounded-lg shadow p-4">
                <div class="text-gray-500 text-xs uppercase">Total Equity</div>
                <div class="text-xl font-bold">₩{{ "{:,.0f}".format(portfolio.total_equity) }}</div>
                <div class="text-sm {% if portfolio.daily_pnl_pct >= 0 %}positive{% else %}negative{% endif %}">
                    {{ "{:+.2f}".format(portfolio.daily_pnl_pct) }}% today
                </div>
            </div>
            <div class="bg-white rounded-lg shadow p-4">
                <div class="text-gray-500 text-xs uppercase">Cash</div>
                <div class="text-xl font-bold">₩{{ "{:,.0f}".format(portfolio.cash) }}</div>
                <div class="text-sm text-gray-500">
                    {% if portfolio.total_equity > 0 %}{{ "{:.0f}".format(portfolio.cash / portfolio.total_equity * 100) }}%{% else %}0%{% endif %}
                </div>
            </div>
            <div class="bg-white rounded-lg shadow p-4">
                <div class="text-gray-500 text-xs uppercase">Unrealized P&L</div>
                <div class="text-xl font-bold {% if portfolio.unrealized_pnl >= 0 %}positive{% else %}negative{% endif %}">
                    ₩{{ "{:+,.0f}".format(portfolio.unrealized_pnl) }}
                </div>
                <div class="text-sm {% if portfolio.unrealized_pnl_pct >= 0 %}positive{% else %}negative{% endif %}">
                    {{ "{:+.2f}".format(portfolio.unrealized_pnl_pct) }}%
                </div>
            </div>
            <div class="bg-white rounded-lg shadow p-4">
                <div class="text-gray-500 text-xs uppercase">Drawdown</div>
                <div class="text-xl font-bold text-red-500">{{ "{:.1f}".format(portfolio.drawdown) }}%</div>
                <div class="text-sm text-gray-500">Max: {{ "{:.1f}".format(portfolio.max_drawdown) }}%</div>
            </div>
            <div class="bg-white rounded-lg shadow p-4">
                <div class="text-gray-500 text-xs uppercase">Deployed</div>
                <div class="text-xl font-bold">₩{{ "{:,.0f}".format(allocation.deployed_capital) }}</div>
                <div class="text-sm text-gray-500">
                    {% if allocation.total_capital > 0 %}{{ "{:.0f}".format(allocation.deployed_capital / allocation.total_capital * 100) }}%{% else %}0%{% endif %}
                </div>
            </div>
            <div class="bg-white rounded-lg shadow p-4">
                <div class="text-gray-500 text-xs uppercase">Daily Risk</div>
                <div class="text-xl font-bold {% if allocation.daily_risk_used > allocation.max_daily_risk * 0.8 %}text-red-500{% else %}text-green-500{% endif %}">
                    {{ "{:.1f}".format(allocation.daily_risk_used * 100) }}%
                </div>
                <div class="text-sm text-gray-500">Max: {{ "{:.1f}".format(allocation.max_daily_risk * 100) }}%</div>
            </div>
        </div>

        <!-- Tabs Navigation -->
        <div class="bg-white rounded-lg shadow mb-6">
            <div class="flex border-b">
                <button onclick="showTab('overview')" id="tab-overview" class="px-6 py-3 font-medium tab-active">📊 Overview</button>
                <button onclick="showTab('hypotheses')" id="tab-hypotheses" class="px-6 py-3 font-medium text-gray-500 hover:text-gray-700">🎯 Strategies</button>
                <button onclick="showTab('decisions')" id="tab-decisions" class="px-6 py-3 font-medium text-gray-500 hover:text-gray-700">📝 Decisions</button>
                <button onclick="showTab('research')" id="tab-research" class="px-6 py-3 font-medium text-gray-500 hover:text-gray-700">🔬 Research</button>
                <button onclick="showTab('analytics')" id="tab-analytics" class="px-6 py-3 font-medium text-gray-500 hover:text-gray-700">📈 Analytics</button>
            </div>

            <!-- Overview Tab -->
            <div id="content-overview" class="p-4">
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <!-- Equity Chart -->
                    <div class="lg:col-span-2">
                        <h3 class="font-bold mb-3">Equity Curve</h3>
                        <div class="bg-gray-50 rounded p-4" style="height: 300px;">
                            <canvas id="equityChart"></canvas>
                        </div>
                    </div>

                    <!-- Capital Allocation Pie -->
                    <div>
                        <h3 class="font-bold mb-3">Capital Allocation</h3>
                        <div class="bg-gray-50 rounded p-4" style="height: 300px;">
                            <canvas id="allocationChart"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Positions and Trades -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                    <!-- Positions -->
                    <div>
                        <h3 class="font-bold mb-3">📋 Positions ({{ positions | length }})</h3>
                        <div class="overflow-x-auto max-h-64 overflow-y-auto">
                            <table class="w-full text-sm">
                                <thead class="bg-gray-50 text-xs text-gray-500 uppercase sticky top-0">
                                    <tr>
                                        <th class="p-2 text-left">Symbol</th>
                                        <th class="p-2 text-right">Qty</th>
                                        <th class="p-2 text-right">Value</th>
                                        <th class="p-2 text-right">P&L</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for pos in positions %}
                                    <tr class="border-t hover:bg-gray-50">
                                        <td class="p-2">
                                            <div class="font-medium">{{ pos.symbol }}</div>
                                            <div class="text-xs text-gray-500">{{ pos.name[:15] }}</div>
                                        </td>
                                        <td class="p-2 text-right">{{ pos.quantity }}</td>
                                        <td class="p-2 text-right">₩{{ "{:,.0f}".format(pos.market_value) }}</td>
                                        <td class="p-2 text-right {% if pos.pnl_pct >= 0 %}positive{% else %}negative{% endif %}">
                                            {{ "{:+.1f}".format(pos.pnl_pct) }}%
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <!-- Recent Trades -->
                    <div>
                        <h3 class="font-bold mb-3">📜 Recent Trades</h3>
                        <div class="overflow-x-auto max-h-64 overflow-y-auto">
                            <table class="w-full text-sm">
                                <thead class="bg-gray-50 text-xs text-gray-500 uppercase sticky top-0">
                                    <tr>
                                        <th class="p-2 text-left">Time</th>
                                        <th class="p-2 text-left">Side</th>
                                        <th class="p-2 text-left">Symbol</th>
                                        <th class="p-2 text-right">Qty</th>
                                        <th class="p-2 text-right">Price</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for trade in trades %}
                                    <tr class="border-t hover:bg-gray-50">
                                        <td class="p-2 text-xs text-gray-500">{{ trade.timestamp }}</td>
                                        <td class="p-2">
                                            <span class="px-2 py-0.5 rounded text-xs font-medium
                                                {% if trade.side == 'buy' %}bg-green-100 text-green-800{% else %}bg-red-100 text-red-800{% endif %}">
                                                {{ trade.side | upper }}
                                            </span>
                                        </td>
                                        <td class="p-2 font-medium">{{ trade.symbol if trade.symbol else trade.name }}</td>
                                        <td class="p-2 text-right">{{ trade.quantity }}</td>
                                        <td class="p-2 text-right">₩{{ "{:,.0f}".format(trade.price) }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Hypotheses Tab -->
            <div id="content-hypotheses" class="p-4 hidden">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="font-bold">Trading Strategies ({{ hypotheses | length }})</h3>
                    <div class="flex gap-2">
                        <select id="hypothesis-filter" onchange="filterHypotheses()" class="border rounded px-3 py-1 text-sm">
                            <option value="all">All Status</option>
                            <option value="live">Live</option>
                            <option value="paper_trading">Paper Trading</option>
                            <option value="incubating">Incubating</option>
                            <option value="research">Research</option>
                        </select>
                    </div>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead class="bg-gray-50 text-xs text-gray-500 uppercase">
                            <tr>
                                <th class="p-3 text-left">ID / Name</th>
                                <th class="p-3 text-center">Status</th>
                                <th class="p-3 text-center">Type</th>
                                <th class="p-3 text-right">Sharpe</th>
                                <th class="p-3 text-right">Win%</th>
                                <th class="p-3 text-right">Return</th>
                                <th class="p-3 text-right">Max DD</th>
                                <th class="p-3 text-right">Trades</th>
                                <th class="p-3 text-right">Score</th>
                                <th class="p-3 text-center">Actions</th>
                            </tr>
                        </thead>
                        <tbody id="hypotheses-table">
                            {% for hyp in hypotheses %}
                            <tr class="border-t hover:bg-gray-50 hypothesis-row" data-status="{{ hyp.status }}">
                                <td class="p-3">
                                    <div class="font-medium">{{ hyp.id_short }}</div>
                                    <div class="text-xs text-gray-500" title="{{ hyp.name }}">{{ hyp.name[:30] }}{% if hyp.name|length > 30 %}...{% endif %}</div>
                                </td>
                                <td class="p-3 text-center">
                                    <span class="px-2 py-1 rounded-full text-xs font-medium
                                        {% if hyp.status == 'live' %}bg-green-100 text-green-800
                                        {% elif hyp.status == 'paper_trading' %}bg-blue-100 text-blue-800
                                        {% elif hyp.status == 'incubating' %}bg-yellow-100 text-yellow-800
                                        {% elif hyp.status == 'research' %}bg-purple-100 text-purple-800
                                        {% else %}bg-gray-100 text-gray-800{% endif %}">
                                        {{ hyp.status }}
                                    </span>
                                </td>
                                <td class="p-3 text-center text-xs">{{ hyp.strategy_type }}</td>
                                <td class="p-3 text-right {% if hyp.sharpe >= 1 %}positive{% elif hyp.sharpe < 0 %}negative{% endif %}">
                                    {{ "{:.2f}".format(hyp.sharpe) }}
                                </td>
                                <td class="p-3 text-right">{{ "{:.0f}".format(hyp.win_rate) }}%</td>
                                <td class="p-3 text-right {% if hyp.total_return >= 0 %}positive{% else %}negative{% endif %}">
                                    {{ "{:+.1f}".format(hyp.total_return) }}%
                                </td>
                                <td class="p-3 text-right text-red-500">{{ "{:.1f}".format(hyp.max_drawdown) }}%</td>
                                <td class="p-3 text-right">{{ hyp.total_trades }}</td>
                                <td class="p-3 text-right">
                                    {% if hyp.promotion_score > 0 %}
                                    <span class="text-blue-600">{{ "{:.2f}".format(hyp.promotion_score) }}</span>
                                    {% else %}-{% endif %}
                                </td>
                                <td class="p-3 text-center">
                                    <div class="flex gap-1 justify-center">
                                        {% if hyp.status in ['live', 'paper_trading'] %}
                                        <button onclick="pauseHypothesis('{{ hyp.id }}')" class="text-yellow-600 hover:text-yellow-800" title="Pause">⏸️</button>
                                        {% endif %}
                                        {% if hyp.status in ['incubating', 'paper_trading'] %}
                                        <button onclick="promoteHypothesis('{{ hyp.id }}')" class="text-green-600 hover:text-green-800" title="Promote">⬆️</button>
                                        {% endif %}
                                        <button onclick="retireHypothesis('{{ hyp.id }}')" class="text-red-600 hover:text-red-800" title="Retire">🗑️</button>
                                    </div>
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Decisions Tab -->
            <div id="content-decisions" class="p-4 hidden">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="font-bold">Decision Journal</h3>
                    <select id="decision-filter" onchange="loadDecisions()" class="border rounded px-3 py-1 text-sm">
                        <option value="">All Types</option>
                        <option value="trade_entry">Trade Entry</option>
                        <option value="trade_exit">Trade Exit</option>
                        <option value="hypothesis_create">Hypothesis Create</option>
                        <option value="hypothesis_promote">Hypothesis Promote</option>
                        <option value="hypothesis_retire">Hypothesis Retire</option>
                        <option value="risk_reduction">Risk Reduction</option>
                    </select>
                </div>
                <div id="decisions-list" class="space-y-3 max-h-96 overflow-y-auto">
                    {% for decision in decisions %}
                    <div class="bg-gray-50 rounded p-3 border-l-4
                        {% if 'trade' in decision.decision_type %}border-blue-500
                        {% elif 'hypothesis' in decision.decision_type %}border-purple-500
                        {% elif 'risk' in decision.decision_type %}border-red-500
                        {% else %}border-gray-500{% endif %}">
                        <div class="flex justify-between items-start">
                            <div>
                                <span class="font-medium">{{ decision.description }}</span>
                                <span class="text-xs ml-2 px-2 py-0.5 bg-gray-200 rounded">{{ decision.decision_type }}</span>
                            </div>
                            <span class="text-xs text-gray-500">{{ decision.timestamp }}</span>
                        </div>
                        {% if decision.reasoning %}
                        <div class="text-sm text-gray-600 mt-1">{{ decision.reasoning[:150] }}{% if decision.reasoning|length > 150 %}...{% endif %}</div>
                        {% endif %}
                        <div class="flex gap-4 mt-2 text-xs text-gray-500">
                            <span>Confidence: {{ "{:.0f}".format((decision.confidence or 0) * 100) }}%</span>
                            {% if decision.outcome_quality is not none %}
                            <span class="{% if decision.outcome_quality > 0 %}text-green-600{% else %}text-red-600{% endif %}">
                                Quality: {{ "{:+.0f}".format(decision.outcome_quality * 100) }}%
                            </span>
                            {% endif %}
                        </div>
                    </div>
                    {% else %}
                    <div class="text-gray-500 text-center py-8">No decisions recorded yet</div>
                    {% endfor %}
                </div>
            </div>

            <!-- Research Tab -->
            <div id="content-research" class="p-4 hidden">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <!-- Papers -->
                    <div>
                        <h3 class="font-bold mb-3">📄 Recent Papers</h3>
                        <div id="papers-list" class="space-y-2 max-h-80 overflow-y-auto">
                            <p class="text-gray-500 text-sm">Loading...</p>
                        </div>
                    </div>
                    <!-- Ideas -->
                    <div>
                        <h3 class="font-bold mb-3">💡 Extracted Ideas</h3>
                        <div id="ideas-list" class="space-y-2 max-h-80 overflow-y-auto">
                            <p class="text-gray-500 text-sm">Loading...</p>
                        </div>
                    </div>
                </div>
                <div class="mt-6 p-4 bg-gray-50 rounded" id="research-stats">
                    <!-- Stats loaded via JS -->
                </div>
            </div>

            <!-- Analytics Tab -->
            <div id="content-analytics" class="p-4 hidden">
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                    <div class="bg-gray-50 rounded p-4 text-center">
                        <div class="text-3xl font-bold" id="stat-total-trades">-</div>
                        <div class="text-gray-500 text-sm">Total Trades</div>
                    </div>
                    <div class="bg-gray-50 rounded p-4 text-center">
                        <div class="text-3xl font-bold" id="stat-win-rate">-</div>
                        <div class="text-gray-500 text-sm">Win Rate</div>
                    </div>
                    <div class="bg-gray-50 rounded p-4 text-center">
                        <div class="text-3xl font-bold" id="stat-avg-return">-</div>
                        <div class="text-gray-500 text-sm">Avg Return</div>
                    </div>
                    <div class="bg-gray-50 rounded p-4 text-center">
                        <div class="text-3xl font-bold" id="stat-holding-period">-</div>
                        <div class="text-gray-500 text-sm">Avg Holding (days)</div>
                    </div>
                </div>
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div>
                        <h3 class="font-bold mb-3">Returns Distribution</h3>
                        <div class="bg-gray-50 rounded p-4" style="height: 250px;">
                            <canvas id="returnsChart"></canvas>
                        </div>
                    </div>
                    <div>
                        <h3 class="font-bold mb-3">Performance by Strategy</h3>
                        <div class="bg-gray-50 rounded p-4" style="height: 250px;">
                            <canvas id="strategyChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="bg-gray-800 text-white p-4 text-center text-sm mt-8">
        <p>Living Trading System v0.2.0 | Last update: <span id="last-update">{{ status.last_update }}</span></p>
    </footer>

    <script>
        // Tab Management
        function showTab(tabName) {
            ['overview', 'hypotheses', 'decisions', 'research', 'analytics'].forEach(t => {
                document.getElementById('content-' + t).classList.add('hidden');
                document.getElementById('tab-' + t).classList.remove('tab-active');
                document.getElementById('tab-' + t).classList.add('text-gray-500');
            });
            document.getElementById('content-' + tabName).classList.remove('hidden');
            document.getElementById('tab-' + tabName).classList.add('tab-active');
            document.getElementById('tab-' + tabName).classList.remove('text-gray-500');

            // Load data for specific tabs
            if (tabName === 'research') loadResearch();
            if (tabName === 'analytics') loadAnalytics();
        }

        // Control functions
        async function pauseSystem() {
            if (confirm('Pause the trading system?')) {
                const res = await fetch('/api/control/pause', { method: 'POST' });
                const data = await res.json();
                alert(data.message);
                location.reload();
            }
        }

        async function resumeSystem() {
            if (confirm('Resume the trading system?')) {
                const res = await fetch('/api/control/resume', { method: 'POST' });
                const data = await res.json();
                alert(data.message);
                location.reload();
            }
        }

        async function stopSystem() {
            if (confirm('⚠️ EMERGENCY STOP - This will halt all trading. Continue?')) {
                const res = await fetch('/api/control/stop', { method: 'POST' });
                const data = await res.json();
                alert(data.message);
                location.reload();
            }
        }

        // Hypothesis controls
        async function pauseHypothesis(id) {
            if (confirm('Pause this strategy?')) {
                const res = await fetch(`/api/hypothesis/${id}/pause`, { method: 'POST' });
                const data = await res.json();
                alert(data.message);
                location.reload();
            }
        }

        async function promoteHypothesis(id) {
            if (confirm('Force promote this strategy?')) {
                const res = await fetch(`/api/hypothesis/${id}/promote`, { method: 'POST' });
                const data = await res.json();
                alert(data.message);
                location.reload();
            }
        }

        async function retireHypothesis(id) {
            if (confirm('Retire this strategy?')) {
                const res = await fetch(`/api/hypothesis/${id}/retire`, { method: 'POST' });
                const data = await res.json();
                alert(data.message);
                location.reload();
            }
        }

        // Filter hypotheses
        function filterHypotheses() {
            const filter = document.getElementById('hypothesis-filter').value;
            document.querySelectorAll('.hypothesis-row').forEach(row => {
                if (filter === 'all' || row.dataset.status === filter) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        }

        // Load decisions
        async function loadDecisions() {
            const filter = document.getElementById('decision-filter').value;
            const url = filter ? `/api/decisions?decision_type=${filter}` : '/api/decisions';
            const res = await fetch(url);
            const decisions = await res.json();

            const container = document.getElementById('decisions-list');
            if (decisions.length === 0) {
                container.innerHTML = '<div class="text-gray-500 text-center py-8">No decisions found</div>';
                return;
            }

            container.innerHTML = decisions.map(d => `
                <div class="bg-gray-50 rounded p-3 border-l-4 ${d.decision_type.includes('trade') ? 'border-blue-500' : d.decision_type.includes('hypothesis') ? 'border-purple-500' : 'border-gray-500'}">
                    <div class="flex justify-between items-start">
                        <div>
                            <span class="font-medium">${d.description}</span>
                            <span class="text-xs ml-2 px-2 py-0.5 bg-gray-200 rounded">${d.decision_type}</span>
                        </div>
                        <span class="text-xs text-gray-500">${d.timestamp}</span>
                    </div>
                    ${d.reasoning ? `<div class="text-sm text-gray-600 mt-1">${d.reasoning.substring(0, 150)}${d.reasoning.length > 150 ? '...' : ''}</div>` : ''}
                    <div class="flex gap-4 mt-2 text-xs text-gray-500">
                        <span>Confidence: ${Math.round((d.confidence || 0) * 100)}%</span>
                    </div>
                </div>
            `).join('');
        }

        // Load research
        async function loadResearch() {
            const res = await fetch('/api/research');
            const data = await res.json();

            // Papers
            const papersHtml = data.papers.length ? data.papers.map(p => `
                <div class="bg-white rounded p-2 shadow-sm">
                    <div class="font-medium text-sm">${p.title}</div>
                    <div class="text-xs text-gray-500">${p.authors.join(', ')} | ${p.source}</div>
                    <div class="flex gap-1 mt-1">${p.keywords.map(k => `<span class="text-xs bg-blue-100 text-blue-800 px-1 rounded">${k}</span>`).join('')}</div>
                </div>
            `).join('') : '<p class="text-gray-500">No papers fetched yet</p>';
            document.getElementById('papers-list').innerHTML = papersHtml;

            // Ideas
            const ideasHtml = data.ideas.length ? data.ideas.map(i => `
                <div class="bg-white rounded p-2 shadow-sm">
                    <div class="font-medium text-sm">${i.title}</div>
                    <div class="text-xs">
                        <span class="bg-purple-100 text-purple-800 px-1 rounded">${i.type}</span>
                        <span class="ml-1 ${i.confidence === 'high' ? 'text-green-600' : i.confidence === 'medium' ? 'text-yellow-600' : 'text-gray-600'}">${i.confidence}</span>
                    </div>
                </div>
            `).join('') : '<p class="text-gray-500">No ideas extracted yet</p>';
            document.getElementById('ideas-list').innerHTML = ideasHtml;

            // Stats
            const stats = data.knowledge_stats || {};
            document.getElementById('research-stats').innerHTML = `
                <div class="grid grid-cols-4 gap-4 text-center">
                    <div><div class="text-2xl font-bold">${data.papers.length}</div><div class="text-sm text-gray-500">Papers</div></div>
                    <div><div class="text-2xl font-bold">${data.ideas.length}</div><div class="text-sm text-gray-500">Ideas</div></div>
                    <div><div class="text-2xl font-bold">${data.hypotheses_generated}</div><div class="text-sm text-gray-500">Hypotheses</div></div>
                    <div><div class="text-2xl font-bold">${stats.total_entries || 0}</div><div class="text-sm text-gray-500">Knowledge Entries</div></div>
                </div>
            `;
        }

        // Load analytics
        async function loadAnalytics() {
            const res = await fetch('/api/performance-analytics');
            const data = await res.json();

            document.getElementById('stat-total-trades').textContent = data.total_trades || 0;
            document.getElementById('stat-win-rate').textContent = (data.overall?.win_rate * 100 || 0).toFixed(0) + '%';
            document.getElementById('stat-avg-return').textContent = (data.overall?.avg_return * 100 || 0).toFixed(1) + '%';
            document.getElementById('stat-holding-period').textContent = (data.overall?.avg_holding_period || 0).toFixed(1);
        }

        // Dismiss alert
        async function dismissAlert(id) {
            await fetch(`/api/alerts/read/${id}`, { method: 'POST' });
            location.reload();
        }

        // Initialize Charts
        let equityChart, allocationChart;

        async function initCharts() {
            // Equity Chart
            const equityRes = await fetch('/api/equity-curve?days=30');
            const equityData = await equityRes.json();

            const ctx1 = document.getElementById('equityChart').getContext('2d');
            equityChart = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: equityData.dates.length ? equityData.dates : ['No data'],
                    datasets: [{
                        label: 'Equity',
                        data: equityData.equity.length ? equityData.equity : [0],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: false }
                    }
                }
            });

            // Allocation Chart
            const allocRes = await fetch('/api/capital-allocation');
            const allocData = await allocRes.json();

            const ctx2 = document.getElementById('allocationChart').getContext('2d');
            const allocLabels = ['Available', 'Deployed', 'Reserved'];
            const allocValues = [allocData.available_capital, allocData.deployed_capital, allocData.reserved_capital];

            allocationChart = new Chart(ctx2, {
                type: 'doughnut',
                data: {
                    labels: allocLabels,
                    datasets: [{
                        data: allocValues,
                        backgroundColor: ['#10b981', '#3b82f6', '#f59e0b']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom' }
                    }
                }
            });
        }

        // WebSocket for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/ws`);

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'update') {
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            }
        };

        ws.onclose = function() {
            console.log('WebSocket closed, reconnecting...');
            setTimeout(() => location.reload(), 10000);
        };

        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
        });
    </script>
</body>
</html>'''

    template_path = TEMPLATES_DIR / "dashboard.html"
    with open(template_path, 'w') as f:
        f.write(dashboard_html)

    logger.info(f"Created dashboard template at {template_path}")


if __name__ == "__main__":
    run_dashboard()
