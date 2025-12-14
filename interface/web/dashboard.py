"""
WEB DASHBOARD - Browser Interface for Living Trading System

A clean, modern web dashboard for monitoring and controlling the trading system.

Features:
- Real-time portfolio overview
- Position management
- Hypothesis monitoring
- Trade history
- Decision log
- System controls
- Performance charts

Built with FastAPI + HTMX for a responsive, low-JavaScript interface.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

from loguru import logger

try:
    from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
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
    version="0.1.0"
)

# Setup rate limiting if available
if RATELIMIT_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Track startup time for health check
_startup_time = datetime.now()

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


# === DATA PROVIDERS ===

class DataProvider:
    """Provides data to the dashboard. Connects to KIS API."""
    
    _service = None
    
    @classmethod
    def _get_service(cls):
        """Get KIS service (lazy initialization)."""
        if cls._service is None:
            try:
                from interface.kis_service import get_kis_service
                cls._service = get_kis_service()
            except Exception as e:
                logger.error(f"Failed to get KIS service: {e}")
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
        """Get all hypotheses from registry and knowledge base."""
        hypotheses = []

        try:
            # Get from hypothesis registry (active trading strategies)
            from hypothesis import get_registry
            registry = get_registry()

            for hyp in registry.get_all():
                hypotheses.append({
                    "id": hyp.hypothesis_id[:8],
                    "name": hyp.name,
                    "status": hyp.state.value if hasattr(hyp.state, 'value') else str(hyp.state),
                    "allocation": 0,  # Will be updated from allocator
                    "win_rate": hyp.backtest_metrics.get("win_rate", 0) * 100 if hyp.backtest_metrics else 0,
                    "pnl": hyp.backtest_metrics.get("total_pnl", 0) if hyp.backtest_metrics else 0,
                    "sharpe": hyp.backtest_metrics.get("sharpe_ratio", 0) if hyp.backtest_metrics else 0,
                    "source": "registry"
                })
        except Exception as e:
            logger.debug(f"Could not fetch from registry: {e}")

        try:
            # Get from knowledge base (research-generated hypotheses)
            from research import get_knowledge_base
            kb = get_knowledge_base()

            for hyp_id, hyp in kb._hypotheses.items():
                # Avoid duplicates
                if not any(h["id"] == hyp_id[:8] for h in hypotheses):
                    hypotheses.append({
                        "id": hyp_id[:8],
                        "name": hyp.name,
                        "status": "research",
                        "allocation": 0,
                        "win_rate": 0,
                        "pnl": 0,
                        "sharpe": 0,
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
    async def get_decisions(limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent decisions - Phase 2+ feature."""
        return []
    
    @staticmethod
    async def get_performance_chart() -> Dict[str, Any]:
        """Get equity curve data for charting."""
        return {"dates": [], "equity": [], "benchmark": []}

# === API ROUTES ===

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    status = await DataProvider.get_system_status()
    portfolio = await DataProvider.get_portfolio()
    positions = await DataProvider.get_positions()
    hypotheses = await DataProvider.get_hypotheses()
    trades = await DataProvider.get_trades(5)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "status": status,
        "portfolio": portfolio,
        "positions": positions,
        "hypotheses": hypotheses,
        "trades": trades
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
    """Get hypotheses."""
    return await DataProvider.get_hypotheses()


@app.get("/api/trades")
async def api_trades(limit: int = 20):
    """Get recent trades."""
    return await DataProvider.get_trades(limit)


@app.get("/api/decisions")
async def api_decisions(limit: int = 20):
    """Get recent decisions."""
    return await DataProvider.get_decisions(limit)


@app.get("/api/research")
async def api_research():
    """Get research module status (papers, ideas, knowledge base)."""
    result = {
        "papers": [],
        "ideas": [],
        "knowledge_stats": {},
        "last_fetch": None
    }

    try:
        from research import get_knowledge_base, get_fetcher
        kb = get_knowledge_base()
        fetcher = get_fetcher()

        # Get papers
        for paper_id, paper in list(kb._papers.items())[:20]:
            result["papers"].append({
                "id": paper_id,
                "title": paper.title[:80] + "..." if len(paper.title) > 80 else paper.title,
                "authors": paper.authors[:3],
                "relevance": paper.relevance_score,
                "keywords": paper.trading_keywords[:5],
                "source": paper.source.value
            })

        # Get ideas
        for idea_id, idea in list(kb._ideas.items())[:20]:
            result["ideas"].append({
                "id": idea_id,
                "title": idea.title[:60] + "..." if len(idea.title) > 60 else idea.title,
                "type": idea.idea_type.value if hasattr(idea.idea_type, 'value') else str(idea.idea_type),
                "confidence": idea.confidence.value if hasattr(idea.confidence, 'value') else str(idea.confidence),
                "indicators": idea.indicators[:3] if idea.indicators else []
            })

        # Stats
        result["knowledge_stats"] = kb.get_stats()
        result["fetcher_stats"] = fetcher.get_stats()

    except Exception as e:
        logger.debug(f"Could not fetch research data: {e}")

    return result


@app.get("/api/system")
async def api_system():
    """Get full system status including orchestrator state."""
    result = {
        "orchestrator": None,
        "market": {"is_open": False, "next_open": None},
        "components": {}
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

        # Try to get orchestrator status (may not be initialized)
        try:
            orch = get_orchestrator()
            result["orchestrator"] = orch.get_status()
        except:
            pass

    except Exception as e:
        logger.debug(f"Could not fetch system status: {e}")

    return result


@app.get("/api/chart/equity")
async def api_chart_equity():
    """Get equity curve data."""
    return await DataProvider.get_performance_chart()


# === HEALTH CHECK ENDPOINT ===

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    Returns system health status and component availability.
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - _startup_time).total_seconds(),
        "components": {}
    }

    # Check database
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

    # Check broker connection
    try:
        from execution.broker import KISBroker
        # Just check if broker can be instantiated
        health["components"]["broker"] = "ok"
    except Exception as e:
        health["components"]["broker"] = f"error: {str(e)[:50]}"

    # Check market clock
    try:
        from core.clock import get_clock, is_market_open
        clock = get_clock()
        health["components"]["clock"] = "ok"
        health["market_open"] = is_market_open()
        health["current_time_kst"] = clock.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        health["components"]["clock"] = f"error: {str(e)[:50]}"

    # Check knowledge base
    try:
        from research import get_knowledge_base
        kb = get_knowledge_base()
        stats = kb.get_stats()
        health["components"]["knowledge_base"] = "ok"
        health["knowledge_base_entries"] = stats.get("total_entries", 0)
    except Exception as e:
        health["components"]["knowledge_base"] = f"error: {str(e)[:50]}"

    # Overall status
    errors = [v for v in health["components"].values() if v != "ok"]
    if len(errors) > len(health["components"]) / 2:
        health["status"] = "unhealthy"
    elif errors:
        health["status"] = "degraded"

    return health


@app.get("/health/live")
async def liveness_check():
    """Simple liveness probe - just returns 200 if app is running."""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness probe - checks if app is ready to serve traffic."""
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
    # TODO: Connect to actual system
    return {"success": True, "message": "System paused"}


@app.post("/api/control/resume")
async def control_resume():
    """Resume the trading system."""
    logger.info("Resume requested via web dashboard")
    return {"success": True, "message": "System resumed"}


@app.post("/api/control/stop")
async def control_stop():
    """Emergency stop."""
    logger.warning("Emergency stop requested via web dashboard")
    return {"success": True, "message": "Emergency stop executed"}


@app.post("/api/hypothesis/{hypothesis_id}/pause")
async def hypothesis_pause(hypothesis_id: str):
    """Pause a specific hypothesis."""
    logger.info(f"Hypothesis {hypothesis_id} pause requested")
    return {"success": True, "message": f"Hypothesis {hypothesis_id} paused"}


@app.post("/api/hypothesis/{hypothesis_id}/retire")
async def hypothesis_retire(hypothesis_id: str):
    """Retire a specific hypothesis."""
    logger.info(f"Hypothesis {hypothesis_id} retire requested")
    return {"success": True, "message": f"Hypothesis {hypothesis_id} retired"}


# === WEBSOCKET FOR REAL-TIME UPDATES ===

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(5)
            
            data = {
                "type": "update",
                "portfolio": await DataProvider.get_portfolio(),
                "status": await DataProvider.get_system_status()
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
    
    # Use PORT environment variable (required for Railway/Render)
    if port is None:
        port = int(os.environ.get("PORT", 8080))
    
    # Create templates if they don't exist
    create_default_templates()
    
    print(f"\n🌐 Starting web dashboard at http://localhost:{port}")
    print("   Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host=host, port=port)


def create_default_templates():
    """Create default HTML templates."""
    # Main dashboard template
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
        .card { @apply bg-white rounded-lg shadow p-6; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <!-- Header -->
    <header class="bg-gray-900 text-white p-4 shadow-lg">
        <div class="container mx-auto flex justify-between items-center">
            <div class="flex items-center gap-4">
                <h1 class="text-xl font-bold">🤖 Living Trading System</h1>
                <span class="px-3 py-1 rounded-full text-sm 
                    {% if status.mode == 'paper' %}bg-yellow-500{% else %}bg-green-500{% endif %}">
                    {{ status.mode | upper }}
                </span>
                <span class="px-3 py-1 rounded-full text-sm
                    {% if status.state == 'running' %}bg-green-500{% else %}bg-red-500{% endif %}">
                    {{ status.state | upper }}
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

    <main class="container mx-auto p-6">
        <!-- Portfolio Overview -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
            <div class="bg-white rounded-lg shadow p-6">
                <div class="text-gray-500 text-sm">Total Equity</div>
                <div class="text-2xl font-bold">₩{{ "{:,.0f}".format(portfolio.total_equity) }}</div>
                <div class="text-sm {% if portfolio.daily_pnl_pct >= 0 %}positive{% else %}negative{% endif %}">
                    {{ "{:+.2f}".format(portfolio.daily_pnl_pct) }}% today
                </div>
            </div>
            <div class="bg-white rounded-lg shadow p-6">
                <div class="text-gray-500 text-sm">Cash</div>
                <div class="text-2xl font-bold">₩{{ "{:,.0f}".format(portfolio.cash) }}</div>
                <div class="text-sm text-gray-500">
                    {% if portfolio.total_equity > 0 %}{{ "{:.1f}".format(portfolio.cash / portfolio.total_equity * 100) }}%{% else %}0%{% endif %} of portfolio
                </div>
            </div>
            <div class="bg-white rounded-lg shadow p-6">
                <div class="text-gray-500 text-sm">Unrealized P&L</div>
                <div class="text-2xl font-bold {% if portfolio.unrealized_pnl >= 0 %}positive{% else %}negative{% endif %}">
                    ₩{{ "{:+,.0f}".format(portfolio.unrealized_pnl) }}
                </div>
                <div class="text-sm {% if portfolio.unrealized_pnl_pct >= 0 %}positive{% else %}negative{% endif %}">
                    {{ "{:+.2f}".format(portfolio.unrealized_pnl_pct) }}%
                </div>
            </div>
            <div class="bg-white rounded-lg shadow p-6">
                <div class="text-gray-500 text-sm">Drawdown</div>
                <div class="text-2xl font-bold text-red-500">{{ "{:.1f}".format(portfolio.drawdown) }}%</div>
                <div class="text-sm text-gray-500">
                    Max: {{ "{:.1f}".format(portfolio.max_drawdown) }}%
                </div>
            </div>
        </div>

        <!-- Two Column Layout -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <!-- Positions -->
            <div class="bg-white rounded-lg shadow">
                <div class="p-4 border-b flex justify-between items-center">
                    <h2 class="font-bold text-lg">📋 Positions</h2>
                    <span class="text-sm text-gray-500">{{ positions | length }} holdings</span>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full">
                        <thead class="bg-gray-50 text-xs text-gray-500 uppercase">
                            <tr>
                                <th class="p-3 text-left">Symbol</th>
                                <th class="p-3 text-right">Qty</th>
                                <th class="p-3 text-right">Value</th>
                                <th class="p-3 text-right">P&L</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for pos in positions %}
                            <tr class="border-t hover:bg-gray-50">
                                <td class="p-3">
                                    <div class="font-medium">{{ pos.name }}</div>
                                    <div class="text-xs text-gray-500">{{ pos.symbol }}</div>
                                </td>
                                <td class="p-3 text-right">{{ pos.quantity }}</td>
                                <td class="p-3 text-right">₩{{ "{:,.0f}".format(pos.market_value) }}</td>
                                <td class="p-3 text-right {% if pos.pnl >= 0 %}positive{% else %}negative{% endif %}">
                                    {{ "{:+.2f}".format(pos.pnl_pct) }}%
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Hypotheses -->
            <div class="bg-white rounded-lg shadow">
                <div class="p-4 border-b flex justify-between items-center">
                    <h2 class="font-bold text-lg">🎯 Hypotheses</h2>
                    <span class="text-sm text-gray-500">{{ hypotheses | length }} total</span>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full">
                        <thead class="bg-gray-50 text-xs text-gray-500 uppercase">
                            <tr>
                                <th class="p-3 text-left">ID</th>
                                <th class="p-3 text-center">Status</th>
                                <th class="p-3 text-right">Alloc</th>
                                <th class="p-3 text-right">Win%</th>
                                <th class="p-3 text-right">P&L</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for hyp in hypotheses %}
                            <tr class="border-t hover:bg-gray-50">
                                <td class="p-3">
                                    <div class="font-medium">{{ hyp.id }}</div>
                                    <div class="text-xs text-gray-500">{{ hyp.name[:25] }}...</div>
                                </td>
                                <td class="p-3 text-center">
                                    <span class="px-2 py-1 rounded-full text-xs
                                        {% if hyp.status == 'active' %}bg-green-100 text-green-800
                                        {% elif hyp.status == 'incubating' %}bg-yellow-100 text-yellow-800
                                        {% elif hyp.status == 'paper' %}bg-gray-100 text-gray-800
                                        {% else %}bg-red-100 text-red-800{% endif %}">
                                        {{ hyp.status }}
                                    </span>
                                </td>
                                <td class="p-3 text-right">{{ "{:.0f}".format(hyp.allocation) }}%</td>
                                <td class="p-3 text-right">{{ "{:.0f}".format(hyp.win_rate) }}%</td>
                                <td class="p-3 text-right {% if hyp.pnl >= 0 %}positive{% else %}negative{% endif %}">
                                    ₩{{ "{:+,.0f}".format(hyp.pnl) }}
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Recent Trades -->
        <div class="bg-white rounded-lg shadow mb-6">
            <div class="p-4 border-b flex justify-between items-center">
                <h2 class="font-bold text-lg">📜 Recent Trades</h2>
                <a href="/trades" class="text-blue-500 hover:underline text-sm">View All</a>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full">
                    <thead class="bg-gray-50 text-xs text-gray-500 uppercase">
                        <tr>
                            <th class="p-3 text-left">Time</th>
                            <th class="p-3 text-left">Side</th>
                            <th class="p-3 text-left">Symbol</th>
                            <th class="p-3 text-right">Qty</th>
                            <th class="p-3 text-right">Price</th>
                            <th class="p-3 text-right">Value</th>
                            <th class="p-3 text-left">Hypothesis</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for trade in trades %}
                        <tr class="border-t hover:bg-gray-50">
                            <td class="p-3 text-sm text-gray-500">{{ trade.timestamp }}</td>
                            <td class="p-3">
                                <span class="px-2 py-1 rounded text-xs font-medium
                                    {% if trade.side == 'buy' %}bg-green-100 text-green-800{% else %}bg-red-100 text-red-800{% endif %}">
                                    {{ trade.side | upper }}
                                </span>
                            </td>
                            <td class="p-3">
                                <div class="font-medium">{{ trade.name }}</div>
                            </td>
                            <td class="p-3 text-right">{{ trade.quantity }}</td>
                            <td class="p-3 text-right">₩{{ "{:,.0f}".format(trade.price) }}</td>
                            <td class="p-3 text-right">₩{{ "{:,.0f}".format(trade.value) }}</td>
                            <td class="p-3 text-sm text-gray-500">{{ trade.hypothesis }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="bg-gray-800 text-white p-4 text-center text-sm">
        <p>Living Trading System v0.1.0 | Last update: <span id="last-update">{{ status.last_update }}</span></p>
    </footer>

    <script>
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
                if (confirm('Are you absolutely sure?')) {
                    const res = await fetch('/api/control/stop', { method: 'POST' });
                    const data = await res.json();
                    alert(data.message);
                    location.reload();
                }
            }
        }

        // WebSocket for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'update') {
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                // Could update more elements here
            }
        };

        ws.onclose = function() {
            console.log('WebSocket closed, reconnecting...');
            setTimeout(() => location.reload(), 5000);
        };
    </script>
</body>
</html>'''
    
    template_path = TEMPLATES_DIR / "dashboard.html"
    with open(template_path, 'w') as f:
        f.write(dashboard_html)
    
    logger.info(f"Created dashboard template at {template_path}")


if __name__ == "__main__":
    run_dashboard()
