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
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. Run: pip install fastapi uvicorn jinja2")


# === APPLICATION SETUP ===

app = FastAPI(
    title="Living Trading System",
    description="Web dashboard for the Living Trading System",
    version="0.1.0"
)

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


# === DATA PROVIDERS (mock for now) ===

class DataProvider:
    """Provides data to the dashboard. Connects to the actual system."""
    
    @staticmethod
    async def get_system_status() -> Dict[str, Any]:
        """Get current system status."""
        return {
            "mode": "paper",
            "state": "running",
            "uptime": "2h 34m",
            "market_status": "open",
            "last_update": datetime.now().strftime("%H:%M:%S")
        }
    
    @staticmethod
    async def get_portfolio() -> Dict[str, Any]:
        """Get portfolio summary."""
        return {
            "total_equity": 10234500,
            "cash": 2341200,
            "positions_value": 7893300,
            "unrealized_pnl": 156700,
            "unrealized_pnl_pct": 1.53,
            "daily_pnl": 42800,
            "daily_pnl_pct": 0.42,
            "weekly_pnl_pct": 1.23,
            "monthly_pnl_pct": 3.87,
            "drawdown": 2.1,
            "max_drawdown": 5.4,
            "high_water_mark": 10450000
        }
    
    @staticmethod
    async def get_positions() -> List[Dict[str, Any]]:
        """Get current positions."""
        return [
            {
                "symbol": "069500",
                "name": "KODEX 200",
                "quantity": 15,
                "avg_cost": 34250,
                "current_price": 35038,
                "market_value": 525570,
                "pnl": 11820,
                "pnl_pct": 2.3,
                "weight": 5.1,
                "hypothesis": "hyp_a3f2"
            },
            {
                "symbol": "360750",
                "name": "TIGER 미국S&P500",
                "quantity": 8,
                "avg_cost": 15200,
                "current_price": 15367,
                "market_value": 122936,
                "pnl": 1336,
                "pnl_pct": 1.1,
                "weight": 1.2,
                "hypothesis": "hyp_b7c1"
            },
            {
                "symbol": "005930",
                "name": "삼성전자",
                "quantity": 5,
                "avg_cost": 72000,
                "current_price": 71712,
                "market_value": 358560,
                "pnl": -1440,
                "pnl_pct": -0.4,
                "weight": 3.5,
                "hypothesis": "hyp_a3f2"
            },
        ]
    
    @staticmethod
    async def get_hypotheses() -> List[Dict[str, Any]]:
        """Get all hypotheses."""
        return [
            {
                "id": "hyp_a3f2",
                "name": "Momentum ETF Rotation",
                "status": "active",
                "allocation": 23.0,
                "trades": 47,
                "win_rate": 62.0,
                "pnl": 234000,
                "sharpe": 1.42,
                "created": "2024-10-15",
                "last_signal": "14:32"
            },
            {
                "id": "hyp_b7c1",
                "name": "Mean Reversion Bounce",
                "status": "active",
                "allocation": 18.0,
                "trades": 32,
                "win_rate": 58.0,
                "pnl": 156000,
                "sharpe": 1.18,
                "created": "2024-10-22",
                "last_signal": "11:15"
            },
            {
                "id": "hyp_c9d4",
                "name": "Volatility Breakout",
                "status": "incubating",
                "allocation": 5.0,
                "trades": 12,
                "win_rate": 54.0,
                "pnl": 12000,
                "sharpe": 0.89,
                "created": "2024-11-01",
                "last_signal": "09:45"
            },
            {
                "id": "hyp_d2e8",
                "name": "Sector Rotation v2",
                "status": "paper",
                "allocation": 0.0,
                "trades": 28,
                "win_rate": 51.0,
                "pnl": -8500,
                "sharpe": 0.34,
                "created": "2024-11-10",
                "last_signal": "15:20"
            },
        ]
    
    @staticmethod
    async def get_trades(limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trades."""
        return [
            {
                "id": "trd_001",
                "timestamp": "2024-12-05 14:32:15",
                "symbol": "069500",
                "name": "KODEX 200",
                "side": "buy",
                "quantity": 15,
                "price": 34250,
                "value": 513750,
                "hypothesis": "hyp_a3f2",
                "status": "filled"
            },
            {
                "id": "trd_002",
                "timestamp": "2024-12-05 11:15:42",
                "symbol": "091230",
                "name": "TIGER 반도체",
                "side": "sell",
                "quantity": 10,
                "price": 12340,
                "value": 123400,
                "hypothesis": "hyp_b7c1",
                "status": "filled"
            },
            {
                "id": "trd_003",
                "timestamp": "2024-12-04 14:55:03",
                "symbol": "005930",
                "name": "삼성전자",
                "side": "buy",
                "quantity": 5,
                "price": 72000,
                "value": 360000,
                "hypothesis": "hyp_a3f2",
                "status": "filled"
            },
        ][:limit]
    
    @staticmethod
    async def get_decisions(limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent decisions."""
        return [
            {
                "id": "dec_001",
                "timestamp": "2024-12-05 14:32:00",
                "type": "trade_entry",
                "description": "Enter long position in KODEX 200",
                "reasoning": "Momentum signal triggered - 20-day ROC above threshold",
                "confidence": 0.72,
                "outcome": "pending"
            },
            {
                "id": "dec_002",
                "timestamp": "2024-12-05 11:15:00",
                "type": "trade_exit",
                "description": "Exit TIGER 반도체 position",
                "reasoning": "Stop loss triggered at -5% threshold",
                "confidence": 0.95,
                "outcome": "executed"
            },
            {
                "id": "dec_003",
                "timestamp": "2024-12-05 09:00:00",
                "type": "allocation",
                "description": "Increase allocation to hyp_a3f2",
                "reasoning": "Rolling Sharpe ratio improved to 1.42",
                "confidence": 0.68,
                "outcome": "executed"
            },
        ][:limit]
    
    @staticmethod
    async def get_performance_chart() -> Dict[str, Any]:
        """Get equity curve data for charting."""
        # Generate sample equity curve
        dates = []
        equity = []
        base = 10000000
        
        for i in range(60):
            date = datetime.now() - timedelta(days=59-i)
            dates.append(date.strftime("%Y-%m-%d"))
            # Simulate some growth with volatility
            import random
            base *= (1 + random.uniform(-0.02, 0.025))
            equity.append(round(base))
        
        return {
            "dates": dates,
            "equity": equity,
            "benchmark": [10000000 * (1 + 0.0003 * i) for i in range(60)]  # Simple benchmark
        }


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


@app.get("/api/chart/equity")
async def api_chart_equity():
    """Get equity curve data."""
    return await DataProvider.get_performance_chart()


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
                    {{ "{:.1f}".format(portfolio.cash / portfolio.total_equity * 100) }}% of portfolio
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
