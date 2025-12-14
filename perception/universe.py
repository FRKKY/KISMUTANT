"""
UNIVERSE MANAGER - Automatic ETF Discovery and Management

Automatically discovers and manages the tradeable ETF universe from KRX.
Filters by liquidity, AUM, and trading activity.
Categorizes ETFs for strategy allocation.

The system trades ETFs rather than individual stocks because:
1. Diversification built-in
2. Lower single-name risk
3. Easier to model (less idiosyncratic noise)
4. Better liquidity characteristics
5. ISA-eligible for tax efficiency
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Set
from enum import Enum, auto
import json
from pathlib import Path

from loguru import logger

from memory.models import get_database, Instrument
from core.events import get_event_bus, Event, EventType


class ETFCategory(str, Enum):
    """Categories of ETFs for strategy allocation."""
    
    # Domestic Index
    INDEX_KOSPI = "index_kospi"           # KOSPI 200, KOSPI 50, etc.
    INDEX_KOSDAQ = "index_kosdaq"         # KOSDAQ 150, etc.
    INDEX_KRX = "index_krx"               # KRX 300, etc.
    
    # Sector
    SECTOR_TECH = "sector_tech"           # IT, Semiconductor
    SECTOR_FINANCE = "sector_finance"     # Banks, Insurance
    SECTOR_HEALTHCARE = "sector_healthcare"
    SECTOR_ENERGY = "sector_energy"
    SECTOR_MATERIALS = "sector_materials"
    SECTOR_INDUSTRIALS = "sector_industrials"
    SECTOR_CONSUMER = "sector_consumer"
    SECTOR_UTILITIES = "sector_utilities"
    
    # Theme
    THEME_ESG = "theme_esg"
    THEME_DIVIDEND = "theme_dividend"
    THEME_VALUE = "theme_value"
    THEME_GROWTH = "theme_growth"
    THEME_MOMENTUM = "theme_momentum"
    THEME_QUALITY = "theme_quality"
    THEME_LOW_VOL = "theme_low_volatility"
    THEME_BATTERY = "theme_battery"
    THEME_BIO = "theme_bio"
    THEME_AI = "theme_ai"
    
    # International
    INTL_US = "intl_us"                   # S&P 500, NASDAQ, etc.
    INTL_CHINA = "intl_china"
    INTL_JAPAN = "intl_japan"
    INTL_EUROPE = "intl_europe"
    INTL_EMERGING = "intl_emerging"
    INTL_GLOBAL = "intl_global"
    
    # Fixed Income
    BOND_GOVT = "bond_government"
    BOND_CORP = "bond_corporate"
    BOND_CREDIT = "bond_credit"
    
    # Commodity
    COMMODITY_GOLD = "commodity_gold"
    COMMODITY_OIL = "commodity_oil"
    COMMODITY_METALS = "commodity_metals"
    COMMODITY_AGRI = "commodity_agriculture"
    
    # Leverage/Inverse (trade with caution)
    LEVERAGE_2X = "leverage_2x"
    INVERSE_1X = "inverse_1x"
    INVERSE_2X = "inverse_2x"
    
    # Other
    REITS = "reits"
    CURRENCY = "currency"
    OTHER = "other"


@dataclass
class ETFInfo:
    """Detailed information about an ETF."""
    
    symbol: str
    name: str
    category: ETFCategory = ETFCategory.OTHER
    
    # Basic info
    issuer: str = ""                      # Asset manager (Samsung, Mirae, KB, etc.)
    underlying_index: str = ""
    inception_date: Optional[date] = None
    expense_ratio: Optional[float] = None  # TER
    
    # Size & Liquidity
    aum: float = 0.0                      # Assets Under Management (KRW)
    avg_daily_volume: float = 0.0         # 20-day average
    avg_daily_value: float = 0.0          # 20-day average traded value
    shares_outstanding: int = 0
    
    # Trading characteristics
    current_price: float = 0.0
    nav: float = 0.0                      # Net Asset Value
    premium_discount: float = 0.0         # (Price - NAV) / NAV
    tracking_error: Optional[float] = None
    
    # Derived metrics
    bid_ask_spread_bps: float = 0.0       # Average bid-ask spread
    volatility_30d: float = 0.0
    
    # Flags
    is_leverage: bool = False
    is_inverse: bool = False
    leverage_ratio: float = 1.0
    
    # Eligibility
    is_isa_eligible: bool = True          # Tax-advantaged account eligible
    is_tradeable: bool = True
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "category": self.category.value,
            "issuer": self.issuer,
            "aum": self.aum,
            "avg_daily_volume": self.avg_daily_volume,
            "avg_daily_value": self.avg_daily_value,
            "current_price": self.current_price,
            "volatility_30d": self.volatility_30d,
            "is_leverage": self.is_leverage,
            "is_inverse": self.is_inverse,
            "is_tradeable": self.is_tradeable,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class UniverseFilter:
    """Filters for universe construction."""
    
    # Minimum thresholds
    min_aum: float = 10_000_000_000       # 10B KRW minimum AUM
    min_avg_daily_volume: float = 10_000   # 10K shares/day minimum
    min_avg_daily_value: float = 500_000_000  # 500M KRW/day minimum traded value
    min_price: float = 1_000              # Minimum price
    max_price: float = 500_000            # Maximum price
    min_age_days: int = 90                # Minimum 90 days since inception
    
    # Category filters
    include_categories: Optional[Set[ETFCategory]] = None  # None = all
    exclude_categories: Optional[Set[ETFCategory]] = None
    
    # Special filters
    include_leverage: bool = False        # Exclude leverage by default
    include_inverse: bool = False         # Exclude inverse by default
    isa_eligible_only: bool = True        # ISA-eligible only
    
    def passes(self, etf: ETFInfo) -> bool:
        """Check if ETF passes all filters."""
        
        # Size filters
        if etf.aum < self.min_aum:
            return False
        if etf.avg_daily_volume < self.min_avg_daily_volume:
            return False
        if etf.avg_daily_value < self.min_avg_daily_value:
            return False
        
        # Price filters
        if etf.current_price < self.min_price or etf.current_price > self.max_price:
            return False
        
        # Age filter
        if etf.inception_date:
            age = (date.today() - etf.inception_date).days
            if age < self.min_age_days:
                return False
        
        # Category filters
        if self.include_categories and etf.category not in self.include_categories:
            return False
        if self.exclude_categories and etf.category in self.exclude_categories:
            return False
        
        # Leverage/Inverse filters
        if etf.is_leverage and not self.include_leverage:
            return False
        if etf.is_inverse and not self.include_inverse:
            return False
        
        # ISA filter
        if self.isa_eligible_only and not etf.is_isa_eligible:
            return False
        
        # Tradeable check
        if not etf.is_tradeable:
            return False
        
        return True


class UniverseManager:
    """
    Manages the tradeable ETF universe.
    
    Responsibilities:
    1. Discover all ETFs available on KRX
    2. Fetch and update ETF metadata
    3. Categorize ETFs automatically
    4. Apply filters to create tradeable universe
    5. Track universe changes over time
    """
    
    def __init__(self, broker=None):
        """
        Initialize Universe Manager.
        
        Args:
            broker: KIS broker instance for data fetching
        """
        self._broker = broker
        self._db = get_database()
        
        # ETF data cache
        self._etf_cache: Dict[str, ETFInfo] = {}
        self._universe: List[str] = []  # Current filtered universe
        
        # Default filter
        self._filter = UniverseFilter()
        
        # Category keywords for auto-classification
        self._category_keywords = self._build_category_keywords()
        
        # Last refresh time
        self._last_refresh: Optional[datetime] = None
        
        logger.info("UniverseManager initialized")
    
    def _build_category_keywords(self) -> Dict[ETFCategory, List[str]]:
        """Build keyword mapping for ETF categorization."""
        return {
            # Domestic Index
            ETFCategory.INDEX_KOSPI: ["코스피", "KOSPI", "200", "50", "대형"],
            ETFCategory.INDEX_KOSDAQ: ["코스닥", "KOSDAQ", "150"],
            ETFCategory.INDEX_KRX: ["KRX", "300"],
            
            # Sectors
            ETFCategory.SECTOR_TECH: ["IT", "반도체", "테크", "소프트", "인터넷", "게임", "2차전지", "배터리"],
            ETFCategory.SECTOR_FINANCE: ["금융", "은행", "보험", "증권"],
            ETFCategory.SECTOR_HEALTHCARE: ["헬스케어", "바이오", "제약", "의료"],
            ETFCategory.SECTOR_ENERGY: ["에너지", "정유", "가스"],
            ETFCategory.SECTOR_MATERIALS: ["소재", "화학", "철강", "비철"],
            ETFCategory.SECTOR_INDUSTRIALS: ["산업재", "기계", "조선", "건설", "운송"],
            ETFCategory.SECTOR_CONSUMER: ["소비재", "필수소비", "경기소비", "음식료", "유통"],
            ETFCategory.SECTOR_UTILITIES: ["유틸리티", "전력", "가스"],
            
            # Themes
            ETFCategory.THEME_ESG: ["ESG", "친환경", "그린", "탄소"],
            ETFCategory.THEME_DIVIDEND: ["배당", "고배당", "dividend"],
            ETFCategory.THEME_VALUE: ["가치", "밸류", "value"],
            ETFCategory.THEME_GROWTH: ["성장", "그로스", "growth"],
            ETFCategory.THEME_MOMENTUM: ["모멘텀", "momentum"],
            ETFCategory.THEME_QUALITY: ["퀄리티", "quality", "우량"],
            ETFCategory.THEME_LOW_VOL: ["저변동", "로우볼", "low vol"],
            ETFCategory.THEME_BATTERY: ["2차전지", "배터리", "전기차", "EV"],
            ETFCategory.THEME_BIO: ["바이오", "헬스케어", "제약"],
            ETFCategory.THEME_AI: ["AI", "인공지능", "로봇", "자율주행"],
            
            # International
            ETFCategory.INTL_US: ["미국", "S&P", "나스닥", "NASDAQ", "다우", "US", "미국채"],
            ETFCategory.INTL_CHINA: ["중국", "차이나", "China", "CSI", "항셍"],
            ETFCategory.INTL_JAPAN: ["일본", "Japan", "니케이", "TOPIX"],
            ETFCategory.INTL_EUROPE: ["유럽", "유로", "Europe", "독일", "DAX"],
            ETFCategory.INTL_EMERGING: ["신흥", "이머징", "emerging", "EM"],
            ETFCategory.INTL_GLOBAL: ["글로벌", "global", "선진국", "MSCI"],
            
            # Fixed Income
            ETFCategory.BOND_GOVT: ["국채", "국고채", "통안채"],
            ETFCategory.BOND_CORP: ["회사채", "크레딧"],
            ETFCategory.BOND_CREDIT: ["하이일드", "신용"],
            
            # Commodity
            ETFCategory.COMMODITY_GOLD: ["금", "골드", "gold"],
            ETFCategory.COMMODITY_OIL: ["원유", "WTI", "oil", "석유"],
            ETFCategory.COMMODITY_METALS: ["은", "실버", "구리", "팔라듐", "플래티넘"],
            ETFCategory.COMMODITY_AGRI: ["농산물", "곡물", "콩", "옥수수"],
            
            # Leverage/Inverse
            ETFCategory.LEVERAGE_2X: ["레버리지", "2X", "2배"],
            ETFCategory.INVERSE_1X: ["인버스", "inverse"],
            ETFCategory.INVERSE_2X: ["인버스2X", "곱버스"],
            
            # Other
            ETFCategory.REITS: ["리츠", "REITs", "부동산"],
            ETFCategory.CURRENCY: ["달러", "엔화", "유로화", "환"],
        }
    
    def _classify_etf(self, name: str) -> ETFCategory:
        """Classify ETF based on name keywords."""
        name_upper = name.upper()
        name_lower = name.lower()

        # Check leverage/inverse first (highest priority)
        if any(kw in name for kw in ["레버리지", "2X", "2배"]):
            return ETFCategory.LEVERAGE_2X
        if any(kw in name for kw in ["인버스2X", "곱버스"]):
            return ETFCategory.INVERSE_2X
        if any(kw in name for kw in ["인버스", "inverse"]):
            return ETFCategory.INVERSE_1X

        # Check international categories first (more specific keywords)
        intl_categories = [
            ETFCategory.INTL_US, ETFCategory.INTL_CHINA, ETFCategory.INTL_JAPAN,
            ETFCategory.INTL_EUROPE, ETFCategory.INTL_EMERGING, ETFCategory.INTL_GLOBAL
        ]
        for category in intl_categories:
            keywords = self._category_keywords.get(category, [])
            for keyword in keywords:
                if keyword.lower() in name_lower or keyword.upper() in name_upper:
                    return category

        # Check other categories
        skip_categories = {
            ETFCategory.LEVERAGE_2X, ETFCategory.INVERSE_1X, ETFCategory.INVERSE_2X,
            *intl_categories
        }
        for category, keywords in self._category_keywords.items():
            if category in skip_categories:
                continue  # Already checked

            for keyword in keywords:
                if keyword.lower() in name_lower or keyword.upper() in name_upper:
                    return category

        return ETFCategory.OTHER
    
    def _is_leverage_or_inverse(self, name: str) -> tuple[bool, bool, float]:
        """Detect if ETF is leveraged or inverse."""
        is_leverage = any(kw in name for kw in ["레버리지", "2X", "2배"])
        is_inverse = any(kw in name for kw in ["인버스", "inverse", "곱버스"])
        
        leverage_ratio = 1.0
        if "2X" in name or "2배" in name or "곱버스" in name:
            leverage_ratio = 2.0
        if is_inverse:
            leverage_ratio = -leverage_ratio
        
        return is_leverage, is_inverse, abs(leverage_ratio)
    
    async def discover_etfs(self) -> List[ETFInfo]:
        """
        Discover all ETFs available on KRX.
        
        Uses KIS API to fetch ETF list and metadata.
        """
        if not self._broker:
            logger.warning("No broker available for ETF discovery")
            return []
        
        logger.info("Starting ETF discovery...")
        discovered = []
        
        try:
            # KIS API doesn't have a direct ETF list endpoint
            # We need to use sector/theme queries or a master file
            # For now, we'll use known major ETF symbols and expand
            
            # Major Korean ETF issuers and their prefixes
            # Samsung: KODEX (varies)
            # Mirae: TIGER (varies)
            # KB: KBSTAR (varies)
            # NH: HANARO (varies)
            # Shinhan: SOL (varies)
            
            # Fetch from a predefined seed list first
            seed_etfs = await self._get_seed_etf_list()
            
            for symbol in seed_etfs:
                try:
                    etf_info = await self._fetch_etf_info(symbol)
                    if etf_info:
                        discovered.append(etf_info)
                        self._etf_cache[symbol] = etf_info
                except Exception as e:
                    logger.debug(f"Failed to fetch ETF {symbol}: {e}")
                
                # Rate limiting
                await asyncio.sleep(0.15)
            
            logger.info(f"Discovered {len(discovered)} ETFs")
            
            # Store in database
            await self._store_etfs(discovered)
            
            self._last_refresh = datetime.now()
            
            return discovered
            
        except Exception as e:
            logger.error(f"ETF discovery failed: {e}")
            return []
    
    async def _get_seed_etf_list(self) -> List[str]:
        """
        Get seed list of ETF symbols to discover.
        
        This is a curated list of major ETFs. The system will
        expand this through discovery over time.
        """
        # Major Korean ETFs by category
        seed_etfs = [
            # KOSPI Index
            "069500",  # KODEX 200
            "102110",  # TIGER 200
            "148020",  # KBSTAR 200
            "069660",  # KOSEF 200
            "278540",  # KODEX KOSPI
            "292150",  # TIGER TOP10
            
            # KOSDAQ
            "229200",  # KODEX KOSDAQ 150
            "232080",  # TIGER KOSDAQ 150
            "270810",  # KBSTAR KOSDAQ 150
            
            # Sector - Tech/Semiconductor
            "091160",  # KODEX 반도체
            "091180",  # KODEX 자동차
            "139260",  # TIGER 200 IT
            "157490",  # TIGER 소프트웨어
            "091230",  # TIGER 반도체
            "363580",  # KBSTAR 비메모리반도체
            
            # Sector - Finance
            "091170",  # KODEX 은행
            "139270",  # TIGER 200 금융
            
            # Sector - Healthcare/Bio
            "143860",  # KODEX 헬스케어
            "227540",  # TIGER 200 헬스케어
            "244580",  # KODEX 바이오
            "203780",  # TIGER 200 바이오
            
            # 2차전지/EV
            "305720",  # KODEX 2차전지산업
            "305540",  # TIGER 2차전지테마
            "371160",  # TIGER 2차전지TOP10
            "394660",  # KODEX K-배터리
            
            # Dividend
            "211560",  # TIGER 배당성장
            "279530",  # KODEX 고배당
            "161510",  # ARIRANG 고배당
            "104530",  # KOSEF 고배당
            
            # US Markets
            "360750",  # TIGER 미국S&P500
            "379800",  # KODEX 미국S&P500TR
            "133690",  # TIGER 미국나스닥100
            "367380",  # KBSTAR 미국나스닥100
            "381170",  # TIGER 미국테크TOP10
            "379810",  # KODEX 미국나스닥100TR
            "453850",  # TIGER 미국필라델피아반도체
            
            # China
            "192090",  # TIGER 차이나CSI300
            "217780",  # TIGER 차이나항셍테크
            
            # Global/Other International
            "195930",  # TIGER 유로스탁스50
            "238720",  # KINDEX 일본Nikkei225
            "225060",  # KODEX 선진국MSCI World
            
            # Gold/Commodities
            "132030",  # KODEX 골드선물(H)
            "319640",  # TIGER 골드선물(H)
            "130680",  # TIGER 원유선물Enhanced(H)
            "261220",  # KODEX WTI원유선물(H)
            
            # Bonds
            "114820",  # TIGER 국채3년
            "148070",  # KOSEF 국고채10년
            "152380",  # KODEX 국채선물10년
            "182490",  # TIGER 단기채권
            "136340",  # KBSTAR 중기우량회사채
            
            # REITs
            "329200",  # TIGER 부동산인프라고배당
            
            # Leverage (excluded by default filter)
            "122630",  # KODEX 레버리지
            "233740",  # KODEX KOSDAQ 150 레버리지
            
            # Inverse (excluded by default filter)  
            "114800",  # KODEX 인버스
            "251340",  # KODEX KOSDAQ 150 선물인버스
            "145670",  # KINDEX 인버스
            
            # Low Volatility
            "200250",  # KOSEF 200 저변동성
            
            # ESG
            "289040",  # KODEX MSCI Korea ESG
            "423230",  # TIGER 글로벌클린에너지
            
            # AI/Tech Theme
            "446770",  # KODEX AI반도체핵심장비
            "418660",  # TIGER AI반도체
        ]
        
        return seed_etfs
    
    async def _fetch_etf_info(self, symbol: str) -> Optional[ETFInfo]:
        """Fetch detailed information for a single ETF."""
        try:
            # Get current price data
            price_data = self._broker.get_price(symbol)
            
            if not price_data or price_data.get("current_price", 0) == 0:
                return None
            
            name = price_data.get("name", "")
            
            # Classify the ETF
            category = self._classify_etf(name)
            is_leverage, is_inverse, leverage_ratio = self._is_leverage_or_inverse(name)
            
            # Get historical data for volume calculation
            try:
                history = self._broker.get_daily_ohlcv(
                    symbol,
                    start_date=(datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
                    end_date=datetime.now().strftime("%Y%m%d")
                )
                
                if history and len(history) > 5:
                    avg_volume = sum(bar["volume"] for bar in history) / len(history)
                    avg_value = sum(bar["close"] * bar["volume"] for bar in history) / len(history)
                    
                    # Calculate 30-day volatility
                    closes = [bar["close"] for bar in history]
                    if len(closes) > 1:
                        returns = [(closes[i] - closes[i-1]) / closes[i-1] 
                                   for i in range(1, len(closes))]
                        import statistics
                        volatility = statistics.stdev(returns) * (252 ** 0.5) if len(returns) > 1 else 0
                    else:
                        volatility = 0
                else:
                    avg_volume = float(price_data.get("volume", 0))
                    avg_value = avg_volume * price_data.get("current_price", 0)
                    volatility = 0
                    
            except Exception as e:
                logger.debug(f"Could not fetch history for {symbol}: {e}")
                avg_volume = float(price_data.get("volume", 0))
                avg_value = avg_volume * price_data.get("current_price", 0)
                volatility = 0
            
            # Detect issuer from name
            issuer = self._detect_issuer(name)
            
            # Estimate AUM from market cap (for ETFs, market cap ≈ AUM)
            aum = price_data.get("market_cap", 0) * 1_000_000  # Convert to KRW
            
            return ETFInfo(
                symbol=symbol,
                name=name,
                category=category,
                issuer=issuer,
                aum=aum,
                avg_daily_volume=avg_volume,
                avg_daily_value=avg_value,
                current_price=price_data.get("current_price", 0),
                volatility_30d=volatility,
                is_leverage=is_leverage,
                is_inverse=is_inverse,
                leverage_ratio=leverage_ratio,
                is_tradeable=True,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.debug(f"Error fetching ETF info for {symbol}: {e}")
            return None
    
    def _detect_issuer(self, name: str) -> str:
        """Detect ETF issuer from name."""
        issuers = {
            "KODEX": "Samsung",
            "TIGER": "Mirae",
            "KBSTAR": "KB",
            "HANARO": "NH",
            "SOL": "Shinhan",
            "ARIRANG": "Hanwha",
            "KOSEF": "Samsung",
            "KINDEX": "Korea Investment",
            "FOCUS": "Focus",
            "TIMEFOLIO": "Timefolio",
            "ACE": "ACE",
        }
        
        for prefix, issuer in issuers.items():
            if prefix in name.upper():
                return issuer
        
        return "Unknown"
    
    async def _store_etfs(self, etfs: List[ETFInfo]) -> None:
        """Store ETF information in database."""
        session = self._db.get_session()
        
        try:
            for etf in etfs:
                # Check if exists
                existing = session.query(Instrument).filter(
                    Instrument.symbol == etf.symbol
                ).first()
                
                if existing:
                    # Update
                    existing.name = etf.name
                    existing.instrument_type = "etf"
                    existing.avg_daily_volume = etf.avg_daily_volume
                    existing.volatility_30d = etf.volatility_30d
                    existing.is_tradeable = etf.is_tradeable
                    existing.metadata_json = etf.to_dict()
                    existing.last_updated = datetime.now()
                else:
                    # Insert
                    instrument = Instrument(
                        symbol=etf.symbol,
                        name=etf.name,
                        instrument_type="etf",
                        market="KRX",
                        avg_daily_volume=etf.avg_daily_volume,
                        volatility_30d=etf.volatility_30d,
                        is_tradeable=etf.is_tradeable,
                        metadata_json=etf.to_dict()
                    )
                    session.add(instrument)
            
            session.commit()
            logger.info(f"Stored {len(etfs)} ETFs in database")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store ETFs: {e}")
        finally:
            session.close()
    
    def set_filter(self, filter_config: UniverseFilter) -> None:
        """Set the universe filter."""
        self._filter = filter_config
        logger.info("Universe filter updated")
    
    def apply_filter(self) -> List[str]:
        """
        Apply filter to cached ETFs and return filtered universe.
        """
        filtered = []
        
        for symbol, etf in self._etf_cache.items():
            if self._filter.passes(etf):
                filtered.append(symbol)
        
        self._universe = filtered
        logger.info(f"Filtered universe: {len(filtered)} ETFs from {len(self._etf_cache)}")
        
        return filtered
    
    def get_universe(self) -> List[str]:
        """Get current filtered universe."""
        if not self._universe and self._etf_cache:
            self.apply_filter()
        return self._universe
    
    def get_etf_info(self, symbol: str) -> Optional[ETFInfo]:
        """Get ETF info by symbol."""
        return self._etf_cache.get(symbol)
    
    def get_etfs_by_category(self, category: ETFCategory) -> List[ETFInfo]:
        """Get all ETFs in a category."""
        return [
            etf for etf in self._etf_cache.values()
            if etf.category == category
        ]
    
    def get_universe_by_category(self) -> Dict[ETFCategory, List[str]]:
        """Get universe grouped by category."""
        by_category: Dict[ETFCategory, List[str]] = {}
        
        for symbol in self._universe:
            etf = self._etf_cache.get(symbol)
            if etf:
                if etf.category not in by_category:
                    by_category[etf.category] = []
                by_category[etf.category].append(symbol)
        
        return by_category
    
    def get_stats(self) -> Dict[str, Any]:
        """Get universe statistics."""
        if not self._etf_cache:
            return {"status": "empty", "total": 0}
        
        by_category = {}
        for etf in self._etf_cache.values():
            cat = etf.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
        
        return {
            "status": "ready",
            "total_discovered": len(self._etf_cache),
            "filtered_universe": len(self._universe),
            "by_category": by_category,
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "filter": {
                "min_aum": self._filter.min_aum,
                "min_avg_daily_volume": self._filter.min_avg_daily_volume,
                "include_leverage": self._filter.include_leverage,
                "include_inverse": self._filter.include_inverse,
            }
        }
    
    async def refresh(self) -> None:
        """Refresh the entire universe."""
        await self.discover_etfs()
        self.apply_filter()
        
        # Emit event
        get_event_bus().publish(Event(
            event_type=EventType.REGIME_CHANGE_DETECTED,  # Reusing event type
            source="universe_manager",
            payload={
                "action": "universe_refreshed",
                "stats": self.get_stats()
            }
        ))


# Singleton instance
_universe_manager: Optional[UniverseManager] = None


def get_universe_manager(broker=None) -> UniverseManager:
    """Get the singleton UniverseManager instance."""
    global _universe_manager
    if _universe_manager is None:
        _universe_manager = UniverseManager(broker)
    elif broker and not _universe_manager._broker:
        _universe_manager._broker = broker
    return _universe_manager
