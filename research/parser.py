"""
IDEA EXTRACTOR - Parse academic papers and extract trading ideas

Uses NLP and pattern matching to identify:
- Trading strategies mentioned
- Key indicators and signals
- Entry/exit conditions
- Risk management approaches
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import re
from loguru import logger

from research.fetcher import AcademicPaper


class IdeaType(str, Enum):
    """Types of trading ideas that can be extracted."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    FACTOR = "factor"
    SENTIMENT = "sentiment"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MACHINE_LEARNING = "machine_learning"
    VOLATILITY = "volatility"
    OTHER = "other"


class ConfidenceLevel(str, Enum):
    """Confidence in extracted idea."""
    HIGH = "high"      # Clear, explicit strategy description
    MEDIUM = "medium"  # Implied strategy or partial description
    LOW = "low"        # Weak signals, needs interpretation


@dataclass
class ExtractedIdea:
    """A trading idea extracted from research."""
    idea_id: str
    source_paper_id: str
    idea_type: IdeaType
    title: str
    description: str
    confidence: ConfidenceLevel

    # Strategy components
    entry_conditions: List[str] = field(default_factory=list)
    exit_conditions: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    timeframe: Optional[str] = None
    asset_class: Optional[str] = None

    # Parameters mentioned
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Claims from paper
    claimed_sharpe: Optional[float] = None
    claimed_returns: Optional[float] = None
    backtest_period: Optional[str] = None

    # Metadata
    extracted_at: datetime = field(default_factory=datetime.utcnow)
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idea_id": self.idea_id,
            "source_paper_id": self.source_paper_id,
            "idea_type": self.idea_type.value,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence.value,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
            "indicators": self.indicators,
            "timeframe": self.timeframe,
            "parameters": self.parameters,
            "claimed_sharpe": self.claimed_sharpe,
            "claimed_returns": self.claimed_returns,
            "keywords": self.keywords,
        }


class IdeaExtractor:
    """
    Extracts trading ideas from academic papers.

    Uses pattern matching and NLP to identify:
    - Strategy descriptions
    - Trading rules
    - Performance claims
    - Key indicators
    """

    # Patterns for different strategy types
    STRATEGY_PATTERNS = {
        IdeaType.MOMENTUM: [
            r"momentum\s+strateg",
            r"trend\s+follow",
            r"price\s+momentum",
            r"cross[- ]?sectional\s+momentum",
            r"time[- ]?series\s+momentum",
            r"winner[s]?\s+minus\s+loser[s]?",
            r"WML",
            r"52[- ]?week\s+high",
        ],
        IdeaType.MEAN_REVERSION: [
            r"mean\s+reversion",
            r"reversion\s+to\s+(the\s+)?mean",
            r"contrarian",
            r"overreaction",
            r"reversal\s+strateg",
            r"short[- ]?term\s+reversal",
            r"pairs\s+trading",
        ],
        IdeaType.BREAKOUT: [
            r"breakout\s+strateg",
            r"support\s+and\s+resistance",
            r"channel\s+breakout",
            r"range\s+breakout",
            r"volatility\s+breakout",
        ],
        IdeaType.FACTOR: [
            r"factor\s+invest",
            r"factor\s+model",
            r"value\s+factor",
            r"size\s+factor",
            r"quality\s+factor",
            r"low\s+volatility\s+factor",
            r"Fama[- ]?French",
            r"multi[- ]?factor",
        ],
        IdeaType.SENTIMENT: [
            r"sentiment\s+analy",
            r"news\s+sentiment",
            r"social\s+media",
            r"twitter",
            r"text\s+mining",
            r"natural\s+language",
            r"investor\s+sentiment",
        ],
        IdeaType.STATISTICAL_ARBITRAGE: [
            r"statistical\s+arbitrage",
            r"stat[- ]?arb",
            r"cointegrat",
            r"pairs\s+trading",
            r"spread\s+trading",
            r"relative\s+value",
        ],
        IdeaType.MACHINE_LEARNING: [
            r"machine\s+learning",
            r"deep\s+learning",
            r"neural\s+network",
            r"random\s+forest",
            r"gradient\s+boost",
            r"LSTM",
            r"reinforcement\s+learning",
            r"supervised\s+learning",
        ],
        IdeaType.VOLATILITY: [
            r"volatility\s+trad",
            r"VIX\s+strateg",
            r"variance\s+swap",
            r"volatility\s+premium",
            r"straddle",
            r"strangle",
        ],
    }

    # Patterns for extracting trading rules
    ENTRY_PATTERNS = [
        r"buy\s+(?:when|if|signal)",
        r"enter\s+(?:long|short)\s+(?:when|if)",
        r"go\s+long\s+(?:when|if)",
        r"open\s+(?:a\s+)?position\s+(?:when|if)",
        r"entry\s+(?:signal|rule|condition)",
        r"trigger[s]?\s+a\s+buy",
    ]

    EXIT_PATTERNS = [
        r"sell\s+(?:when|if|signal)",
        r"exit\s+(?:when|if)",
        r"close\s+(?:the\s+)?position",
        r"stop[- ]?loss",
        r"take\s+profit",
        r"exit\s+(?:signal|rule|condition)",
    ]

    # Common indicators
    INDICATOR_PATTERNS = [
        (r"RSI|relative\s+strength\s+index", "RSI"),
        (r"MACD|moving\s+average\s+convergence", "MACD"),
        (r"SMA|simple\s+moving\s+average", "SMA"),
        (r"EMA|exponential\s+moving\s+average", "EMA"),
        (r"Bollinger\s+[Bb]ands?", "Bollinger Bands"),
        (r"ATR|average\s+true\s+range", "ATR"),
        (r"stochastic\s+oscillator", "Stochastic"),
        (r"ADX|average\s+directional", "ADX"),
        (r"volume[- ]?weighted", "VWAP"),
        (r"momentum\s+indicator", "Momentum"),
        (r"z[- ]?score", "Z-Score"),
    ]

    _instance: Optional['IdeaExtractor'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._extracted_ideas: Dict[str, ExtractedIdea] = {}
        self._idea_counter = 0

        self._initialized = True
        logger.info("IdeaExtractor initialized")

    def extract_ideas(self, paper: AcademicPaper) -> List[ExtractedIdea]:
        """
        Extract trading ideas from a paper.

        Returns list of ideas found in the paper.
        """
        ideas = []
        text = f"{paper.title}\n{paper.abstract}"

        # Identify strategy types
        detected_types = self._detect_strategy_types(text)

        for idea_type, confidence in detected_types:
            idea = self._create_idea(paper, text, idea_type, confidence)
            if idea:
                ideas.append(idea)
                self._extracted_ideas[idea.idea_id] = idea

        return ideas

    def _detect_strategy_types(self, text: str) -> List[Tuple[IdeaType, ConfidenceLevel]]:
        """Detect strategy types mentioned in text."""
        detected = []
        text_lower = text.lower()

        for idea_type, patterns in self.STRATEGY_PATTERNS.items():
            match_count = 0
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    match_count += 1

            if match_count >= 3:
                detected.append((idea_type, ConfidenceLevel.HIGH))
            elif match_count >= 2:
                detected.append((idea_type, ConfidenceLevel.MEDIUM))
            elif match_count >= 1:
                detected.append((idea_type, ConfidenceLevel.LOW))

        # If no specific type detected, mark as OTHER
        if not detected:
            detected.append((IdeaType.OTHER, ConfidenceLevel.LOW))

        return detected

    def _create_idea(
        self,
        paper: AcademicPaper,
        text: str,
        idea_type: IdeaType,
        confidence: ConfidenceLevel
    ) -> Optional[ExtractedIdea]:
        """Create an extracted idea from paper content."""
        import uuid

        self._idea_counter += 1

        # Extract indicators
        indicators = self._extract_indicators(text)

        # Extract entry conditions
        entry_conditions = self._extract_conditions(text, self.ENTRY_PATTERNS)

        # Extract exit conditions
        exit_conditions = self._extract_conditions(text, self.EXIT_PATTERNS)

        # Extract parameters
        parameters = self._extract_parameters(text)

        # Extract performance claims
        sharpe = self._extract_sharpe(text)
        returns = self._extract_returns(text)

        # Generate description
        description = self._generate_description(idea_type, indicators, paper.title)

        idea = ExtractedIdea(
            idea_id=f"idea_{self._idea_counter}_{uuid.uuid4().hex[:8]}",
            source_paper_id=paper.paper_id,
            idea_type=idea_type,
            title=f"{idea_type.value.replace('_', ' ').title()} Strategy",
            description=description,
            confidence=confidence,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            indicators=indicators,
            parameters=parameters,
            claimed_sharpe=sharpe,
            claimed_returns=returns,
            keywords=paper.trading_keywords,
        )

        return idea

    def _extract_indicators(self, text: str) -> List[str]:
        """Extract technical indicators mentioned in text."""
        indicators = []

        for pattern, name in self.INDICATOR_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if name not in indicators:
                    indicators.append(name)

        return indicators

    def _extract_conditions(self, text: str, patterns: List[str]) -> List[str]:
        """Extract trading conditions from text."""
        conditions = []

        for pattern in patterns:
            matches = re.findall(f"({pattern}[^.]*\\.)", text, re.IGNORECASE)
            for match in matches[:3]:  # Limit to 3 per pattern
                clean = match.strip()[:200]  # Limit length
                if clean and clean not in conditions:
                    conditions.append(clean)

        return conditions[:5]  # Return max 5 conditions

    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract numerical parameters from text."""
        parameters = {}

        # Look for lookback periods
        lookback = re.search(r"(\d+)[- ]?(?:day|week|month)\s+(?:lookback|window|period)", text, re.IGNORECASE)
        if lookback:
            parameters["lookback_period"] = int(lookback.group(1))

        # Look for thresholds
        threshold = re.search(r"threshold\s+(?:of\s+)?(\d+\.?\d*)%?", text, re.IGNORECASE)
        if threshold:
            parameters["threshold"] = float(threshold.group(1))

        # Look for moving average periods
        ma_period = re.search(r"(\d+)[- ]?(?:day|period)\s+(?:moving\s+average|MA|SMA|EMA)", text, re.IGNORECASE)
        if ma_period:
            parameters["ma_period"] = int(ma_period.group(1))

        return parameters

    def _extract_sharpe(self, text: str) -> Optional[float]:
        """Extract Sharpe ratio claims from text."""
        match = re.search(r"[Ss]harpe\s+(?:ratio\s+)?(?:of\s+)?(\d+\.?\d*)", text)
        if match:
            return float(match.group(1))
        return None

    def _extract_returns(self, text: str) -> Optional[float]:
        """Extract return claims from text."""
        match = re.search(r"(?:annual|yearly)\s+return[s]?\s+(?:of\s+)?(\d+\.?\d*)%", text, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100
        return None

    def _generate_description(
        self,
        idea_type: IdeaType,
        indicators: List[str],
        title: str
    ) -> str:
        """Generate a description for the idea."""
        type_desc = {
            IdeaType.MOMENTUM: "Exploits price momentum by buying recent winners",
            IdeaType.MEAN_REVERSION: "Profits from price reversion to historical averages",
            IdeaType.TREND_FOLLOWING: "Follows established price trends",
            IdeaType.BREAKOUT: "Capitalizes on price breaking through key levels",
            IdeaType.FACTOR: "Systematically captures factor risk premiums",
            IdeaType.SENTIMENT: "Trades based on market sentiment signals",
            IdeaType.STATISTICAL_ARBITRAGE: "Exploits statistical relationships between assets",
            IdeaType.MACHINE_LEARNING: "Uses ML models for price prediction",
            IdeaType.VOLATILITY: "Trades volatility patterns and premiums",
            IdeaType.OTHER: "Alternative trading approach",
        }

        desc = type_desc.get(idea_type, "Trading strategy")

        if indicators:
            desc += f". Key indicators: {', '.join(indicators[:3])}"

        desc += f". Based on: {title[:100]}"

        return desc

    def get_idea(self, idea_id: str) -> Optional[ExtractedIdea]:
        """Get a specific idea by ID."""
        return self._extracted_ideas.get(idea_id)

    def get_all_ideas(self) -> List[ExtractedIdea]:
        """Get all extracted ideas."""
        return list(self._extracted_ideas.values())

    def get_ideas_by_type(self, idea_type: IdeaType) -> List[ExtractedIdea]:
        """Get ideas of a specific type."""
        return [i for i in self._extracted_ideas.values() if i.idea_type == idea_type]

    def get_high_confidence_ideas(self) -> List[ExtractedIdea]:
        """Get ideas with high confidence."""
        return [i for i in self._extracted_ideas.values()
                if i.confidence == ConfidenceLevel.HIGH]

    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics."""
        ideas = list(self._extracted_ideas.values())
        return {
            "total_ideas": len(ideas),
            "by_type": {
                t.value: len([i for i in ideas if i.idea_type == t])
                for t in IdeaType
            },
            "by_confidence": {
                c.value: len([i for i in ideas if i.confidence == c])
                for c in ConfidenceLevel
            },
            "avg_indicators_per_idea": sum(len(i.indicators) for i in ideas) / len(ideas)
                                       if ideas else 0,
        }


# Singleton accessor
_extractor_instance: Optional[IdeaExtractor] = None

def get_extractor() -> IdeaExtractor:
    """Get the singleton IdeaExtractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = IdeaExtractor()
    return _extractor_instance
