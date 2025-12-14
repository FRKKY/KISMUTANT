"""
IDEA EXTRACTOR - Parse academic papers and extract trading ideas

Uses advanced NLP and pattern matching to identify:
- Trading strategies mentioned
- Key indicators and signals
- Entry/exit conditions
- Risk management approaches
- Performance claims and backtesting results
- Numerical parameters and thresholds
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import defaultdict
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


class TextAnalyzer:
    """
    Advanced text analysis utilities for trading idea extraction.

    Provides:
    - Sentence splitting and tokenization
    - Entity extraction (numbers, percentages, time periods)
    - Context-aware pattern matching
    - Semantic similarity scoring
    """

    # Sentence boundary patterns
    SENTENCE_SPLITTERS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    # Number extraction patterns
    NUMBER_PATTERNS = {
        "percentage": re.compile(r'(\d+\.?\d*)\s*%'),
        "decimal": re.compile(r'(\d+\.\d+)'),
        "integer": re.compile(r'\b(\d+)\b'),
        "ratio": re.compile(r'(\d+\.?\d*)\s*:\s*(\d+\.?\d*)'),
    }

    # Time period patterns
    TIME_PATTERNS = {
        "days": re.compile(r'(\d+)\s*(?:day|trading day)s?', re.IGNORECASE),
        "weeks": re.compile(r'(\d+)\s*weeks?', re.IGNORECASE),
        "months": re.compile(r'(\d+)\s*months?', re.IGNORECASE),
        "years": re.compile(r'(\d+)\s*years?', re.IGNORECASE),
    }

    # Trading-specific entity patterns
    TRADING_ENTITIES = {
        "asset_class": [
            (re.compile(r'\b(equit(?:y|ies)|stocks?)\b', re.IGNORECASE), "equity"),
            (re.compile(r'\b(ETFs?|exchange[- ]traded funds?)\b', re.IGNORECASE), "ETF"),
            (re.compile(r'\b(forex|currencies|FX)\b', re.IGNORECASE), "forex"),
            (re.compile(r'\b(futures?|commodit(?:y|ies))\b', re.IGNORECASE), "futures"),
            (re.compile(r'\b(options?|derivatives?)\b', re.IGNORECASE), "options"),
            (re.compile(r'\b(bonds?|fixed[- ]income)\b', re.IGNORECASE), "fixed_income"),
            (re.compile(r'\b(crypt(?:o|ocurrenc(?:y|ies))|bitcoin|ethereum)\b', re.IGNORECASE), "crypto"),
        ],
        "market": [
            (re.compile(r'\b(US\s+market|S&P\s*500|NYSE|NASDAQ)\b', re.IGNORECASE), "US"),
            (re.compile(r'\b(Korean\s+market|KOSPI|KOSDAQ|KRX)\b', re.IGNORECASE), "Korea"),
            (re.compile(r'\b(emerging\s+markets?|EM)\b', re.IGNORECASE), "emerging"),
            (re.compile(r'\b(developed\s+markets?|DM)\b', re.IGNORECASE), "developed"),
        ],
        "frequency": [
            (re.compile(r'\b(high[- ]frequency|HFT|intraday)\b', re.IGNORECASE), "high_frequency"),
            (re.compile(r'\b(daily|day[- ]trading)\b', re.IGNORECASE), "daily"),
            (re.compile(r'\b(weekly)\b', re.IGNORECASE), "weekly"),
            (re.compile(r'\b(monthly)\b', re.IGNORECASE), "monthly"),
        ],
    }

    # Causal relationship patterns
    CAUSAL_PATTERNS = [
        re.compile(r'(when|if|once)\s+(.{10,100}?)\s*,?\s+(then\s+)?(.{10,100}?)[.]', re.IGNORECASE),
        re.compile(r'(.{10,50}?)\s+(leads?\s+to|results?\s+in|causes?)\s+(.{10,100}?)[.]', re.IGNORECASE),
        re.compile(r'(.{10,50}?)\s+(signals?|indicates?|suggests?)\s+(.{10,100}?)[.]', re.IGNORECASE),
    ]

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """Split text into sentences."""
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Split on sentence boundaries
        sentences = cls.SENTENCE_SPLITTERS.split(text)
        return [s.strip() for s in sentences if s.strip()]

    @classmethod
    def extract_numbers(cls, text: str) -> Dict[str, List[float]]:
        """Extract all numbers from text by type."""
        results = {}
        for num_type, pattern in cls.NUMBER_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                if num_type == "ratio":
                    results[num_type] = [(float(m[0]), float(m[1])) for m in matches]
                else:
                    results[num_type] = [float(m) for m in matches]
        return results

    @classmethod
    def extract_time_periods(cls, text: str) -> Dict[str, List[int]]:
        """Extract time periods from text."""
        results = {}
        for period_type, pattern in cls.TIME_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                results[period_type] = [int(m) for m in matches]
        return results

    @classmethod
    def extract_entities(cls, text: str) -> Dict[str, Set[str]]:
        """Extract trading-specific entities from text."""
        results = defaultdict(set)
        for entity_type, patterns in cls.TRADING_ENTITIES.items():
            for pattern, label in patterns:
                if pattern.search(text):
                    results[entity_type].add(label)
        return dict(results)

    @classmethod
    def extract_context(cls, text: str, keyword: str, window: int = 100) -> List[str]:
        """Extract context around a keyword."""
        contexts = []
        keyword_pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for match in keyword_pattern.finditer(text):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            context = text[start:end].strip()
            if context:
                contexts.append(context)
        return contexts

    @classmethod
    def extract_causal_relationships(cls, text: str) -> List[Tuple[str, str]]:
        """Extract causal relationships (condition -> outcome)."""
        relationships = []
        for pattern in cls.CAUSAL_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if len(match) >= 2:
                    condition = match[1] if len(match) > 2 else match[0]
                    outcome = match[-1]
                    relationships.append((condition.strip(), outcome.strip()))
        return relationships

    @classmethod
    def score_relevance(cls, text: str, keywords: List[str]) -> float:
        """
        Score text relevance based on keyword density and distribution.

        Returns score from 0 to 1.
        """
        if not keywords or not text:
            return 0.0

        text_lower = text.lower()
        total_matches = 0
        unique_matches = 0

        for keyword in keywords:
            count = len(re.findall(re.escape(keyword.lower()), text_lower))
            if count > 0:
                unique_matches += 1
                total_matches += count

        # Score based on unique matches and total density
        unique_ratio = unique_matches / len(keywords)
        density = min(total_matches / (len(text.split()) + 1), 1.0)

        return (unique_ratio * 0.7 + density * 0.3)


class IdeaExtractor:
    """
    Extracts trading ideas from academic papers.

    Uses advanced NLP and pattern matching to identify:
    - Strategy descriptions
    - Trading rules
    - Performance claims
    - Key indicators
    - Numerical parameters
    - Causal relationships
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
        """Create an extracted idea from paper content using advanced NLP."""
        import uuid

        self._idea_counter += 1

        # Use TextAnalyzer for enhanced extraction
        sentences = TextAnalyzer.split_sentences(text)

        # Extract indicators
        indicators = self._extract_indicators(text)

        # Extract entry conditions using sentence-level analysis
        entry_conditions = self._extract_conditions_enhanced(text, sentences, self.ENTRY_PATTERNS)

        # Extract exit conditions
        exit_conditions = self._extract_conditions_enhanced(text, sentences, self.EXIT_PATTERNS)

        # Extract parameters with enhanced number extraction
        parameters = self._extract_parameters_enhanced(text)

        # Extract entities (asset class, market, frequency)
        entities = TextAnalyzer.extract_entities(text)

        # Extract causal relationships
        relationships = TextAnalyzer.extract_causal_relationships(text)

        # Add relationships to entry/exit conditions
        for condition, outcome in relationships[:3]:
            if any(kw in condition.lower() for kw in ["buy", "long", "enter", "signal"]):
                if condition not in entry_conditions:
                    entry_conditions.append(f"If {condition}, then {outcome}")
            elif any(kw in condition.lower() for kw in ["sell", "exit", "close", "stop"]):
                if condition not in exit_conditions:
                    exit_conditions.append(f"If {condition}, then {outcome}")

        # Extract performance claims
        sharpe = self._extract_sharpe(text)
        returns = self._extract_returns(text)

        # Determine timeframe from entities
        timeframe = None
        if "frequency" in entities:
            timeframe = list(entities["frequency"])[0]

        # Determine asset class
        asset_class = None
        if "asset_class" in entities:
            asset_class = ", ".join(entities["asset_class"])

        # Generate enhanced description
        description = self._generate_description_enhanced(
            idea_type, indicators, paper.title, entities, relationships
        )

        # Adjust confidence based on extraction quality
        adjusted_confidence = self._adjust_confidence(
            confidence, indicators, entry_conditions, exit_conditions, parameters
        )

        idea = ExtractedIdea(
            idea_id=f"idea_{self._idea_counter}_{uuid.uuid4().hex[:8]}",
            source_paper_id=paper.paper_id,
            idea_type=idea_type,
            title=f"{idea_type.value.replace('_', ' ').title()} Strategy",
            description=description,
            confidence=adjusted_confidence,
            entry_conditions=entry_conditions[:5],
            exit_conditions=exit_conditions[:5],
            indicators=indicators,
            timeframe=timeframe,
            asset_class=asset_class,
            parameters=parameters,
            claimed_sharpe=sharpe,
            claimed_returns=returns,
            keywords=paper.trading_keywords,
        )

        return idea

    def _extract_conditions_enhanced(
        self,
        text: str,
        sentences: List[str],
        patterns: List[str]
    ) -> List[str]:
        """Extract trading conditions with sentence-level context."""
        conditions = []

        for sentence in sentences:
            for pattern in patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Clean and validate the sentence
                    clean = sentence.strip()[:300]
                    if len(clean) > 20 and clean not in conditions:
                        conditions.append(clean)
                        break  # One match per sentence

        # Also use the original method for phrases within sentences
        phrase_conditions = self._extract_conditions(text, patterns)
        for cond in phrase_conditions:
            if cond not in conditions:
                conditions.append(cond)

        return conditions[:8]

    def _extract_parameters_enhanced(self, text: str) -> Dict[str, Any]:
        """Extract parameters with enhanced NLP analysis."""
        parameters = {}

        # Use TextAnalyzer for time periods
        time_periods = TextAnalyzer.extract_time_periods(text)
        for period_type, values in time_periods.items():
            if values:
                parameters[f"{period_type}_period"] = values[0]

        # Use TextAnalyzer for numbers
        numbers = TextAnalyzer.extract_numbers(text)

        # Look for specific parameter patterns
        param_patterns = {
            "lookback": [
                r"(\d+)[- ]?(?:day|period)\s+(?:lookback|window|rolling)",
                r"lookback\s+(?:period|window)?\s*(?:of\s+)?(\d+)",
            ],
            "threshold": [
                r"threshold\s+(?:of\s+)?(\d+\.?\d*)%?",
                r"(\d+\.?\d*)%?\s+threshold",
                r"trigger[s]?\s+(?:at|when)\s+(\d+\.?\d*)%?",
            ],
            "ma_period": [
                r"(\d+)[- ]?(?:day|period)\s+(?:moving\s+average|MA|SMA|EMA)",
                r"(?:MA|SMA|EMA)\s*\(\s*(\d+)\s*\)",
            ],
            "stop_loss": [
                r"stop[- ]?loss\s+(?:of\s+)?(\d+\.?\d*)%",
                r"(\d+\.?\d*)%\s+stop[- ]?loss",
            ],
            "take_profit": [
                r"take[- ]?profit\s+(?:of\s+)?(\d+\.?\d*)%",
                r"profit[- ]?target\s+(?:of\s+)?(\d+\.?\d*)%",
            ],
            "rsi_level": [
                r"RSI\s+(?:below|under|<)\s+(\d+)",
                r"RSI\s+(?:above|over|>)\s+(\d+)",
            ],
            "holding_period": [
                r"hold(?:ing)?\s+(?:period|for)\s+(?:of\s+)?(\d+)\s*(?:day|week|month)",
                r"(\d+)[- ]?(?:day|week|month)\s+hold(?:ing)?",
            ],
        }

        for param_name, patterns in param_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        parameters[param_name] = value
                    except (ValueError, IndexError):
                        pass
                    break

        # Add original parameter extraction
        basic_params = self._extract_parameters(text)
        for key, value in basic_params.items():
            if key not in parameters:
                parameters[key] = value

        return parameters

    def _generate_description_enhanced(
        self,
        idea_type: IdeaType,
        indicators: List[str],
        title: str,
        entities: Dict[str, Set[str]],
        relationships: List[Tuple[str, str]]
    ) -> str:
        """Generate enhanced description with entity and relationship info."""
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

        # Add asset class info
        if "asset_class" in entities:
            desc += f". Asset class: {', '.join(entities['asset_class'])}"

        # Add market info
        if "market" in entities:
            desc += f". Markets: {', '.join(entities['market'])}"

        # Add frequency info
        if "frequency" in entities:
            desc += f". Frequency: {', '.join(entities['frequency'])}"

        # Add indicators
        if indicators:
            desc += f". Key indicators: {', '.join(indicators[:3])}"

        # Add key relationship if found
        if relationships:
            cond, outcome = relationships[0]
            if len(cond) < 50 and len(outcome) < 50:
                desc += f". Core rule: When {cond}, {outcome}"

        desc += f". Based on: {title[:80]}"

        return desc

    def _adjust_confidence(
        self,
        base_confidence: ConfidenceLevel,
        indicators: List[str],
        entry_conditions: List[str],
        exit_conditions: List[str],
        parameters: Dict[str, Any]
    ) -> ConfidenceLevel:
        """Adjust confidence based on extraction quality."""
        # Score the extraction quality
        score = 0

        # More indicators = more specific strategy
        if len(indicators) >= 3:
            score += 2
        elif len(indicators) >= 1:
            score += 1

        # Entry conditions are critical
        if len(entry_conditions) >= 2:
            score += 2
        elif len(entry_conditions) >= 1:
            score += 1

        # Exit conditions show completeness
        if len(exit_conditions) >= 1:
            score += 1

        # Parameters indicate quantitative approach
        if len(parameters) >= 3:
            score += 2
        elif len(parameters) >= 1:
            score += 1

        # Map score to confidence
        if base_confidence == ConfidenceLevel.LOW:
            if score >= 5:
                return ConfidenceLevel.MEDIUM
            return ConfidenceLevel.LOW
        elif base_confidence == ConfidenceLevel.MEDIUM:
            if score >= 6:
                return ConfidenceLevel.HIGH
            elif score <= 2:
                return ConfidenceLevel.LOW
            return ConfidenceLevel.MEDIUM
        else:  # HIGH
            if score <= 3:
                return ConfidenceLevel.MEDIUM
            return ConfidenceLevel.HIGH

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
        """Get extractor statistics with enhanced metrics."""
        ideas = list(self._extracted_ideas.values())

        # Calculate extraction quality metrics
        total_indicators = sum(len(i.indicators) for i in ideas)
        total_entry_conds = sum(len(i.entry_conditions) for i in ideas)
        total_exit_conds = sum(len(i.exit_conditions) for i in ideas)
        total_params = sum(len(i.parameters) for i in ideas)
        ideas_with_sharpe = len([i for i in ideas if i.claimed_sharpe])
        ideas_with_returns = len([i for i in ideas if i.claimed_returns])

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
            "extraction_quality": {
                "avg_indicators_per_idea": total_indicators / len(ideas) if ideas else 0,
                "avg_entry_conditions": total_entry_conds / len(ideas) if ideas else 0,
                "avg_exit_conditions": total_exit_conds / len(ideas) if ideas else 0,
                "avg_parameters": total_params / len(ideas) if ideas else 0,
                "with_sharpe_claim": ideas_with_sharpe,
                "with_return_claim": ideas_with_returns,
            },
            "by_asset_class": self._count_by_field(ideas, "asset_class"),
            "by_timeframe": self._count_by_field(ideas, "timeframe"),
        }

    def _count_by_field(self, ideas: List[ExtractedIdea], field: str) -> Dict[str, int]:
        """Count ideas by a specific field value."""
        counts = defaultdict(int)
        for idea in ideas:
            value = getattr(idea, field, None)
            if value:
                counts[value] += 1
            else:
                counts["unknown"] += 1
        return dict(counts)


# Singleton accessor
_extractor_instance: Optional[IdeaExtractor] = None

def get_extractor() -> IdeaExtractor:
    """Get the singleton IdeaExtractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = IdeaExtractor()
    return _extractor_instance
