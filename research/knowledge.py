"""
KNOWLEDGE BASE - Store and index research findings

Provides:
- Persistent storage of papers and ideas
- Semantic search over findings
- Relationship mapping between concepts
- Integration with strategy generation
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import json
from pathlib import Path
from loguru import logger

from research.fetcher import AcademicPaper, PaperSource
from research.parser import ExtractedIdea, IdeaType
from research.generator import ResearchHypothesis


@dataclass
class KnowledgeEntry:
    """A piece of knowledge in the base."""
    entry_id: str
    entry_type: str  # "paper", "idea", "hypothesis", "concept"
    content: Dict[str, Any]
    keywords: List[str]
    related_entries: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "entry_type": self.entry_type,
            "content": self.content,
            "keywords": self.keywords,
            "related_entries": self.related_entries,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
        }


@dataclass
class Concept:
    """A trading concept extracted from research."""
    name: str
    description: str
    related_strategies: List[str]
    supporting_papers: List[str]
    confidence: float
    keywords: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "related_strategies": self.related_strategies,
            "supporting_papers": self.supporting_papers,
            "confidence": self.confidence,
            "keywords": self.keywords,
        }


class KnowledgeBase:
    """
    Central knowledge repository for the trading system.

    Stores and indexes:
    - Academic papers
    - Extracted ideas
    - Generated hypotheses
    - Trading concepts
    - Strategy performance learnings
    """

    # Core trading concepts to track
    CORE_CONCEPTS = [
        "momentum", "mean_reversion", "trend_following", "breakout",
        "factor_investing", "volatility", "sentiment", "machine_learning",
        "risk_management", "position_sizing", "portfolio_optimization",
        "market_microstructure", "statistical_arbitrage",
    ]

    _instance: Optional['KnowledgeBase'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, storage_path: str = "memory/knowledge_base.json"):
        if self._initialized:
            return

        self.storage_path = Path(storage_path)

        # Knowledge stores
        self._entries: Dict[str, KnowledgeEntry] = {}
        self._papers: Dict[str, AcademicPaper] = {}
        self._ideas: Dict[str, ExtractedIdea] = {}
        self._hypotheses: Dict[str, ResearchHypothesis] = {}
        self._concepts: Dict[str, Concept] = {}

        # Indexes
        self._keyword_index: Dict[str, Set[str]] = {}  # keyword -> entry_ids
        self._type_index: Dict[str, Set[str]] = {}     # type -> entry_ids
        self._concept_index: Dict[str, Set[str]] = {}  # concept -> entry_ids

        # Initialize core concepts
        self._init_core_concepts()

        # Load existing data
        self._load()

        self._initialized = True
        logger.info("KnowledgeBase initialized")

    def _init_core_concepts(self) -> None:
        """Initialize tracking for core trading concepts."""
        concept_descriptions = {
            "momentum": "Exploiting price continuation patterns",
            "mean_reversion": "Profiting from price returning to average",
            "trend_following": "Following established market trends",
            "breakout": "Trading price breaks through levels",
            "factor_investing": "Systematic exposure to risk factors",
            "volatility": "Trading volatility patterns",
            "sentiment": "Using market sentiment signals",
            "machine_learning": "ML-based prediction models",
            "risk_management": "Controlling portfolio risk",
            "position_sizing": "Optimal trade sizing",
            "portfolio_optimization": "Optimal portfolio construction",
            "market_microstructure": "Market mechanics and execution",
            "statistical_arbitrage": "Statistical relationship trading",
        }

        for concept_name in self.CORE_CONCEPTS:
            self._concepts[concept_name] = Concept(
                name=concept_name,
                description=concept_descriptions.get(concept_name, ""),
                related_strategies=[],
                supporting_papers=[],
                confidence=0.0,
                keywords=[concept_name],
            )

    def add_paper(self, paper: AcademicPaper) -> str:
        """Add a paper to the knowledge base."""
        self._papers[paper.paper_id] = paper

        # Create entry
        entry = KnowledgeEntry(
            entry_id=paper.paper_id,
            entry_type="paper",
            content=paper.to_dict(),
            keywords=paper.trading_keywords,
            related_entries=[],
        )
        self._entries[entry.entry_id] = entry

        # Index
        self._index_entry(entry)

        # Update concepts
        self._update_concepts_from_paper(paper)

        return paper.paper_id

    def add_idea(self, idea: ExtractedIdea) -> str:
        """Add an extracted idea to the knowledge base."""
        self._ideas[idea.idea_id] = idea

        entry = KnowledgeEntry(
            entry_id=idea.idea_id,
            entry_type="idea",
            content=idea.to_dict(),
            keywords=idea.keywords + idea.indicators,
            related_entries=[idea.source_paper_id],
        )
        self._entries[entry.entry_id] = entry

        # Index
        self._index_entry(entry)

        # Link to paper
        if idea.source_paper_id in self._entries:
            self._entries[idea.source_paper_id].related_entries.append(idea.idea_id)

        return idea.idea_id

    def add_hypothesis(self, hypothesis: ResearchHypothesis) -> str:
        """Add a generated hypothesis to the knowledge base."""
        self._hypotheses[hypothesis.hypothesis_id] = hypothesis

        entry = KnowledgeEntry(
            entry_id=hypothesis.hypothesis_id,
            entry_type="hypothesis",
            content=hypothesis.to_dict(),
            keywords=[hypothesis.strategy_type],
            related_entries=[hypothesis.source_idea_id, hypothesis.source_paper_id],
        )
        self._entries[entry.entry_id] = entry

        # Index
        self._index_entry(entry)

        # Update concept
        if hypothesis.strategy_type in self._concepts:
            self._concepts[hypothesis.strategy_type].related_strategies.append(
                hypothesis.hypothesis_id
            )

        return hypothesis.hypothesis_id

    def _index_entry(self, entry: KnowledgeEntry) -> None:
        """Add entry to indexes."""
        # Keyword index
        for keyword in entry.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self._keyword_index:
                self._keyword_index[keyword_lower] = set()
            self._keyword_index[keyword_lower].add(entry.entry_id)

        # Type index
        if entry.entry_type not in self._type_index:
            self._type_index[entry.entry_type] = set()
        self._type_index[entry.entry_type].add(entry.entry_id)

    def _update_concepts_from_paper(self, paper: AcademicPaper) -> None:
        """Update concept tracking from paper."""
        text = f"{paper.title} {paper.abstract}".lower()

        for concept_name, concept in self._concepts.items():
            if concept_name in text:
                concept.supporting_papers.append(paper.paper_id)
                concept.confidence = min(
                    concept.confidence + 0.1,
                    1.0
                )

    def search(
        self,
        query: str,
        entry_type: Optional[str] = None,
        limit: int = 20
    ) -> List[KnowledgeEntry]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            entry_type: Filter by type (paper, idea, hypothesis)
            limit: Max results

        Returns:
            Matching entries sorted by relevance
        """
        query_lower = query.lower()
        query_terms = query_lower.split()

        # Find matching entries
        matching_ids: Dict[str, int] = {}  # entry_id -> match_score

        for term in query_terms:
            # Exact keyword match
            if term in self._keyword_index:
                for entry_id in self._keyword_index[term]:
                    matching_ids[entry_id] = matching_ids.get(entry_id, 0) + 2

            # Partial match
            for keyword, entry_ids in self._keyword_index.items():
                if term in keyword:
                    for entry_id in entry_ids:
                        matching_ids[entry_id] = matching_ids.get(entry_id, 0) + 1

        # Filter by type
        if entry_type:
            type_ids = self._type_index.get(entry_type, set())
            matching_ids = {k: v for k, v in matching_ids.items() if k in type_ids}

        # Sort by score
        sorted_ids = sorted(matching_ids.keys(), key=lambda x: matching_ids[x], reverse=True)

        # Get entries
        results = []
        for entry_id in sorted_ids[:limit]:
            if entry_id in self._entries:
                entry = self._entries[entry_id]
                entry.last_accessed = datetime.utcnow()
                entry.access_count += 1
                results.append(entry)

        return results

    def get_related(self, entry_id: str, limit: int = 10) -> List[KnowledgeEntry]:
        """Get entries related to a given entry."""
        if entry_id not in self._entries:
            return []

        entry = self._entries[entry_id]
        related = []

        # Direct relations
        for related_id in entry.related_entries:
            if related_id in self._entries:
                related.append(self._entries[related_id])

        # Keyword-based relations
        for keyword in entry.keywords[:5]:  # Top 5 keywords
            keyword_lower = keyword.lower()
            if keyword_lower in self._keyword_index:
                for related_id in self._keyword_index[keyword_lower]:
                    if related_id != entry_id and related_id not in [r.entry_id for r in related]:
                        if related_id in self._entries:
                            related.append(self._entries[related_id])

        return related[:limit]

    def get_concept_summary(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """Get summary of a trading concept."""
        if concept_name not in self._concepts:
            return None

        concept = self._concepts[concept_name]

        return {
            "name": concept.name,
            "description": concept.description,
            "num_supporting_papers": len(concept.supporting_papers),
            "num_strategies": len(concept.related_strategies),
            "confidence": concept.confidence,
            "recent_papers": concept.supporting_papers[-5:],
        }

    def get_all_concepts(self) -> List[Dict[str, Any]]:
        """Get summary of all concepts."""
        return [
            self.get_concept_summary(name)
            for name in self._concepts.keys()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "total_entries": len(self._entries),
            "papers": len(self._papers),
            "ideas": len(self._ideas),
            "hypotheses": len(self._hypotheses),
            "concepts_tracked": len(self._concepts),
            "unique_keywords": len(self._keyword_index),
            "most_accessed": self._get_most_accessed(5),
        }

    def _get_most_accessed(self, n: int) -> List[str]:
        """Get most accessed entry IDs."""
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.access_count,
            reverse=True
        )
        return [e.entry_id for e in sorted_entries[:n]]

    def save(self) -> None:
        """Save knowledge base to disk."""
        data = {
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
            "concepts": {k: v.to_dict() for k, v in self._concepts.items()},
            "saved_at": datetime.utcnow().isoformat(),
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Knowledge base saved: {len(self._entries)} entries")

    def _load(self) -> None:
        """Load knowledge base from disk."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Load entries
            for entry_id, entry_data in data.get("entries", {}).items():
                entry = KnowledgeEntry(
                    entry_id=entry_data["entry_id"],
                    entry_type=entry_data["entry_type"],
                    content=entry_data["content"],
                    keywords=entry_data["keywords"],
                    related_entries=entry_data["related_entries"],
                    created_at=datetime.fromisoformat(entry_data["created_at"]),
                    access_count=entry_data.get("access_count", 0),
                )
                self._entries[entry_id] = entry
                self._index_entry(entry)

            logger.info(f"Knowledge base loaded: {len(self._entries)} entries")

        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")

    def export_for_learning(self) -> Dict[str, Any]:
        """Export knowledge for learning module integration."""
        return {
            "strategies_by_type": {
                concept: len(self._concepts[concept].related_strategies)
                for concept in self.CORE_CONCEPTS
            },
            "papers_by_concept": {
                concept: len(self._concepts[concept].supporting_papers)
                for concept in self.CORE_CONCEPTS
            },
            "high_confidence_concepts": [
                concept for concept, data in self._concepts.items()
                if data.confidence > 0.5
            ],
            "total_knowledge_items": len(self._entries),
        }


# Singleton accessor
_kb_instance: Optional[KnowledgeBase] = None

def get_knowledge_base() -> KnowledgeBase:
    """Get the singleton KnowledgeBase instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
