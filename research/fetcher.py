"""
PAPER FETCHER - Fetch academic papers from various sources

Sources:
- arXiv (quantitative finance: q-fin)
- SSRN (Social Science Research Network)
- RePEc (Research Papers in Economics)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio
import httpx
import xml.etree.ElementTree as ET
from loguru import logger


class PaperSource(str, Enum):
    """Academic paper sources."""
    ARXIV = "arxiv"
    SSRN = "ssrn"
    REPEC = "repec"
    SEMANTIC_SCHOLAR = "semantic_scholar"  # Free academic API


@dataclass
class AcademicPaper:
    """Represents an academic paper."""
    paper_id: str
    source: PaperSource
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: datetime
    url: str
    pdf_url: Optional[str] = None

    # Extracted relevance info
    relevance_score: float = 0.0
    trading_keywords: List[str] = field(default_factory=list)
    strategy_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "source": self.source.value,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "categories": self.categories,
            "published_date": self.published_date.isoformat(),
            "url": self.url,
            "pdf_url": self.pdf_url,
            "relevance_score": self.relevance_score,
            "trading_keywords": self.trading_keywords,
            "strategy_hints": self.strategy_hints,
        }


class PaperFetcher:
    """
    Fetches academic papers from research repositories.

    Focuses on:
    - Quantitative finance (q-fin on arXiv)
    - Trading strategies
    - Market microstructure
    - Technical analysis validation
    - Factor investing
    """

    # Trading-related keywords to search for
    TRADING_KEYWORDS = [
        "trading strategy", "momentum", "mean reversion", "market timing",
        "technical analysis", "price prediction", "algorithmic trading",
        "factor investing", "market microstructure", "volatility forecasting",
        "pairs trading", "statistical arbitrage", "trend following",
        "machine learning trading", "deep learning finance",
        "portfolio optimization", "risk management", "alpha generation",
        "market anomaly", "price pattern", "sentiment analysis",
    ]

    # arXiv categories for quantitative finance
    ARXIV_CATEGORIES = [
        "q-fin.TR",  # Trading and Market Microstructure
        "q-fin.PM",  # Portfolio Management
        "q-fin.ST",  # Statistical Finance
        "q-fin.CP",  # Computational Finance
        "q-fin.RM",  # Risk Management
        "q-fin.MF",  # Mathematical Finance
    ]

    _instance: Optional['PaperFetcher'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._papers: Dict[str, AcademicPaper] = {}
        self._last_fetch: Dict[PaperSource, datetime] = {}
        self._fetch_interval = timedelta(hours=6)  # Fetch new papers every 6 hours

        self._initialized = True
        logger.info("PaperFetcher initialized")

    async def fetch_arxiv_papers(
        self,
        max_results: int = 50,
        days_back: int = 30
    ) -> List[AcademicPaper]:
        """
        Fetch recent papers from arXiv quantitative finance categories.
        """
        papers = []

        # Build query for quantitative finance categories
        categories_query = " OR ".join([f"cat:{cat}" for cat in self.ARXIV_CATEGORIES])

        # Add trading keywords
        keywords_query = " OR ".join([f'abs:"{kw}"' for kw in self.TRADING_KEYWORDS[:5]])

        query = f"({categories_query}) AND ({keywords_query})"

        url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

                # Parse XML response
                root = ET.fromstring(response.text)

                # Namespace handling for Atom feed
                ns = {
                    "atom": "http://www.w3.org/2005/Atom",
                    "arxiv": "http://arxiv.org/schemas/atom",
                }

                for entry in root.findall("atom:entry", ns):
                    try:
                        paper = self._parse_arxiv_entry(entry, ns)
                        if paper:
                            # Calculate relevance
                            paper.relevance_score = self._calculate_relevance(paper)
                            paper.trading_keywords = self._extract_keywords(paper)

                            self._papers[paper.paper_id] = paper
                            papers.append(paper)
                    except Exception as e:
                        logger.debug(f"Failed to parse arXiv entry: {e}")

            self._last_fetch[PaperSource.ARXIV] = datetime.utcnow()
            logger.info(f"Fetched {len(papers)} papers from arXiv")

        except Exception as e:
            logger.error(f"Failed to fetch arXiv papers: {e}")

        return papers

    def _parse_arxiv_entry(
        self,
        entry: ET.Element,
        ns: Dict[str, str]
    ) -> Optional[AcademicPaper]:
        """Parse a single arXiv entry."""
        try:
            # Extract paper ID from URL
            id_elem = entry.find("atom:id", ns)
            if id_elem is None or id_elem.text is None:
                return None

            paper_id = id_elem.text.split("/")[-1]

            # Title
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else ""

            # Authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name_elem = author.find("atom:name", ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)

            # Abstract
            summary_elem = entry.find("atom:summary", ns)
            abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""

            # Categories
            categories = []
            for cat in entry.findall("arxiv:primary_category", ns):
                term = cat.get("term")
                if term:
                    categories.append(term)
            for cat in entry.findall("atom:category", ns):
                term = cat.get("term")
                if term and term not in categories:
                    categories.append(term)

            # Published date
            published_elem = entry.find("atom:published", ns)
            published_str = published_elem.text if published_elem is not None and published_elem.text else ""
            try:
                published_date = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            except:
                published_date = datetime.utcnow()

            # URLs
            url = id_elem.text if id_elem is not None and id_elem.text else ""
            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
                    break

            return AcademicPaper(
                paper_id=f"arxiv:{paper_id}",
                source=PaperSource.ARXIV,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published_date=published_date,
                url=url,
                pdf_url=pdf_url,
            )

        except Exception as e:
            logger.debug(f"Error parsing arXiv entry: {e}")
            return None

    def _calculate_relevance(self, paper: AcademicPaper) -> float:
        """Calculate how relevant a paper is for trading strategy development."""
        score = 0.0
        text = f"{paper.title} {paper.abstract}".lower()

        # Check for trading keywords
        keyword_matches = sum(1 for kw in self.TRADING_KEYWORDS if kw.lower() in text)
        score += min(keyword_matches * 0.1, 0.5)

        # Check for quantitative methods
        quant_terms = ["backtest", "sharpe", "returns", "portfolio", "alpha", "beta",
                      "regression", "prediction", "forecast", "signal"]
        quant_matches = sum(1 for term in quant_terms if term in text)
        score += min(quant_matches * 0.05, 0.3)

        # Check for actionable strategies
        action_terms = ["strategy", "implement", "trade", "buy", "sell", "position",
                       "entry", "exit", "indicator"]
        action_matches = sum(1 for term in action_terms if term in text)
        score += min(action_matches * 0.05, 0.2)

        return min(score, 1.0)

    def _extract_keywords(self, paper: AcademicPaper) -> List[str]:
        """Extract trading-relevant keywords from paper."""
        text = f"{paper.title} {paper.abstract}".lower()
        found = []

        for keyword in self.TRADING_KEYWORDS:
            if keyword.lower() in text:
                found.append(keyword)

        return found

    async def fetch_semantic_scholar(
        self,
        max_results: int = 30,
    ) -> List[AcademicPaper]:
        """
        Fetch papers from Semantic Scholar API (free, no key required).
        Searches for quantitative finance and trading papers.
        """
        papers = []

        # Search queries for trading-related papers
        search_queries = [
            "algorithmic trading machine learning",
            "momentum trading strategy",
            "mean reversion stock market",
            "portfolio optimization neural network",
            "quantitative finance deep learning",
        ]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for query in search_queries[:3]:  # Limit queries to avoid rate limits
                    url = "https://api.semanticscholar.org/graph/v1/paper/search"
                    params = {
                        "query": query,
                        "limit": max_results // 3,
                        "fields": "paperId,title,abstract,authors,year,url,citationCount",
                    }

                    try:
                        response = await client.get(url, params=params)
                        if response.status_code == 429:
                            logger.warning("Semantic Scholar rate limit hit, waiting...")
                            await asyncio.sleep(5)
                            continue
                        response.raise_for_status()
                        data = response.json()

                        for item in data.get("data", []):
                            try:
                                paper = self._parse_semantic_scholar_entry(item)
                                if paper and paper.paper_id not in self._papers:
                                    paper.relevance_score = self._calculate_relevance(paper)
                                    paper.trading_keywords = self._extract_keywords(paper)
                                    self._papers[paper.paper_id] = paper
                                    papers.append(paper)
                            except Exception as e:
                                logger.debug(f"Failed to parse Semantic Scholar entry: {e}")

                        # Rate limit between queries
                        await asyncio.sleep(1)

                    except httpx.HTTPStatusError as e:
                        logger.debug(f"Semantic Scholar query failed: {e}")
                        continue

            self._last_fetch[PaperSource.SEMANTIC_SCHOLAR] = datetime.utcnow()
            logger.info(f"Fetched {len(papers)} papers from Semantic Scholar")

        except Exception as e:
            logger.error(f"Failed to fetch Semantic Scholar papers: {e}")

        return papers

    def _parse_semantic_scholar_entry(self, data: Dict[str, Any]) -> Optional[AcademicPaper]:
        """Parse a Semantic Scholar API response entry."""
        try:
            paper_id = data.get("paperId", "")
            if not paper_id:
                return None

            title = data.get("title", "")
            abstract = data.get("abstract", "") or ""

            # Skip if no useful content
            if not title or len(abstract) < 50:
                return None

            authors = []
            for author in data.get("authors", []):
                name = author.get("name", "")
                if name:
                    authors.append(name)

            year = data.get("year")
            published_date = datetime(year, 1, 1) if year else datetime.utcnow()

            url = data.get("url", f"https://www.semanticscholar.org/paper/{paper_id}")

            return AcademicPaper(
                paper_id=f"ss:{paper_id}",
                source=PaperSource.SEMANTIC_SCHOLAR,
                title=title,
                authors=authors[:5],  # Limit authors
                abstract=abstract,
                categories=["quantitative-finance"],
                published_date=published_date,
                url=url,
            )
        except Exception as e:
            logger.debug(f"Error parsing Semantic Scholar entry: {e}")
            return None

    async def fetch_ssrn_rss(self, max_results: int = 20) -> List[AcademicPaper]:
        """
        Fetch papers from SSRN via RSS feed.
        SSRN provides RSS feeds for various research networks.
        """
        papers = []

        # SSRN RSS feeds for financial research
        rss_urls = [
            "https://papers.ssrn.com/sol3/Jeljour_results.cfm?form_name=journalBrowse&journal_id=3526834&Network=no&lim=false&npage=1&SortOrder=ab_approval_date&stype=rss",  # Quantitative Finance
        ]

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                for rss_url in rss_urls:
                    try:
                        response = await client.get(rss_url)
                        if response.status_code != 200:
                            continue

                        # Parse RSS XML
                        root = ET.fromstring(response.text)
                        channel = root.find("channel")
                        if channel is None:
                            continue

                        for item in channel.findall("item")[:max_results]:
                            try:
                                paper = self._parse_ssrn_rss_item(item)
                                if paper and paper.paper_id not in self._papers:
                                    paper.relevance_score = self._calculate_relevance(paper)
                                    paper.trading_keywords = self._extract_keywords(paper)
                                    self._papers[paper.paper_id] = paper
                                    papers.append(paper)
                            except Exception as e:
                                logger.debug(f"Failed to parse SSRN RSS item: {e}")

                    except Exception as e:
                        logger.debug(f"Failed to fetch SSRN RSS: {e}")
                        continue

            self._last_fetch[PaperSource.SSRN] = datetime.utcnow()
            logger.info(f"Fetched {len(papers)} papers from SSRN RSS")

        except Exception as e:
            logger.error(f"Failed to fetch SSRN papers: {e}")

        return papers

    def _parse_ssrn_rss_item(self, item: ET.Element) -> Optional[AcademicPaper]:
        """Parse an SSRN RSS item."""
        try:
            title_elem = item.find("title")
            title = title_elem.text if title_elem is not None and title_elem.text else ""

            link_elem = item.find("link")
            url = link_elem.text if link_elem is not None and link_elem.text else ""

            # Extract paper ID from URL
            paper_id = url.split("abstract_id=")[-1] if "abstract_id=" in url else url.split("/")[-1]

            desc_elem = item.find("description")
            abstract = desc_elem.text if desc_elem is not None and desc_elem.text else ""

            # Clean HTML from description
            import re
            abstract = re.sub(r'<[^>]+>', '', abstract)

            pub_date_elem = item.find("pubDate")
            try:
                from email.utils import parsedate_to_datetime
                published_date = parsedate_to_datetime(pub_date_elem.text) if pub_date_elem is not None and pub_date_elem.text else datetime.utcnow()
            except:
                published_date = datetime.utcnow()

            if not title or len(abstract) < 30:
                return None

            return AcademicPaper(
                paper_id=f"ssrn:{paper_id}",
                source=PaperSource.SSRN,
                title=title,
                authors=[],  # RSS doesn't include authors easily
                abstract=abstract[:2000],  # Limit length
                categories=["finance", "quantitative"],
                published_date=published_date,
                url=url,
            )
        except Exception as e:
            logger.debug(f"Error parsing SSRN RSS item: {e}")
            return None

    async def fetch_all_sources(self, max_per_source: int = 30) -> List[AcademicPaper]:
        """Fetch papers from all available sources."""
        all_papers = []
        errors = []

        # Fetch from arXiv (primary source)
        try:
            arxiv_papers = await self.fetch_arxiv_papers(max_results=max_per_source)
            all_papers.extend(arxiv_papers)
            logger.info(f"arXiv: fetched {len(arxiv_papers)} papers")
        except Exception as e:
            errors.append(f"arXiv: {e}")

        # Small delay between sources
        await asyncio.sleep(1)

        # Fetch from Semantic Scholar
        try:
            ss_papers = await self.fetch_semantic_scholar(max_results=max_per_source)
            all_papers.extend(ss_papers)
            logger.info(f"Semantic Scholar: fetched {len(ss_papers)} papers")
        except Exception as e:
            errors.append(f"Semantic Scholar: {e}")

        await asyncio.sleep(1)

        # Fetch from SSRN RSS
        try:
            ssrn_papers = await self.fetch_ssrn_rss(max_results=max_per_source // 2)
            all_papers.extend(ssrn_papers)
            logger.info(f"SSRN: fetched {len(ssrn_papers)} papers")
        except Exception as e:
            errors.append(f"SSRN: {e}")

        if errors:
            logger.warning(f"Some sources failed: {errors}")

        # Deduplicate by title similarity
        all_papers = self._deduplicate_papers(all_papers)

        # Sort by relevance
        all_papers.sort(key=lambda p: p.relevance_score, reverse=True)

        logger.info(f"Total unique papers fetched: {len(all_papers)}")
        return all_papers

    def _deduplicate_papers(self, papers: List[AcademicPaper]) -> List[AcademicPaper]:
        """Remove duplicate papers based on title similarity."""
        seen_titles = set()
        unique_papers = []

        for paper in papers:
            # Normalize title for comparison
            normalized = paper.title.lower().strip()
            # Remove common words for better matching
            normalized = " ".join(w for w in normalized.split() if len(w) > 3)

            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique_papers.append(paper)

        return unique_papers

    def get_recent_papers(
        self,
        min_relevance: float = 0.3,
        limit: int = 20
    ) -> List[AcademicPaper]:
        """Get recent papers above relevance threshold."""
        papers = [p for p in self._papers.values() if p.relevance_score >= min_relevance]
        papers.sort(key=lambda p: p.published_date, reverse=True)
        return papers[:limit]

    def get_most_relevant(self, limit: int = 10) -> List[AcademicPaper]:
        """Get most relevant papers for trading strategy development."""
        papers = list(self._papers.values())
        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return papers[:limit]

    def search_papers(self, query: str) -> List[AcademicPaper]:
        """Search cached papers by query."""
        query_lower = query.lower()
        results = []

        for paper in self._papers.values():
            text = f"{paper.title} {paper.abstract}".lower()
            if query_lower in text:
                results.append(paper)

        results.sort(key=lambda p: p.relevance_score, reverse=True)
        return results

    def needs_refresh(self, source: PaperSource) -> bool:
        """Check if a source needs to be refreshed."""
        last = self._last_fetch.get(source)
        if not last:
            return True
        return datetime.utcnow() - last > self._fetch_interval

    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics."""
        return {
            "total_papers": len(self._papers),
            "papers_by_source": {
                source.value: len([p for p in self._papers.values() if p.source == source])
                for source in PaperSource
            },
            "avg_relevance": sum(p.relevance_score for p in self._papers.values()) / len(self._papers)
                            if self._papers else 0,
            "last_fetch": {
                source.value: last.isoformat() if last else None
                for source, last in self._last_fetch.items()
            },
        }


# Singleton accessor
_fetcher_instance: Optional[PaperFetcher] = None

def get_fetcher() -> PaperFetcher:
    """Get the singleton PaperFetcher instance."""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = PaperFetcher()
    return _fetcher_instance
