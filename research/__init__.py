"""
RESEARCH MODULE - External knowledge integration for the Living Trading System

This module enables the system to learn from external sources:
- Fetch academic papers from arXiv, SSRN
- Parse and extract trading ideas
- Generate testable hypotheses from research
- Maintain a knowledge base of findings
"""

from research.fetcher import PaperFetcher, get_fetcher
from research.parser import IdeaExtractor, get_extractor
from research.generator import ResearchHypothesisGenerator, get_research_generator
from research.knowledge import KnowledgeBase, get_knowledge_base

__all__ = [
    "PaperFetcher",
    "get_fetcher",
    "IdeaExtractor",
    "get_extractor",
    "ResearchHypothesisGenerator",
    "get_research_generator",
    "KnowledgeBase",
    "get_knowledge_base",
]
