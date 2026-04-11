"""
AI-powered benchmark consolidation tool.

This module uses Claude to consolidate variations of benchmark names
into canonical forms, distinguishing between true variants and distinct benchmarks.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from datetime import datetime
import json

from ._claude_client import call_claude_json, is_anthropic_available
from .google_search import scrape_google_search
from .benchmark_validation import normalize_benchmark_name

logger = logging.getLogger(__name__)

# T077: Fuzzy matching threshold for benchmark consolidation
FUZZY_MATCH_THRESHOLD = 0.90

# Disambiguation cache to avoid repeated web searches (T083)
_disambiguation_cache: Dict[str, str] = {}


def consolidate_benchmarks(
    benchmark_names: List[str],
    claude_fn: Optional[callable] = None,
    cooccurrences: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Consolidate benchmark name variations into canonical forms.

    Uses Claude to analyze benchmark names and create mappings from variations
    to canonical names, while identifying truly distinct benchmarks.

    T078: Uses FUZZY_MATCH_THRESHOLD for similarity comparisons (configurable via config)
    T080-T082: Triggers web search disambiguation for ambiguous pairs (<90% similarity)
    T084: Adds web_search_used flag to consolidation JSON output

    Args:
        benchmark_names: List of benchmark names to consolidate
        claude_fn: Optional Claude API function for dependency injection.
                   Should accept (prompt, system_prompt) and return dict.
        cooccurrences: Optional list of benchmark pairs that appear side-by-side
                      in the same document sections. These pairs will NOT be merged
                      during consolidation (e.g., "MMLU" and "MMLU-Pro" in same table).
        config: Optional configuration dict with consolidation settings
                (fuzzy_match_threshold, enable_web_search, etc.)

    Returns:
        Dictionary containing consolidation results:
            - consolidations: List of canonical names with their variations
            - distinct_benchmarks: List of benchmarks that are truly distinct
            - uncertain_mappings: List of ambiguous cases requiring review
            - metadata: Consolidation metadata
            - web_search_used: Boolean flag indicating if web search was used (T084)

        See prompts/consolidate.md for full schema.

    Raises:
        ValueError: If benchmark_names is empty or invalid
        RuntimeError: If consolidation fails

    Example:
        >>> names = ["MMLU", "mmlu", "MMLU-Pro", "GSM8K", "GSM-8K"]
        >>> result = consolidate_benchmarks(names)
        >>> for cons in result['consolidations']:
        ...     print(f"{cons['canonical_name']}: {cons['variations']}")
    """
    try:
        if not benchmark_names or not isinstance(benchmark_names, list):
            raise ValueError("benchmark_names must be a non-empty list")

        if len(benchmark_names) == 0:
            raise ValueError("benchmark_names list is empty")

        # Load configuration (T078, T079)
        threshold = FUZZY_MATCH_THRESHOLD
        enable_web_search = True
        benchmark_aliases = {}
        ai_review_min = 0.70
        ai_review_max = 0.89

        if config:
            consolidation_config = config.get("consolidation", {})
            threshold = consolidation_config.get("fuzzy_match_threshold", FUZZY_MATCH_THRESHOLD)
            enable_web_search = consolidation_config.get("enable_web_search", True)
            benchmark_aliases = consolidation_config.get("benchmark_aliases", {})
            ai_review_min = consolidation_config.get("ai_review_min_similarity", 0.70)
            ai_review_max = consolidation_config.get("ai_review_max_similarity", 0.89)

        logger.info(f"Using fuzzy match threshold: {threshold:.2%}, web search: {enable_web_search}")
        if benchmark_aliases:
            logger.info(f"Loaded {len(benchmark_aliases)} benchmark aliases from config")

        # Step 0: Normalize benchmark names (AIME variants, remove "from X" suffixes, etc.)
        normalized_names = []
        normalization_count = 0
        for name in benchmark_names:
            normalized = normalize_benchmark_name(name)
            if normalized != name:
                normalization_count += 1
                logger.debug(f"Normalized: '{name}' → '{normalized}'")
            normalized_names.append(normalized)

        benchmark_names = normalized_names
        if normalization_count > 0:
            logger.info(f"Normalized {normalization_count} benchmark name variants")

        # Step 1: Apply benchmark aliases (resolve known alternate names)
        benchmark_names = _apply_aliases(benchmark_names, benchmark_aliases)

        # Remove duplicates while preserving order
        unique_names = list(dict.fromkeys(benchmark_names))

        logger.info(f"Consolidating {len(unique_names)} unique benchmark names")

        # Step 2: AI review for questionable pairs (70-89% similarity)
        ai_reviews = []
        if enable_web_search and ai_review_min < ai_review_max:
            ai_reviews = _ai_review_questionable_pairs(
                unique_names,
                min_similarity=ai_review_min,
                max_similarity=ai_review_max,
                enable_web_search=enable_web_search
            )

            # Apply AI review decisions: merge pairs identified as same benchmark
            if ai_reviews:
                for review in ai_reviews:
                    if review["same_benchmark"] and review["canonical_name"]:
                        # Replace both names with canonical name in the list
                        name1 = review["benchmark1"]
                        name2 = review["benchmark2"]
                        canonical = review["canonical_name"]

                        unique_names = [
                            canonical if (name == name1 or name == name2) else name
                            for name in unique_names
                        ]

                        logger.info(
                            f"Pre-consolidated based on AI review: '{name1}' + '{name2}' → '{canonical}'"
                        )

                # Remove duplicates again after AI merges
                unique_names = list(dict.fromkeys(unique_names))
                logger.info(f"After AI review: {len(unique_names)} unique benchmark names")

        # Build consolidation prompt
        prompt = _build_consolidation_prompt(unique_names)

        # Call Claude (use injected function or default)
        if claude_fn is None:
            if not is_anthropic_available():
                raise RuntimeError(
                    "Anthropic API not available. Set ANTHROPIC_API_KEY environment "
                    "variable or install anthropic package (pip install anthropic)"
                )
            result = call_claude_json(prompt=prompt)
        else:
            result = claude_fn(prompt=prompt)

        # Validate result structure
        if not isinstance(result, dict):
            raise RuntimeError("Invalid response format from Claude")

        # Ensure required keys exist
        if "consolidations" not in result:
            result["consolidations"] = []
        if "distinct_benchmarks" not in result:
            result["distinct_benchmarks"] = []
        if "uncertain_mappings" not in result:
            result["uncertain_mappings"] = []

        # T080-T082: Apply web search disambiguation for uncertain pairs
        web_search_used = False
        if enable_web_search and result.get("uncertain_mappings"):
            web_search_used = _apply_web_search_disambiguation(
                result, threshold
            )

        # Apply side-by-side disambiguation
        if cooccurrences:
            result = _apply_cooccurrence_disambiguation(result, cooccurrences)
            logger.info(f"Applied {len(cooccurrences)} co-occurrence constraints")

        # Add metadata
        if "metadata" not in result:
            result["metadata"] = {}

        result["metadata"]["total_input_names"] = len(benchmark_names)  # Original count before aliases
        result["metadata"]["aliases_resolved"] = len(benchmark_names) - len(unique_names)
        result["metadata"]["total_canonical_names"] = len(result["consolidations"])
        result["metadata"]["consolidation_date"] = datetime.utcnow().isoformat()
        result["metadata"]["cooccurrence_constraints"] = len(cooccurrences) if cooccurrences else 0
        result["metadata"]["fuzzy_match_threshold"] = threshold

        # AI review metadata
        result["metadata"]["ai_reviews_performed"] = len(ai_reviews)
        ai_merges = sum(1 for r in ai_reviews if r["same_benchmark"])
        result["metadata"]["ai_merges"] = ai_merges

        # T084: Add web_search_used flag
        result["web_search_used"] = web_search_used or any(r.get("web_search_used", False) for r in ai_reviews)

        logger.info(
            f"Consolidated {len(unique_names)} names into "
            f"{len(result['consolidations'])} canonical names "
            f"(web search: {web_search_used})"
        )

        return result

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Benchmark consolidation failed: {e}")
        raise RuntimeError(f"Failed to consolidate benchmarks: {e}")


def _build_consolidation_prompt(benchmark_names: List[str]) -> str:
    """
    Build the prompt for benchmark consolidation.

    Loads the consolidation prompt template and fills in the benchmark names.

    Args:
        benchmark_names: List of benchmark names

    Returns:
        Complete prompt string
    """
    # Load prompt template
    prompt_path = Path(__file__).parent.parent / "prompts" / "consolidate.md"

    try:
        with open(prompt_path, "r") as f:
            template = f.read()
    except FileNotFoundError:
        logger.warning("Prompt template not found, using basic prompt")
        template = (
            "Consolidate the following benchmark names into canonical forms. "
            "Distinguish between variations of the same benchmark and truly distinct benchmarks."
        )

    # Build input JSON
    import json
    input_json = json.dumps({"benchmark_names": benchmark_names}, indent=2)

    # Build full prompt
    prompt = f"""{template}

## Benchmark Names to Consolidate

{input_json}

Analyze these benchmark names and return the consolidation results as JSON following the schema defined above.
"""

    return prompt


def create_name_mapping(
    consolidation_result: Dict[str, Any]
) -> Dict[str, str]:
    """
    Create a simple mapping from variation names to canonical names.

    Utility function to extract a direct name mapping dictionary.

    Args:
        consolidation_result: Result from consolidate_benchmarks()

    Returns:
        Dictionary mapping variation names to canonical names

    Example:
        >>> result = consolidate_benchmarks(names)
        >>> mapping = create_name_mapping(result)
        >>> print(mapping["mmlu"])  # "MMLU"
        >>> print(mapping["GSM-8K"])  # "GSM8K"
    """
    mapping = {}

    for consolidation in consolidation_result.get("consolidations", []):
        canonical = consolidation["canonical_name"]
        variations = consolidation.get("variations", [])

        for variation in variations:
            mapping[variation] = canonical

    return mapping


def apply_consolidation(
    benchmarks: List[Dict[str, Any]],
    consolidation_result: Dict[str, Any],
    add_canonical_field: bool = True,
) -> List[Dict[str, Any]]:
    """
    Apply consolidation mapping to benchmark data.

    Updates benchmark entries with canonical names based on consolidation results.

    Args:
        benchmarks: List of benchmark dictionaries (from extraction)
        consolidation_result: Result from consolidate_benchmarks()
        add_canonical_field: If True, adds "canonical_name" field.
                            If False, replaces "name" field.

    Returns:
        Updated benchmark list with canonical names

    Example:
        >>> benchmarks = [
        ...     {"name": "mmlu", "score": 82.5},
        ...     {"name": "GSM-8K", "score": 94.2}
        ... ]
        >>> result = consolidate_benchmarks(["mmlu", "MMLU", "GSM-8K", "GSM8K"])
        >>> updated = apply_consolidation(benchmarks, result)
        >>> print(updated[0]["canonical_name"])  # "MMLU"
    """
    # Create name mapping
    mapping = create_name_mapping(consolidation_result)

    # Apply mapping to benchmarks
    updated_benchmarks = []

    for benchmark in benchmarks:
        # Make a copy
        updated = benchmark.copy()

        original_name = benchmark.get("name")
        if original_name:
            canonical_name = mapping.get(original_name, original_name)

            if add_canonical_field:
                updated["canonical_name"] = canonical_name
            else:
                updated["name"] = canonical_name

        updated_benchmarks.append(updated)

    return updated_benchmarks


def extract_benchmark_names(
    benchmarks: List[Dict[str, Any]]
) -> List[str]:
    """
    Extract unique benchmark names from benchmark data.

    Utility function to get all unique benchmark names from extracted data.

    Args:
        benchmarks: List of benchmark dictionaries

    Returns:
        List of unique benchmark names

    Example:
        >>> benchmarks = [
        ...     {"name": "MMLU", "score": 82.5},
        ...     {"name": "mmlu", "score": 83.0},
        ...     {"name": "GSM8K", "score": 94.2}
        ... ]
        >>> names = extract_benchmark_names(benchmarks)
        >>> print(names)  # ["MMLU", "mmlu", "GSM8K"]
    """
    names: Set[str] = set()

    for benchmark in benchmarks:
        name = benchmark.get("name")
        if name:
            names.add(name)

    return list(names)


def _apply_most_common_nomenclature(
    consolidation_result: Dict[str, Any],
    usage_counts: Dict[str, int]
) -> Dict[str, Any]:
    """
    Apply "most common nomenclature" rule to select canonical names.

    For each group of consolidated variants, selects the variant used by
    the most models as the canonical name. Implements tie-breaking rules
    from SPECIFICATIONS.md Section 4.3.

    Tie-breaking rules (if counts are equal):
    1. Prefer uppercase > lowercase > mixed case
    2. Examples: "MMLU" > "mmlu" > "Mmlu"

    Args:
        consolidation_result: Result from Claude consolidation
        usage_counts: Dict mapping benchmark names to usage counts

    Returns:
        Updated consolidation result with canonical names adjusted

    Example:
        >>> result = {"consolidations": [{"canonical_name": "mmlu", "variations": ["MMLU", "mmlu"]}]}
        >>> usage = {"MMLU": 10, "mmlu": 3}
        >>> updated = _apply_most_common_nomenclature(result, usage)
        >>> print(updated["consolidations"][0]["canonical_name"])  # "MMLU"
    """
    for consolidation in consolidation_result.get("consolidations", []):
        variations = consolidation.get("variations", [])
        current_canonical = consolidation.get("canonical_name")

        if not variations or len(variations) <= 1:
            continue

        # Count usage for each variation
        variant_counts = {}
        for variant in variations:
            variant_counts[variant] = usage_counts.get(variant, 0)

        # Find max usage count
        max_count = max(variant_counts.values()) if variant_counts else 0

        # Get all variants with max count (for tie-breaking)
        top_variants = [v for v, c in variant_counts.items() if c == max_count]

        if len(top_variants) == 0:
            # No usage data, keep AI's choice
            logger.debug(f"No usage data for {current_canonical}, keeping AI choice")
            continue
        elif len(top_variants) == 1:
            # Clear winner
            selected = top_variants[0]
            if selected != current_canonical:
                logger.info(
                    f"Most common nomenclature: '{selected}' (used by {max_count} models) "
                    f"selected over '{current_canonical}'"
                )
                consolidation["canonical_name"] = selected
                consolidation["notes"] = (
                    f"{consolidation.get('notes', '')} "
                    f"Canonical name selected based on usage: {max_count} models use '{selected}'."
                ).strip()
        else:
            # Tie - apply tie-breaking rules
            selected = _tie_break_canonical_name(top_variants, max_count)
            if selected != current_canonical:
                logger.info(
                    f"Tie-breaking: '{selected}' selected from {top_variants} "
                    f"(all used by {max_count} models)"
                )
                consolidation["canonical_name"] = selected
                consolidation["notes"] = (
                    f"{consolidation.get('notes', '')} "
                    f"Tie-breaking applied: {len(top_variants)} variants tied at {max_count} models. "
                    f"Selected '{selected}' (uppercase > lowercase > mixed case)."
                ).strip()

    return consolidation_result


def _apply_cooccurrence_disambiguation(
    consolidation_result: Dict[str, Any],
    cooccurrences: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Apply side-by-side benchmark disambiguation.

    When two benchmarks appear together in the same document section (same table,
    same paragraph), they should be treated as distinct benchmarks, not merged.

    This function splits any consolidation groups where members co-occur.

    Args:
        consolidation_result: Result from Claude consolidation
        cooccurrences: List of benchmark pairs found side-by-side
                      [{"benchmark_a": "MMLU", "benchmark_b": "MMLU-Pro", "location": "Table 1"}, ...]

    Returns:
        Updated consolidation result with co-occurring benchmarks separated

    Example:
        Before: consolidations = [{"canonical_name": "MMLU", "variations": ["MMLU", "mmlu", "MMLU-Pro"]}]
        After (with MMLU+MMLU-Pro cooccurrence):
              consolidations = [
                  {"canonical_name": "MMLU", "variations": ["MMLU", "mmlu"]},
                  {"canonical_name": "MMLU-Pro", "variations": ["MMLU-Pro"]}
              ]
    """
    from typing import Tuple

    # Build set of co-occurring pairs (both directions for easy lookup)
    cooccurring_pairs: Set[Tuple[str, str]] = set()
    for cooccur in cooccurrences:
        a = cooccur["benchmark_a"]
        b = cooccur["benchmark_b"]
        cooccurring_pairs.add((a, b))
        cooccurring_pairs.add((b, a))  # Symmetric

    # Process each consolidation group
    new_consolidations = []
    split_count = 0

    for consolidation in consolidation_result.get("consolidations", []):
        variations = consolidation.get("variations", [])

        if len(variations) <= 1:
            # No variations to split
            new_consolidations.append(consolidation)
            continue

        # Check if any variations co-occur
        needs_split = False
        for i in range(len(variations)):
            for j in range(i + 1, len(variations)):
                if (variations[i], variations[j]) in cooccurring_pairs:
                    needs_split = True
                    break
            if needs_split:
                break

        if not needs_split:
            # No co-occurrences, keep as-is
            new_consolidations.append(consolidation)
            continue

        # Split the group: separate co-occurring benchmarks
        # Strategy: Create separate consolidations for each unique variation
        # that co-occurs with another in this group
        separated_variations = set()
        for i in range(len(variations)):
            for j in range(i + 1, len(variations)):
                if (variations[i], variations[j]) in cooccurring_pairs:
                    separated_variations.add(variations[i])
                    separated_variations.add(variations[j])

        # If some variations need separation, split them out
        if separated_variations:
            # Keep non-separated variations together
            kept_variations = [v for v in variations if v not in separated_variations]

            if kept_variations:
                # Create consolidation for non-separated variations
                new_consolidations.append({
                    "canonical_name": kept_variations[0],
                    "variations": kept_variations,
                    "benchmark_type": consolidation.get("benchmark_type", "same"),
                    "confidence": consolidation.get("confidence", 1.0),
                    "notes": f"{consolidation.get('notes', '')} (Split from original group due to co-occurrence)".strip()
                })

            # Create individual consolidations for separated variations
            for var in sorted(separated_variations):
                new_consolidations.append({
                    "canonical_name": var,
                    "variations": [var],
                    "benchmark_type": "distinct",
                    "confidence": 1.0,
                    "notes": f"Separated due to side-by-side appearance with similar benchmark name"
                })

            split_count += 1
        else:
            new_consolidations.append(consolidation)

    consolidation_result["consolidations"] = new_consolidations

    if split_count > 0:
        logger.info(f"Split {split_count} consolidation groups due to co-occurrence")

    return consolidation_result


def _tie_break_canonical_name(variants: List[str], count: int) -> str:
    """
    Apply tie-breaking rules when multiple variants have equal usage.

    Tie-breaking order:
    1. Uppercase (all characters uppercase)
    2. Lowercase (all characters lowercase)
    3. Mixed case

    Args:
        variants: List of variant names with equal usage counts
        count: The tied usage count

    Returns:
        Selected canonical name

    Example:
        >>> _tie_break_canonical_name(["MMLU", "mmlu", "Mmlu"], 5)
        'MMLU'
        >>> _tie_break_canonical_name(["mmlu", "Mmlu"], 5)
        'mmlu'
    """
    # Categorize variants
    uppercase = []
    lowercase = []
    mixed_case = []

    for variant in variants:
        # Only consider alphabetic characters for case classification
        alpha_chars = ''.join(c for c in variant if c.isalpha())
        if not alpha_chars:
            # No alphabetic characters, treat as mixed
            mixed_case.append(variant)
        elif alpha_chars.isupper():
            uppercase.append(variant)
        elif alpha_chars.islower():
            lowercase.append(variant)
        else:
            mixed_case.append(variant)

    # Apply preference: uppercase > lowercase > mixed
    if uppercase:
        selected = uppercase[0]
        logger.debug(f"Tie-break: Selected uppercase variant '{selected}'")
        return selected
    elif lowercase:
        selected = lowercase[0]
        logger.debug(f"Tie-break: Selected lowercase variant '{selected}'")
        return selected
    else:
        selected = mixed_case[0] if mixed_case else variants[0]
        logger.debug(f"Tie-break: Selected mixed-case variant '{selected}'")
        return selected


def trigger_web_search(benchmark1: str, benchmark2: str, similarity: float) -> Dict[str, Any]:
    """
    Trigger web search disambiguation when similarity is below threshold.

    T080: Implements web search for ambiguous benchmark pairs (<90% similarity).
    T081: Fetches top 3 Google search results for "{benchmark1} vs {benchmark2}".
    T082: Uses Claude to analyze search results and determine if same/different.
    T083: Caches disambiguation decisions to avoid repeated searches.

    Args:
        benchmark1: First benchmark name
        benchmark2: Second benchmark name
        similarity: Fuzzy match similarity score (0.0 to 1.0)

    Returns:
        Dictionary containing:
            - are_same: Boolean indicating if benchmarks are the same
            - confidence: Float (0.0 to 1.0) indicating confidence level
            - evidence: String describing the evidence found
            - search_results_used: Number of search results analyzed
            - cached: Boolean indicating if result was from cache

    Example:
        >>> result = trigger_web_search("MMLU", "MMLU-Pro", 0.85)
        >>> print(result["are_same"])  # False
        >>> print(result["evidence"])  # "MMLU-Pro is an enhanced version..."
    """
    # T083: Check cache first
    cache_key = f"{benchmark1.lower()}|{benchmark2.lower()}"
    reverse_cache_key = f"{benchmark2.lower()}|{benchmark1.lower()}"

    if cache_key in _disambiguation_cache:
        cached_result = json.loads(_disambiguation_cache[cache_key])
        cached_result["cached"] = True
        logger.info(f"Using cached disambiguation for '{benchmark1}' vs '{benchmark2}'")
        return cached_result

    if reverse_cache_key in _disambiguation_cache:
        cached_result = json.loads(_disambiguation_cache[reverse_cache_key])
        cached_result["cached"] = True
        logger.info(f"Using cached disambiguation for '{benchmark1}' vs '{benchmark2}' (reversed)")
        return cached_result

    logger.info(
        f"Triggering web search disambiguation for '{benchmark1}' vs '{benchmark2}' "
        f"(similarity: {similarity:.2%})"
    )

    # T081: Search Google for top 3 results
    query = f'"{benchmark1}" vs "{benchmark2}"'
    try:
        search_results = scrape_google_search(query, max_results=3, delay=1.0)
    except Exception as e:
        logger.warning(f"Web search failed for '{benchmark1}' vs '{benchmark2}': {e}")
        search_results = []

    # If search failed or returned no results, fall back to heuristic
    if not search_results:
        logger.warning(f"No search results found for '{benchmark1}' vs '{benchmark2}', using heuristic")
        result = _heuristic_disambiguation(benchmark1, benchmark2, similarity)
        # Cache heuristic result too (T083)
        _disambiguation_cache[cache_key] = json.dumps(result)
        result["cached"] = False
        return result

    # T082: Use Claude to analyze search results
    result = _analyze_search_results_with_claude(
        benchmark1, benchmark2, search_results, similarity
    )

    # T083: Cache the result
    _disambiguation_cache[cache_key] = json.dumps(result)
    result["cached"] = False

    return result


def _analyze_search_results_with_claude(
    benchmark1: str,
    benchmark2: str,
    search_results: List[Dict[str, str]],
    similarity: float
) -> Dict[str, Any]:
    """
    Use Claude to analyze web search results and determine if benchmarks are same/different.

    Args:
        benchmark1: First benchmark name
        benchmark2: Second benchmark name
        search_results: List of search result dicts with url, title, snippet
        similarity: Fuzzy match similarity score

    Returns:
        Disambiguation result dictionary
    """
    # Build prompt for Claude
    search_summary = []
    for i, result in enumerate(search_results[:3], 1):
        search_summary.append({
            "position": i,
            "title": result.get("title", ""),
            "snippet": result.get("snippet", ""),
            "url": result.get("url", "")
        })

    prompt = f"""Analyze these web search results to determine if "{benchmark1}" and "{benchmark2}" are the same benchmark or different benchmarks.

BENCHMARK NAMES:
- Benchmark A: {benchmark1}
- Benchmark B: {benchmark2}
- Name similarity score: {similarity:.2%}

WEB SEARCH RESULTS for "{benchmark1} vs {benchmark2}":
{json.dumps(search_summary, indent=2)}

Based on the search results, determine:
1. Are these the SAME benchmark (just different naming variants)?
2. Or are they DIFFERENT benchmarks (e.g., different versions, subsets, or entirely distinct)?

Return JSON with this structure:
{{
  "are_same": true/false,
  "confidence": 0.0-1.0,
  "evidence": "Brief explanation based on search results",
  "search_results_used": {len(search_summary)}
}}

Guidelines:
- If search results explicitly state they are different versions/variants (e.g., "MMLU-Pro is an enhanced version of MMLU"), return are_same=false
- If results use the names interchangeably or don't distinguish them, return are_same=true
- If results are inconclusive, use confidence to reflect uncertainty
"""

    try:
        if not is_anthropic_available():
            logger.warning("Anthropic API not available, using heuristic")
            return _heuristic_disambiguation(benchmark1, benchmark2, similarity)

        result = call_claude_json(prompt=prompt)

        # Validate result structure
        if not isinstance(result, dict):
            raise ValueError("Invalid response format from Claude")

        # Ensure required fields
        result.setdefault("are_same", similarity >= FUZZY_MATCH_THRESHOLD)
        result.setdefault("confidence", 0.5)
        result.setdefault("evidence", "Analysis completed")
        result.setdefault("search_results_used", len(search_summary))

        logger.info(
            f"Web search analysis: '{benchmark1}' vs '{benchmark2}' -> "
            f"{'SAME' if result['are_same'] else 'DIFFERENT'} "
            f"(confidence: {result['confidence']:.2%})"
        )

        return result

    except Exception as e:
        logger.error(f"Claude analysis failed for web search results: {e}")
        return _heuristic_disambiguation(benchmark1, benchmark2, similarity)


def _heuristic_disambiguation(
    benchmark1: str,
    benchmark2: str,
    similarity: float
) -> Dict[str, Any]:
    """
    Fallback heuristic when web search or Claude analysis fails.

    Uses simple rules:
    - If similarity >= threshold, treat as same
    - If one name is substring of other or contains version number, treat as different
    - Otherwise use similarity score

    Args:
        benchmark1: First benchmark name
        benchmark2: Second benchmark name
        similarity: Fuzzy match similarity score

    Returns:
        Disambiguation result dictionary
    """
    b1_lower = benchmark1.lower()
    b2_lower = benchmark2.lower()

    # Check if one is a substring of the other
    is_substring = b1_lower in b2_lower or b2_lower in b1_lower

    # Check for version patterns (e.g., "MMLU" vs "MMLU-Pro", "GSM8K" vs "GSM8K-v2")
    import re
    version_pattern = r'(-pro|-plus|-v\d+|-\d+\.\d+|_v\d+)'
    has_version = bool(re.search(version_pattern, b1_lower)) or bool(re.search(version_pattern, b2_lower))

    if is_substring and has_version:
        # Likely different versions of same benchmark
        are_same = False
        confidence = 0.8
        evidence = f"One name contains the other with version indicator (substring: {is_substring}, version: {has_version})"
    elif similarity >= FUZZY_MATCH_THRESHOLD:
        # High similarity, likely same
        are_same = True
        confidence = similarity
        evidence = f"High similarity score ({similarity:.2%}) above threshold ({FUZZY_MATCH_THRESHOLD:.2%})"
    else:
        # Low similarity, likely different
        are_same = False
        confidence = 1.0 - similarity
        evidence = f"Low similarity score ({similarity:.2%}) below threshold ({FUZZY_MATCH_THRESHOLD:.2%})"

    logger.info(
        f"Heuristic disambiguation: '{benchmark1}' vs '{benchmark2}' -> "
        f"{'SAME' if are_same else 'DIFFERENT'} (confidence: {confidence:.2%})"
    )

    return {
        "are_same": are_same,
        "confidence": confidence,
        "evidence": evidence,
        "search_results_used": 0
    }


def _apply_web_search_disambiguation(
    consolidation_result: Dict[str, Any],
    threshold: float
) -> bool:
    """
    Apply web search disambiguation to uncertain benchmark mappings.

    Processes uncertain_mappings and uses web search to resolve ambiguous pairs.
    Updates consolidations list based on disambiguation results.

    Args:
        consolidation_result: Result from Claude consolidation with uncertain_mappings
        threshold: Fuzzy match threshold for similarity

    Returns:
        True if web search was used, False otherwise
    """
    uncertain = consolidation_result.get("uncertain_mappings", [])
    if not uncertain:
        return False

    logger.info(f"Processing {len(uncertain)} uncertain mappings with web search")

    web_search_used = False
    resolved_consolidations = []

    for uncertain_pair in uncertain:
        # Extract benchmark names and similarity
        benchmark1 = uncertain_pair.get("benchmark1")
        benchmark2 = uncertain_pair.get("benchmark2")
        similarity = uncertain_pair.get("similarity", 0.0)

        if not benchmark1 or not benchmark2:
            logger.warning(f"Skipping uncertain pair with missing names: {uncertain_pair}")
            continue

        # Trigger web search (T080-T082)
        disambiguation = trigger_web_search(benchmark1, benchmark2, similarity)
        web_search_used = True

        # Update consolidations based on result
        if disambiguation["are_same"]:
            # Merge into same consolidation
            resolved_consolidations.append({
                "canonical_name": benchmark1,  # Use first as canonical
                "variations": [benchmark1, benchmark2],
                "benchmark_type": "same",
                "confidence": disambiguation["confidence"],
                "notes": f"Web search disambiguation: {disambiguation['evidence']}"
            })
            logger.info(f"Merged '{benchmark1}' and '{benchmark2}' based on web search")
        else:
            # Keep as distinct benchmarks
            consolidation_result["distinct_benchmarks"].extend([benchmark1, benchmark2])
            logger.info(f"Separated '{benchmark1}' and '{benchmark2}' based on web search")

    # Add resolved consolidations to result
    consolidation_result["consolidations"].extend(resolved_consolidations)

    # Clear uncertain mappings (all resolved)
    consolidation_result["uncertain_mappings"] = []

    return web_search_used


def _apply_aliases(benchmark_names: List[str], aliases: Dict[str, str]) -> List[str]:
    """
    Apply benchmark aliases to normalize known alternate names.

    Resolves known alternate names to their canonical forms before consolidation.
    This prevents false negatives where known aliases would otherwise be treated
    as distinct benchmarks.

    Args:
        benchmark_names: List of benchmark names to process
        aliases: Dictionary mapping alternate names to canonical names
                 Example: {"ARC-C": "ARC-Challenge", "BBH": "BIG-Bench Hard"}

    Returns:
        List of benchmark names with aliases resolved

    Example:
        >>> aliases = {"ARC-C": "ARC-Challenge", "BBH": "BIG-Bench Hard"}
        >>> names = ["ARC-C", "MMLU", "BBH", "HumanEval"]
        >>> _apply_aliases(names, aliases)
        ['ARC-Challenge', 'MMLU', 'BIG-Bench Hard', 'HumanEval']
    """
    if not aliases:
        return benchmark_names

    resolved_names = []
    alias_count = 0

    for name in benchmark_names:
        # Check exact match first
        if name in aliases:
            canonical = aliases[name]
            resolved_names.append(canonical)
            alias_count += 1
            logger.debug(f"Resolved alias: '{name}' → '{canonical}'")
        else:
            # Check case-insensitive match
            name_lower = name.lower()
            alias_found = False
            for alias, canonical in aliases.items():
                if alias.lower() == name_lower:
                    resolved_names.append(canonical)
                    alias_count += 1
                    logger.debug(f"Resolved alias (case-insensitive): '{name}' → '{canonical}'")
                    alias_found = True
                    break
            if not alias_found:
                resolved_names.append(name)

    if alias_count > 0:
        logger.info(f"Resolved {alias_count} benchmark aliases")

    return resolved_names


def _ai_review_questionable_pairs(
    benchmark_names: List[str],
    min_similarity: float = 0.70,
    max_similarity: float = 0.89,
    enable_web_search: bool = True
) -> List[Dict[str, Any]]:
    """
    Use AI to review questionable benchmark pairs with medium similarity.

    For pairs in the "questionable" similarity range (70-89%), use AI + web search
    to determine if they represent the same benchmark or distinct benchmarks.

    This catches cases like:
    - "ARC-C" vs "ARC-Challenge" (abbreviation vs full name)
    - "BBH" vs "BIG-Bench Hard" (acronym vs full name)
    - "GSM8K" vs "GSM-8K" (spacing variation)

    Args:
        benchmark_names: List of benchmark names to analyze
        min_similarity: Minimum similarity to trigger review (default: 0.70)
        max_similarity: Maximum similarity to trigger review (default: 0.89)
        enable_web_search: Whether to use web search for context (default: True)

    Returns:
        List of AI review results with consolidation decisions

    Example:
        >>> names = ["ARC-C", "ARC-Challenge", "MMLU", "HumanEval"]
        >>> results = _ai_review_questionable_pairs(names)
        >>> # Results show ARC-C and ARC-Challenge should consolidate
    """
    import difflib

    questionable_pairs = []

    # Find all pairs in the questionable similarity range
    for i, name1 in enumerate(benchmark_names):
        for name2 in benchmark_names[i+1:]:
            # Use difflib.SequenceMatcher for similarity (Python standard library)
            similarity = difflib.SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

            if min_similarity <= similarity <= max_similarity:
                questionable_pairs.append({
                    "benchmark1": name1,
                    "benchmark2": name2,
                    "similarity": similarity
                })

    if not questionable_pairs:
        logger.debug("No questionable pairs found in similarity range "
                    f"{min_similarity:.0%}-{max_similarity:.0%}")
        return []

    logger.info(f"Found {len(questionable_pairs)} questionable pairs for AI review "
                f"(similarity {min_similarity:.0%}-{max_similarity:.0%})")

    # AI review each questionable pair
    ai_reviews = []

    for pair in questionable_pairs:
        name1 = pair["benchmark1"]
        name2 = pair["benchmark2"]
        similarity = pair["similarity"]

        logger.info(f"AI reviewing: '{name1}' vs '{name2}' (similarity: {similarity:.1%})")

        # Step 1: Web search for context (if enabled)
        web_context = ""
        if enable_web_search:
            try:
                # Search for both benchmarks
                search_query = f'"{name1}" benchmark AND "{name2}" benchmark'
                search_results = scrape_google_search(search_query, max_results=3)

                if search_results:
                    web_context = "\n\nWeb search results:\n"
                    for i, result in enumerate(search_results, 1):
                        web_context += f"\n{i}. {result.get('title', '')}\n"
                        web_context += f"   {result.get('snippet', '')}\n"
                else:
                    web_context = "\n\nNo web search results found."
            except Exception as e:
                logger.warning(f"Web search failed for '{name1}' vs '{name2}': {e}")
                web_context = "\n\nWeb search unavailable."

        # Step 2: AI analysis
        prompt = f"""Determine if these two benchmark names refer to the same benchmark or are distinct:

Benchmark 1: "{name1}"
Benchmark 2: "{name2}"

Similarity score: {similarity:.1%}
{web_context}

Are these the same benchmark (just different names/abbreviations) or distinct benchmarks?

Respond with JSON:
{{
  "same_benchmark": true or false,
  "confidence": "high" or "medium" or "low",
  "reasoning": "brief explanation",
  "canonical_name": "preferred name if same_benchmark=true, otherwise null"
}}

Examples of SAME benchmark:
- "ARC-C" and "ARC-Challenge" (abbreviation)
- "BBH" and "BIG-Bench Hard" (acronym)
- "GSM8K" and "GSM-8K" (spacing)

Examples of DISTINCT benchmarks:
- "MMLU" and "MMLU-Pro" (different difficulty)
- "HumanEval" and "HumanEval+" (different test sets)
- "MBPP" and "MBPP++" (different versions)
"""

        try:
            ai_result = call_claude_json(prompt=prompt, max_tokens=1024)

            ai_reviews.append({
                "benchmark1": name1,
                "benchmark2": name2,
                "similarity": similarity,
                "same_benchmark": ai_result.get("same_benchmark", False),
                "confidence": ai_result.get("confidence", "medium"),
                "reasoning": ai_result.get("reasoning", ""),
                "canonical_name": ai_result.get("canonical_name"),
                "web_search_used": bool(web_context and "unavailable" not in web_context)
            })

            decision = "SAME" if ai_result.get("same_benchmark") else "DISTINCT"
            logger.info(f"  AI decision: {decision} (confidence: {ai_result.get('confidence')})")

        except Exception as e:
            logger.error(f"AI review failed for '{name1}' vs '{name2}': {e}")
            # Conservative default: treat as distinct
            ai_reviews.append({
                "benchmark1": name1,
                "benchmark2": name2,
                "similarity": similarity,
                "same_benchmark": False,
                "confidence": "low",
                "reasoning": f"AI review failed: {e}",
                "canonical_name": None,
                "web_search_used": False
            })

    return ai_reviews
