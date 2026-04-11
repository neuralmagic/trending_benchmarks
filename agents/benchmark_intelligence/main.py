"""
Main orchestrator for the Benchmark Intelligence Agent.

This module provides the main agent that coordinates the full workflow:
1. Discover trending models from configured labs
2. Process each model (parse card, extract benchmarks, fetch docs, classify)
3. Store results in cache
4. Generate reports

Usage:
    python -m agents.benchmark_intelligence.main [OPTIONS]
"""

import logging
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import yaml

from .tools.discover_models import discover_trending_models
from .tools.parse_model_card import parse_model_card
from .tools.extract_benchmarks import (
    extract_benchmarks_from_text,
    extract_benchmarks_from_multiple_sources,
    aggregate_benchmark_results,
)
from .tools.fetch_docs import fetch_documentation
from .tools.parallel_fetcher import fetch_documents_parallel, prepare_document_specs_for_model
from .tools.consolidate import (
    consolidate_benchmarks,
    extract_benchmark_names,
    apply_consolidation,
)
from .tools.classify import classify_benchmarks_batch, enrich_benchmarks_with_classification
from .tools.cache import CacheManager
from .tools.taxonomy_manager import (
    load_current_taxonomy,
    analyze_benchmark_fit,
    propose_new_categories,
    evolve_taxonomy,
    archive_taxonomy_if_changed,
    update_taxonomy_file,
)
from .reporting import ReportGenerator
from .clients.factory import get_hf_client


# Progress reporting symbols
SYMBOLS = {"success": "✓", "error": "✗", "cached": "↻", "new": "⊕"}


logger = logging.getLogger(__name__)


class BenchmarkIntelligenceAgent:
    """
    Main orchestrator for benchmark intelligence workflow.

    Coordinates model discovery, benchmark extraction, classification,
    and reporting across configured labs.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        cache_path: Optional[str] = None,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the benchmark intelligence agent.

        Args:
            config_path: Path to configuration YAML file (default: config.yaml at project root)
            cache_path: Path to cache database (default: benchmark_cache.db)
            dry_run: If True, don't write to cache or files
            verbose: If True, enable verbose logging
        """
        self.dry_run = dry_run
        self.verbose = verbose

        # Setup logging
        self._setup_logging()

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)

        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize cache manager
        if cache_path is None:
            cache_path = "benchmark_cache.db"

        if self.dry_run:
            logger.info("DRY RUN MODE: No cache or file writes will occur")
            self.cache = None
        else:
            logger.info(f"Initializing cache at {cache_path}")
            self.cache = CacheManager(cache_path)

        # Initialize clients
        self.hf_client = get_hf_client()

        # Statistics
        self.stats = {
            "models_discovered": 0,
            "models_processed": 0,
            "models_skipped": 0,
            "models_failed": 0,
            "benchmarks_extracted": 0,
            "documents_fetched": 0,
            "errors": [],
        }

    def _setup_logging(self):
        """Configure logging based on verbosity."""
        level = logging.DEBUG if self.verbose else logging.INFO

        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        # Reduce noise from other libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    def run(self, incremental: bool = True, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Run the full benchmark intelligence workflow.

        Args:
            incremental: If True, only process new/changed models
            force_reprocess: If True, reprocess all models (ignores cache)

        Returns:
            Dictionary with run statistics and results

        Workflow:
            1. Load configuration
            2. Discover trending models (from configured labs)
            3. For each model:
                a. Check cache - skip if model card unchanged
                b. Parse model card
                c. Extract benchmarks from card
                d. Fetch related documentation (blogs, papers)
                e. Extract benchmarks from documentation
                f. Consolidate all benchmarks
                g. Classify benchmarks
                h. Store in cache
            4. Create snapshot
            5. Generate report
        """
        try:
            logger.info("=" * 80)
            logger.info("Starting Benchmark Intelligence Agent")
            logger.info("=" * 80)

            # Step 1: Discover trending models
            logger.info("\n[Discovery] Starting model discovery...")
            models = self._discover_models()
            self.stats["models_discovered"] = len(models)

            # Get lab names for reporting
            labs = self.config.get("labs", [])
            lab_preview = ', '.join(labs[:3]) + ('...' if len(labs) > 3 else '')
            logger.info(f"[Discovery] Found {len(models)} models from {lab_preview}")

            if not models:
                logger.warning("No models discovered. Exiting.")
                return self._create_result(success=False, message="No models discovered")

            # Step 2: Process each model
            logger.info(f"\n[Processing] Processing {len(models)} models...")
            for i, model in enumerate(models, 1):
                try:
                    logger.info(f"[Processing] Model {i}/{len(models)}: {model['id']}")

                    # Check if we should process this model
                    if incremental and not force_reprocess:
                        if self._should_skip_model(model):
                            logger.info(f"  {SYMBOLS['cached']} Cached (no changes)")
                            self.stats["models_skipped"] += 1
                            continue

                    # Process the model
                    self._process_model(model)
                    self.stats["models_processed"] += 1

                except Exception as e:
                    logger.error(f"Failed to process model {model.get('id', 'unknown')}: {e}")
                    self.stats["models_failed"] += 1
                    self.stats["errors"].append({
                        "model_id": model.get("id"),
                        "error": str(e),
                    })
                    # Continue with next model
                    continue

            # Step 3: Consolidate benchmarks across all models
            logger.info("\n[Consolidation] Consolidating benchmarks...")
            self._consolidate_all_benchmarks()

            # Step 4: Create snapshot
            logger.info("\n[Step 4/5] Creating snapshot...")
            snapshot_id = self._create_snapshot()

            # Step 5: Generate report
            logger.info("\n[Reporting] Generating 7 sections...")
            report = self._generate_report()

            # Summary
            logger.info("\n" + "=" * 80)
            logger.info("Benchmark Intelligence Agent - Run Complete")
            logger.info("=" * 80)
            logger.info(f"Models discovered: {self.stats['models_discovered']}")
            logger.info(f"Models processed:  {self.stats['models_processed']}")
            logger.info(f"Models skipped:    {self.stats['models_skipped']}")
            logger.info(f"Models failed:     {self.stats['models_failed']}")
            logger.info(f"Benchmarks extracted: {self.stats['benchmarks_extracted']}")
            logger.info(f"Documents fetched: {self.stats['documents_fetched']}")
            if self.stats["errors"]:
                logger.warning(f"Errors encountered: {len(self.stats['errors'])}")
            logger.info("=" * 80)

            return self._create_result(
                success=True,
                snapshot_id=snapshot_id,
                report=report,
            )

        except Exception as e:
            logger.error(f"Fatal error in agent run: {e}", exc_info=True)
            return self._create_result(
                success=False,
                message=f"Fatal error: {e}",
            )

    def _discover_models(self) -> List[Dict[str, Any]]:
        """Discover trending models from configured labs."""
        labs = self.config.get("labs", [])
        discovery_config = self.config.get("discovery", {})

        logger.info(f"[Discovery] Querying {len(labs)} labs...")
        logger.debug(f"Labs: {', '.join(labs)}")

        return discover_trending_models(
            labs=labs,
            config=discovery_config,
            hf_client=self.hf_client,
        )

    def _should_skip_model(self, model: Dict[str, Any]) -> bool:
        """
        Check if model should be skipped (no changes).

        Args:
            model: Model information dictionary

        Returns:
            True if should skip, False if should process
        """
        if self.dry_run or self.cache is None:
            return False

        model_id = model.get("id")
        if not model_id:
            return False

        # Get cached model
        cached_model = self.cache.get_model(model_id)
        if not cached_model:
            # New model, don't skip
            return False

        # Check if model card has changed
        model_card = model.get("model_card")
        if model_card:
            # Compute hash
            import hashlib
            new_hash = hashlib.sha256(model_card.encode('utf-8')).hexdigest()
            cached_hash = cached_model.get("model_card_hash")

            if new_hash == cached_hash:
                # No changes
                return True

        return False

    def _process_model(self, model: Dict[str, Any]):
        """
        Process a single model through the full pipeline.

        Args:
            model: Model information dictionary
        """
        model_id = model.get("id")
        if not model_id:
            raise ValueError("Model missing 'id' field")

        # Step 2a: Parse model card
        model_card_data = parse_model_card(model_id, hf_client=self.hf_client)
        logger.info(f"  {SYMBOLS['success']} Fetched model card")

        # Step 2b: Extract benchmarks from model card
        card_benchmarks = extract_benchmarks_from_text(
            text=model_card_data["content"],
            source_type="model_card",
            source_name=model_id,
        )

        # Tag benchmarks with source_type for tracking
        for bench in card_benchmarks.get("benchmarks", []):
            bench["source_type"] = "model_card"

        # Step 2c: Fetch related documentation in parallel
        logger.info(f"Fetching related documentation...")
        try:
            docs = self._fetch_documents_parallel(model_id, model, model_card_data)
            if docs:
                logger.info(f"  {SYMBOLS['success']} Fetched {len(docs)} documents in parallel")
        except Exception as e:
            logger.warning(f"Failed to fetch documentation: {e}")
            docs = []

        # Step 2d: Extract benchmarks from documentation
        doc_benchmarks = []
        if docs:
            logger.info(f"Extracting benchmarks from {len(docs)} documents...")
            sources = [
                {
                    "text": doc["content"],
                    "source_type": doc["doc_type"],
                    "source_name": doc["title"],
                }
                for doc in docs if doc.get("content")
            ]

            if sources:
                doc_extraction_results = extract_benchmarks_from_multiple_sources(sources)
                doc_benchmarks_agg = aggregate_benchmark_results(doc_extraction_results)
                doc_benchmarks = doc_benchmarks_agg.get("benchmarks", [])

                # Tag benchmarks with source_type from their respective documents
                for i, source in enumerate(sources):
                    if i < len(doc_extraction_results):
                        for bench in doc_extraction_results[i].get("benchmarks", []):
                            bench["source_type"] = source.get("source_type", "unknown")

        # Step 2e: Consolidate benchmarks
        all_benchmarks = card_benchmarks.get("benchmarks", []) + doc_benchmarks
        if all_benchmarks:
            logger.info(f"  {SYMBOLS['success']} Extracted {len(all_benchmarks)} benchmarks")
        self.stats["benchmarks_extracted"] += len(all_benchmarks)

        if all_benchmarks:
            # Get unique benchmark names
            benchmark_names = extract_benchmark_names(all_benchmarks)

            if benchmark_names:
                logger.info(f"Consolidating {len(benchmark_names)} unique benchmark names...")
                consolidation_result = consolidate_benchmarks(benchmark_names)

                # Apply consolidation
                all_benchmarks = apply_consolidation(
                    all_benchmarks,
                    consolidation_result,
                    add_canonical_field=True,
                )

        # Step 2f: Classify benchmarks
        if all_benchmarks:
            logger.info(f"Classifying benchmarks...")
            # Get unique canonical names for classification
            unique_benchmarks = {}
            for bench in all_benchmarks:
                canonical_name = bench.get("canonical_name", bench.get("name"))
                if canonical_name and canonical_name not in unique_benchmarks:
                    unique_benchmarks[canonical_name] = {
                        "name": canonical_name,
                        "description": bench.get("description"),
                    }

            if unique_benchmarks:
                classification_input = list(unique_benchmarks.values())
                classifications = classify_benchmarks_batch(classification_input)

                # Enrich benchmarks with classifications
                all_benchmarks = enrich_benchmarks_with_classification(
                    all_benchmarks,
                    classifications,
                )

        # Step 2g: Store in cache
        if not self.dry_run and self.cache:
            logger.info(f"Storing results in cache...")
            self._store_model_in_cache(model, model_card_data, all_benchmarks, docs)
        else:
            logger.debug("Skipping cache storage (dry run mode)")

    def _store_model_in_cache(
        self,
        model: Dict[str, Any],
        model_card_data: Dict[str, Any],
        benchmarks: List[Dict[str, Any]],
        docs: List[Dict[str, Any]],
    ):
        """Store model and benchmark data in cache."""
        model_id = model.get("id")

        # Add model to cache
        model_info = {
            "id": model_id,
            "name": model.get("name", model_id),
            "lab": model.get("author"),
            "release_date": model.get("created_at"),
            "downloads": model.get("downloads", 0),
            "likes": model.get("likes", 0),
            "tags": model.get("tags", []),
            "model_card": model_card_data.get("content", ""),
        }
        self.cache.add_model(model_info)

        # Add benchmarks
        for bench in benchmarks:
            canonical_name = bench.get("canonical_name", bench.get("name"))
            if not canonical_name:
                continue

            # Add benchmark to cache
            benchmark_id = self.cache.add_benchmark(
                name=canonical_name,
                categories=bench.get("categories", []),
                attributes={
                    "modality": bench.get("modality"),
                    "domain": bench.get("domain"),
                    "difficulty_level": bench.get("difficulty_level"),
                },
            )

            # Link model to benchmark
            self.cache.add_model_benchmark(
                model_id=model_id,
                benchmark_id=benchmark_id,
                score=bench.get("score"),
                context=bench.get("context", {}),
                source_type=bench.get("source_type", "unknown"),
                source_url=bench.get("source_url"),
            )

        # Add documents
        for doc in docs:
            if doc.get("content"):
                self.cache.add_document(
                    model_id=model_id,
                    doc_type=doc.get("doc_type", "unknown"),
                    url=doc.get("url", ""),
                    content=doc["content"],
                )

        self.stats["documents_fetched"] += len(docs)

    def _fetch_documents_parallel(
        self,
        model_id: str,
        model: Dict[str, Any],
        model_card_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Fetch all documents for a model in parallel.

        Args:
            model_id: Model identifier
            model: Model metadata dictionary
            model_card_data: Parsed model card data

        Returns:
            List of fetched document dictionaries
        """
        # Note: Parallel fetching interface mismatch - using sequential fetch for now
        # TODO: Implement proper parallel document fetching
        logger.debug("Using sequential fetch for documents")
        return self._fetch_documents_sequential(model_id, model, model_card_data)

    def _fetch_documents_sequential(
        self,
        model_id: str,
        model: Dict[str, Any],
        model_card_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Fetch documents sequentially (fallback method).

        Args:
            model_id: Model identifier
            model: Model metadata dictionary
            model_card_data: Parsed model card data

        Returns:
            List of fetched document dictionaries
        """
        try:
            from .tools.fetch_docs_enhanced import fetch_all_documentation

            model_name = model_id.split("/")[-1] if "/" in model_id else model_id
            lab_name = model.get("author", model_id.split("/")[0] if "/" in model_id else "")

            docs = fetch_all_documentation(
                model_id=model_id,
                model_name=model_name,
                lab_name=lab_name,
                config=self.config,
            )
            # Filter out the model card since we already have it
            return [doc for doc in docs if doc.get("doc_type") != "model_card"]
        except Exception as e:
            logger.warning(f"Sequential fetch failed: {e}")
            return []

    def _consolidate_all_benchmarks(self):
        """Consolidate benchmarks across all models in cache."""
        if self.dry_run or self.cache is None:
            logger.debug("Skipping consolidation (dry run mode or no cache)")
            return

        # Get all benchmarks
        all_benchmarks = self.cache.get_all_benchmarks()
        logger.info(f"[Consolidation] Found {len(all_benchmarks)} unique benchmark names")

        # Already consolidated during processing
        # This step is for any additional cross-model consolidation if needed
        logger.debug("Benchmarks already consolidated during model processing")

        # Taxonomy evolution
        logger.info("Evolving taxonomy based on discovered benchmarks...")
        self._evolve_taxonomy(all_benchmarks)

    def _evolve_taxonomy(self, all_benchmarks: List[Dict[str, Any]]):
        """
        Evolve taxonomy based on discovered benchmarks.

        Args:
            all_benchmarks: List of all benchmarks from cache
        """
        if self.dry_run:
            logger.debug("Skipping taxonomy evolution (dry run mode)")
            return

        try:
            # Get taxonomy file path (at project root)
            taxonomy_path = Path(__file__).parent.parent.parent / "benchmark_taxonomy.md"

            # Load current taxonomy
            logger.info("Loading current taxonomy...")
            current_taxonomy = load_current_taxonomy(str(taxonomy_path))

            # Extract unique benchmark names
            benchmark_names = []
            for bench in all_benchmarks:
                name = bench.get("canonical_name") or bench.get("name")
                if name and name not in benchmark_names:
                    benchmark_names.append(name)

            if not benchmark_names:
                logger.info("No benchmarks to analyze for taxonomy evolution")
                return

            logger.info(f"Analyzing {len(benchmark_names)} unique benchmarks against taxonomy...")

            # Analyze benchmark fit
            analysis = analyze_benchmark_fit(benchmark_names, current_taxonomy)

            poor_fit_count = len(analysis.get("poor_fit", []))
            logger.info(
                f"Fit analysis: {len(analysis.get('well_categorized', []))} well-categorized, "
                f"{poor_fit_count} poor fit"
            )

            # If there are poorly-fitting benchmarks, propose new categories
            if analysis.get("poor_fit"):
                logger.info("Proposing new categories for poor-fit benchmarks...")
                proposed_categories = propose_new_categories(
                    analysis["poor_fit"],
                    current_taxonomy
                )

                if proposed_categories:
                    logger.info(f"Proposed {len(proposed_categories)} new categories: {', '.join(proposed_categories)}")

                    # Evolve taxonomy
                    evolved_taxonomy = evolve_taxonomy(current_taxonomy, proposed_categories)

                    # Archive old taxonomy if changed
                    timestamp = datetime.utcnow().strftime("%Y%m%d")
                    archive_path = archive_taxonomy_if_changed(
                        current_taxonomy,
                        evolved_taxonomy,
                        timestamp
                    )

                    if archive_path:
                        logger.info(f"Archived previous taxonomy to {archive_path}")

                    # Update taxonomy file
                    update_taxonomy_file(evolved_taxonomy, str(taxonomy_path))
                    logger.info("Taxonomy updated successfully")

                    # Update snapshot metadata with taxonomy version
                    if self.cache:
                        self.stats["taxonomy_updated"] = True
                        self.stats["new_categories"] = proposed_categories
                else:
                    logger.info("No new categories needed")
            else:
                logger.info("All benchmarks fit well into existing taxonomy")

        except Exception as e:
            logger.error(f"Taxonomy evolution failed: {e}", exc_info=True)
            # Don't fail the entire run if taxonomy evolution fails

    def _create_snapshot(self) -> Optional[int]:
        """Create a snapshot of current cache state with temporal tracking."""
        if self.dry_run or self.cache is None:
            logger.debug("Skipping snapshot creation (dry run mode or no cache)")
            return None

        stats = self.cache.get_stats()

        # Add taxonomy version if updated
        taxonomy_version = None
        if self.stats.get("taxonomy_updated"):
            timestamp = datetime.utcnow().strftime("%Y%m%d")
            taxonomy_version = f"benchmark_taxonomy_{timestamp}.md"

        summary = {
            "run_stats": self.stats,
            "cache_stats": stats,
            "timestamp": datetime.utcnow().isoformat(),
            "taxonomy_version": taxonomy_version,
        }

        # Use create_snapshot_with_window for temporal tracking (T051-T055, T058)
        snapshot_id = self.cache.create_snapshot_with_window(
            window_months=12,
            taxonomy_version=taxonomy_version,
            summary=summary
        )
        logger.info(f"Created snapshot #{snapshot_id} with 12-month window")

        return snapshot_id

    def _generate_report(self) -> Optional[str]:
        """Generate and save report."""
        if self.dry_run or self.cache is None:
            logger.debug("Skipping report generation (dry run mode or no cache)")
            return None

        # Get retry configuration from config
        retry_config = self.config.get("retry_policy", {})

        report_generator = ReportGenerator(self.cache, retry_config=retry_config)
        report_content = report_generator.generate_report()

        if not self.dry_run:
            # Save historical snapshot first to get the path
            report_path = report_generator.save_snapshot(report_content)
            logger.info(f"[Reporting] {SYMBOLS['success']} Report saved")

            # Update root README with link to the saved report
            report_generator.update_readme(report_path)
            logger.info(f"[Reporting] {SYMBOLS['success']} Updated root README.md")

        return report_content

    def _create_result(
        self,
        success: bool,
        message: str = "",
        snapshot_id: Optional[int] = None,
        report: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create result dictionary."""
        return {
            "success": success,
            "message": message,
            "snapshot_id": snapshot_id,
            "report_generated": report is not None,
            "stats": self.stats,
            "timestamp": datetime.utcnow().isoformat(),
        }


def main():
    """Command-line interface for the Benchmark Intelligence Agent."""
    parser = argparse.ArgumentParser(
        description="Benchmark Intelligence Agent - Track and analyze AI model benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow (incremental)
  python -m agents.benchmark_intelligence.main

  # Force reprocess all models
  python -m agents.benchmark_intelligence.main --force

  # Dry run (don't write to cache/files)
  python -m agents.benchmark_intelligence.main --dry-run

  # Verbose output
  python -m agents.benchmark_intelligence.main --verbose

  # Custom config and cache paths
  python -m agents.benchmark_intelligence.main \\
    --config /path/to/config.yaml \\
    --cache /path/to/cache.db
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file (default: config.yaml at project root)",
    )

    parser.add_argument(
        "--cache",
        type=str,
        help="Path to cache database (default: benchmark_cache.db)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - don't write to cache or files",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess all models (ignore cache)",
    )

    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable incremental mode (process all models)",
    )

    args = parser.parse_args()

    # Create and run agent
    agent = BenchmarkIntelligenceAgent(
        config_path=args.config,
        cache_path=args.cache,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    result = agent.run(
        incremental=not args.no_incremental,
        force_reprocess=args.force,
    )

    # Exit with appropriate code
    if result["success"]:
        logger.info("Agent run completed successfully")
        sys.exit(0)
    else:
        logger.error(f"Agent run failed: {result.get('message', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
