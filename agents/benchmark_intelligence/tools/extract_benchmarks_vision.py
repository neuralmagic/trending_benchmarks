"""
Vision AI-powered benchmark extraction from PDFs and images.

Uses Claude's vision capabilities to extract benchmark tables from
PDF documents, charts, and figures.
"""

import logging
import base64
import io
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from ._claude_client import call_claude, is_anthropic_available

logger = logging.getLogger(__name__)


def _extract_section_structure(pdf) -> List[Dict[str, Any]]:
    """
    Extract section titles and page ranges from PDF.

    Uses heuristics to detect section headings:
    - Lines starting with numbers (e.g., "5. Evaluation", "3.2 Results")
    - Title case lines with limited length
    - Common section patterns

    Args:
        pdf: Open pdfplumber PDF object

    Returns:
        List of sections with title, start_page, end_page:
            [{"title": "5. Evaluation", "start_page": 10, "end_page": 15}, ...]
    """
    sections = []

    for page_num, page in enumerate(pdf.pages, 1):
        page_text = page.extract_text()
        if not page_text:
            continue

        # Split into lines and look for section headings
        lines = page_text.split('\n')
        for line in lines:
            line_stripped = line.strip()

            # Skip empty or very long lines (unlikely to be headings)
            if not line_stripped or len(line_stripped) > 100:
                continue

            # Heuristic 1: Lines starting with section numbers (e.g., "5. Evaluation")
            # Pattern: digit(s), optional dot/space, title case text
            import re
            if re.match(r'^\d+(\.\d+)*[\.\s]+[A-Z]', line_stripped):
                sections.append({
                    "title": line_stripped,
                    "start_page": page_num,
                    "end_page": page_num  # Will be updated when next section found
                })

            # Heuristic 2: Common section keywords in title case
            # (helps catch unnumbered sections like "Results" or "Evaluation")
            elif re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', line_stripped):
                keywords = ['evaluation', 'results', 'experiments', 'benchmarks',
                           'performance', 'analysis', 'conclusion', 'appendix']
                if any(kw in line_stripped.lower() for kw in keywords):
                    sections.append({
                        "title": line_stripped,
                        "start_page": page_num,
                        "end_page": page_num
                    })

    # Update end_page for each section (spans until next section starts)
    for i in range(len(sections) - 1):
        sections[i]["end_page"] = sections[i + 1]["start_page"] - 1

    # Last section extends to end of document
    if sections:
        sections[-1]["end_page"] = len(pdf.pages)

    return sections


def _filter_benchmark_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter sections to those likely containing benchmark data.

    Args:
        sections: List of section dictionaries from _extract_section_structure()

    Returns:
        Filtered list of benchmark-relevant sections
    """
    # Keywords indicating benchmark/evaluation content
    relevant_keywords = [
        'evaluation', 'result', 'benchmark', 'experiment',
        'performance', 'comparison', 'analysis', 'testing',
        'assessment', 'metric', 'score', 'accuracy', 'appendix'
    ]

    filtered = []
    for section in sections:
        title_lower = section['title'].lower()
        if any(keyword in title_lower for keyword in relevant_keywords):
            filtered.append(section)
            logger.debug(f"Selected section: {section['title']} (pages {section['start_page']}-{section['end_page']})")

    return filtered


def _extract_images_from_pdf(pdf_content: bytes) -> List[Dict[str, Any]]:
    """
    Extract embedded images from PDF pages.

    Args:
        pdf_content: PDF file content as bytes

    Returns:
        List of image dictionaries with:
            - image_data: Image bytes
            - page_num: Page number where image was found
            - width: Image width in pixels
            - height: Image height in pixels
    """
    try:
        import pdfplumber
        from PIL import Image

        images = []

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract images from this page
                page_images = page.images

                if not page_images:
                    continue

                logger.debug(f"Page {page_num}: Found {len(page_images)} embedded images")

                for img_idx, img_info in enumerate(page_images):
                    try:
                        # Get image object
                        # pdfplumber provides image metadata, need to extract actual image
                        x0 = img_info.get('x0', 0)
                        y0 = img_info.get('top', 0)
                        x1 = img_info.get('x1', page.width)
                        y1 = img_info.get('bottom', page.height)

                        # Crop the image area from the page
                        cropped = page.crop((x0, y0, x1, y1))
                        img = cropped.to_image(resolution=150)

                        # Convert to bytes
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format='PNG')
                        img_bytes = img_buffer.getvalue()

                        if len(img_bytes) < 1000:  # Skip very small images (likely icons/decorations)
                            logger.debug(f"Page {page_num}, image {img_idx}: Too small ({len(img_bytes)} bytes), skipping")
                            continue

                        images.append({
                            'image_data': img_bytes,
                            'page_num': page_num,
                            'width': int(x1 - x0),
                            'height': int(y1 - y0),
                            'index': img_idx
                        })

                        logger.debug(
                            f"Page {page_num}, image {img_idx}: Extracted "
                            f"({int(x1-x0)}x{int(y1-y0)}, {len(img_bytes)} bytes)"
                        )

                    except Exception as e:
                        logger.warning(f"Page {page_num}, image {img_idx}: Failed to extract: {e}")
                        continue

        logger.info(f"Extracted {len(images)} images from PDF")
        return images

    except ImportError:
        logger.error("pdfplumber not installed. Install with: pip install pdfplumber")
        return []
    except Exception as e:
        logger.error(f"Failed to extract images from PDF: {e}")
        return []


def _extract_from_image_with_vision(image_bytes: bytes, source_name: str, max_tokens: int = 16384) -> Dict[str, Any]:
    """
    Extract benchmarks from image using Claude vision API.

    Args:
        image_bytes: Image content as bytes (PNG/JPEG)
        source_name: Name/identifier of the source
        max_tokens: Maximum tokens in response

    Returns:
        Dictionary with extracted benchmarks and metadata
    """
    try:
        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Determine image type
        image_type = "image/png"
        if image_bytes[:2] == b'\xff\xd8':  # JPEG magic number
            image_type = "image/jpeg"

        # Build vision prompt
        prompt = _build_vision_extraction_prompt()

        # Call Claude with vision
        from ._claude_client import call_claude_vision_json

        result = call_claude_vision_json(
            prompt=prompt,
            image_data=image_b64,
            image_type=image_type,
            max_tokens=max_tokens
        )

        # Validate result
        if not isinstance(result, dict):
            logger.warning("Vision API response is not a dict")
            return {"benchmarks": [], "metadata": {"error": "invalid_response"}}

        if "benchmarks" not in result:
            result["benchmarks"] = []

        return result

    except Exception as e:
        logger.error(f"Vision extraction failed: {e}")
        return {
            "benchmarks": [],
            "metadata": {
                "error": str(e),
                "extraction_method": "vision_failed"
            }
        }


def extract_benchmarks_from_image(
    image_content: bytes,
    source_name: Optional[str] = None,
    max_tokens: int = 16384,
) -> Dict[str, Any]:
    """
    Extract benchmarks from a standalone image (chart, figure, screenshot).

    Used for images embedded in blog posts or documentation.

    Args:
        image_content: Image file content as bytes (PNG, JPEG, etc.)
        source_name: Name/identifier of the source
        max_tokens: Maximum tokens in response

    Returns:
        Dictionary containing extracted benchmark data:
            - benchmarks: List of benchmark entries
            - metadata: Extraction metadata

    Example:
        >>> with open("benchmark_chart.png", "rb") as f:
        ...     img_bytes = f.read()
        >>> result = extract_benchmarks_from_image(img_bytes, source_name="Blog Post Chart")
        >>> print(f"Found {len(result['benchmarks'])} benchmarks")
    """
    try:
        if not image_content or not isinstance(image_content, bytes):
            raise ValueError("image_content must be non-empty bytes")

        if len(image_content) < 100:
            logger.warning("Image content is very small, may not be valid")
            return _empty_result(source_name, "image_too_small")

        logger.info(f"Extracting benchmarks from image ({len(image_content)} bytes)")

        # Extract using vision API
        result = _extract_from_image_with_vision(image_content, source_name or "unknown", max_tokens)

        # Add metadata
        if "metadata" not in result:
            result["metadata"] = {}

        result["metadata"]["document_source"] = source_name or "unknown"
        result["metadata"]["extraction_date"] = datetime.utcnow().isoformat()
        result["metadata"]["total_benchmarks"] = len(result.get("benchmarks", []))
        result["metadata"]["source_type"] = "image_vision"
        result["metadata"]["extraction_method"] = "claude_vision_api"

        logger.info(f"Extracted {len(result['benchmarks'])} benchmarks from image using vision AI")

        return result

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Image vision extraction failed: {e}")
        raise RuntimeError(f"Failed to extract benchmarks from image: {e}")


def extract_benchmarks_from_pdf(
    pdf_content: bytes,
    source_name: Optional[str] = None,
    max_tokens: int = 16384,
    chunk_size: int = 8,
) -> Dict[str, Any]:
    """
    Extract benchmarks from PDF using vision AI with text fallback.

    Strategy (FR-004):
    1. Extract section structure from PDF (titles + page ranges)
    2. Filter to benchmark-relevant sections (Evaluation, Results, etc.)
    3. Extract embedded images (charts, figures, tables) from relevant sections
    4. Send each extracted image to Claude vision API for analysis
    5. If no images found, fall back to text-based extraction
    6. Merge results from all images/chunks

    Args:
        pdf_content: PDF file content as bytes
        source_name: Name/identifier of the source document
        max_tokens: Maximum tokens in response per chunk/image
        chunk_size: Pages per chunk for text fallback (default: 8)

    Returns:
        Dictionary containing extracted benchmark data:
            - benchmarks: List of benchmark entries with scores and context
            - metadata: Extraction metadata (source, date, count, etc.)

    Raises:
        ValueError: If pdf_content is empty or invalid
        RuntimeError: If extraction fails

    Example:
        >>> with open("paper.pdf", "rb") as f:
        ...     pdf_bytes = f.read()
        >>> result = extract_benchmarks_from_pdf(pdf_bytes, source_name="Llama-3.1 Paper")
        >>> print(f"Found {result['metadata']['total_benchmarks']} benchmarks")
    """
    try:
        if not pdf_content or not isinstance(pdf_content, bytes):
            raise ValueError("pdf_content must be non-empty bytes")

        if len(pdf_content) < 100:
            logger.warning("PDF content is very small, may not be valid")
            return _empty_result(source_name, "pdf_too_small")

        logger.info(f"Extracting benchmarks from PDF ({len(pdf_content)} bytes)")

        # Extract text from PDF using pdfplumber with section filtering
        import io
        try:
            import pdfplumber
        except ImportError:
            raise RuntimeError("pdfplumber not installed. Install with: pip install pdfplumber")

        # Initialize metadata tracking variables
        sections = []
        relevant_sections = []
        relevant_pages = []
        total_pages = 0

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            total_pages = len(pdf.pages)
            logger.debug(f"PDF has {total_pages} pages")

            # Pass 1: Extract section structure
            sections = _extract_section_structure(pdf)
            logger.info(f"Detected {len(sections)} sections in PDF")

            # Pass 2: Filter to benchmark-relevant sections
            relevant_sections = _filter_benchmark_sections(sections)

            if not relevant_sections:
                logger.warning("No benchmark-relevant sections found, extracting full PDF")
                # Fallback: extract all pages if no sections detected
                relevant_pages = list(range(1, total_pages + 1))
            else:
                # Extract page numbers from relevant sections
                relevant_pages_set = set()
                for section in relevant_sections:
                    for page_num in range(section['start_page'], section['end_page'] + 1):
                        relevant_pages_set.add(page_num)
                relevant_pages = sorted(relevant_pages_set)

                logger.info(
                    f"Filtered to {len(relevant_sections)} relevant sections "
                    f"({len(relevant_pages)}/{total_pages} pages)"
                )

            # Split relevant pages into chunks for robust processing
            page_chunks = []
            for i in range(0, len(relevant_pages), chunk_size):
                chunk = relevant_pages[i:i + chunk_size]
                page_chunks.append(chunk)

            logger.info(
                f"Split {len(relevant_pages)} pages into {len(page_chunks)} chunks "
                f"(chunk_size={chunk_size})"
            )

        # Check if Claude API is available
        if not is_anthropic_available():
            raise RuntimeError(
                "Anthropic API not available. Set ANTHROPIC_API_KEY environment "
                "variable or install anthropic package (pip install anthropic)"
            )

        # Extract benchmarks from each chunk
        all_benchmarks = []
        chunks_processed = 0
        chunks_failed = 0

        # Always attempt vision extraction first (FR-004)
        vision_successful = False
        logger.info(f"Attempting vision AI extraction - extracting embedded images from PDF")

        # Extract all embedded images from the PDF
        pdf_images = _extract_images_from_pdf(pdf_content)

        if not pdf_images:
            logger.info("No images found in PDF")
        else:
            # Filter images to only those in relevant pages
            relevant_images = [
                img for img in pdf_images
                if img['page_num'] in relevant_pages
            ]

            if not relevant_images:
                logger.info("No images in relevant sections")
            else:
                vision_successful = True
                logger.info(
                    f"Found {len(pdf_images)} total images, "
                    f"{len(relevant_images)} in relevant sections"
                )

                # Process each image with vision API
                for img_idx, img_info in enumerate(relevant_images, 1):
                    try:
                        page_num = img_info['page_num']
                        img_bytes = img_info['image_data']

                        logger.debug(
                            f"Image {img_idx}/{len(relevant_images)}: "
                            f"Analyzing page {page_num} image "
                            f"({img_info['width']}x{img_info['height']}, {len(img_bytes)} bytes)"
                        )

                        # Extract benchmarks from image using vision API
                        img_result = _extract_from_image_with_vision(
                            img_bytes,
                            f"{source_name} - Page {page_num} Image {img_info['index']}",
                            max_tokens
                        )

                        img_benchmarks = img_result.get("benchmarks", [])
                        all_benchmarks.extend(img_benchmarks)

                        if img_benchmarks:
                            logger.info(
                                f"Image {img_idx}/{len(relevant_images)}: "
                                f"Extracted {len(img_benchmarks)} benchmarks from page {page_num}"
                            )

                    except Exception as e:
                        logger.error(f"Image {img_idx}/{len(relevant_images)} failed: {e}")
                        chunks_failed += 1
                        # Continue processing other images

                chunks_processed = len(relevant_images) - chunks_failed
                logger.info(
                    f"Vision extraction complete: Analyzed {chunks_processed}/{len(relevant_images)} images"
                )

        # Always perform text extraction (in addition to vision extraction if images were found)
        # Text extraction catches benchmarks in prose that vision might miss
        logger.info(f"Performing text-based extraction for {len(page_chunks)} chunks")

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                for chunk_idx, chunk_pages in enumerate(page_chunks, 1):
                    try:
                        # Extract text from this chunk
                        chunk_text = ""
                        for page_num in chunk_pages:
                            page = pdf.pages[page_num - 1]  # pdfplumber uses 0-based indexing
                            page_text = page.extract_text()
                            if page_text:
                                chunk_text += f"\n\n=== Page {page_num} ===\n\n{page_text}"

                        if not chunk_text or len(chunk_text) < 100:
                            logger.warning(f"Chunk {chunk_idx}/{len(page_chunks)}: No text extracted, skipping")
                            chunks_failed += 1
                            continue

                        logger.info(
                            f"Chunk {chunk_idx}/{len(page_chunks)}: Extracted {len(chunk_text)} chars "
                            f"from pages {chunk_pages[0]}-{chunk_pages[-1]}"
                        )

                        # Build extraction prompt for this chunk
                        prompt = _build_text_extraction_prompt(chunk_text)

                        # Use standard text extraction
                        from ._claude_client import call_claude_json

                        logger.debug(f"Chunk {chunk_idx}/{len(page_chunks)}: Calling Claude")

                        chunk_result = call_claude_json(
                            prompt=prompt,
                            max_tokens=max_tokens
                        )

                        # Merge benchmarks from this chunk
                        chunk_benchmarks = chunk_result.get("benchmarks", [])
                        all_benchmarks.extend(chunk_benchmarks)
                        chunks_processed += 1

                        logger.info(
                            f"Chunk {chunk_idx}/{len(page_chunks)}: Extracted {len(chunk_benchmarks)} benchmarks"
                        )

                    except Exception as e:
                        logger.error(f"Chunk {chunk_idx}/{len(page_chunks)} failed: {e}")
                        chunks_failed += 1
                        # Continue processing other chunks

        if not all_benchmarks and chunks_processed == 0:
            logger.warning("No benchmarks extracted from any chunk")
            return _empty_result(source_name, "no_benchmarks_extracted")

        # Build consolidated result
        result = {"benchmarks": all_benchmarks}
        logger.info(
            f"Chunked extraction complete: {len(all_benchmarks)} benchmarks from "
            f"{chunks_processed}/{len(page_chunks)} chunks ({chunks_failed} failed)"
        )

        # result is already a dict from call_claude_json
        # Validate and enhance result
        if not isinstance(result, dict):
            logger.warning("Response is not a dict, creating empty result")
            result = {"benchmarks": []}

        if "benchmarks" not in result:
            logger.warning("No benchmarks key in response, creating empty result")
            result = {"benchmarks": []}

        # Ensure metadata exists
        if "metadata" not in result:
            result["metadata"] = {}

        result["metadata"]["document_source"] = source_name or "unknown"
        result["metadata"]["extraction_date"] = datetime.utcnow().isoformat()
        result["metadata"]["total_benchmarks"] = len(result.get("benchmarks", []))
        result["metadata"]["source_type"] = "pdf_vision_and_text" if vision_successful else "pdf_text"
        result["metadata"]["extraction_method"] = (
            "claude_vision_api_from_images_plus_text_extraction" if vision_successful
            else "claude_text_extraction_with_section_filtering_and_chunking"
        )
        result["metadata"]["vision_used"] = vision_successful
        result["metadata"]["text_extraction_used"] = True
        result["metadata"]["total_pages"] = total_pages
        result["metadata"]["pages_processed"] = len(relevant_pages)
        result["metadata"]["sections_found"] = len(sections)
        result["metadata"]["sections_used"] = len(relevant_sections)

        if vision_successful:
            result["metadata"]["images_total"] = len(pdf_images) if 'pdf_images' in locals() else 0
            result["metadata"]["images_relevant"] = len(relevant_images) if 'relevant_images' in locals() else 0
            result["metadata"]["images_processed"] = chunks_processed
            result["metadata"]["images_failed"] = chunks_failed
        else:
            result["metadata"]["chunks_total"] = len(page_chunks) if 'page_chunks' in locals() else 0
            result["metadata"]["chunks_processed"] = chunks_processed
            result["metadata"]["chunks_failed"] = chunks_failed
            result["metadata"]["chunk_size"] = chunk_size

        extraction_mode = "vision AI (embedded images) + text extraction" if vision_successful else "text-based AI extraction"
        logger.info(
            f"Extracted {len(result['benchmarks'])} benchmarks from PDF using {extraction_mode} "
            f"({len(relevant_pages)}/{total_pages} pages)"
        )

        return result

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"PDF vision extraction failed: {e}")
        raise RuntimeError(f"Failed to extract benchmarks from PDF: {e}")


def _build_vision_extraction_prompt() -> str:
    """
    Build the prompt for vision-based extraction from images/PDFs.

    Returns:
        Prompt string for Claude vision API
    """
    return """Extract all benchmark evaluation results from this image.

This image may contain:
- Benchmark comparison tables
- Performance charts/graphs
- Evaluation results
- Score listings

Extract all benchmarks you can identify from the image.

For each benchmark, extract:
{
  "name": "benchmark name (e.g., MMLU, GSM8K, HumanEval)",
  "score": numeric_value_or_null,
  "metric": "accuracy|pass@1|f1|wer|etc",
  "context": {
    "shot_count": number_or_null,
    "subset": "specific variant if any (e.g., challenge, easy)",
    "special_conditions": "CoT|PoT|base|instruct|with_subtitles|etc"
  },
  "source_location": "description of where in image (e.g., Table 2, Bar Chart)"
}

Return ONLY valid JSON in this exact format:
{
  "benchmarks": [ ...array of benchmark objects... ],
  "metadata": {
    "total_benchmarks": number,
    "extraction_confidence": "high|medium|low",
    "image_contains": "table|chart|both|text"
  }
}

Important:
- Extract ALL benchmarks visible in the image
- Include shot count if shown (5-shot, 0-shot, etc.)
- Include variant details (CoT, subset names, etc.)
- If a score is unclear or not shown, use null
- Look carefully at charts - extract benchmark names from axes, legends, and labels
- Return ONLY the JSON object, no other text"""


def _build_text_extraction_prompt(pdf_text: str) -> str:
    """
    Build the prompt for text-based extraction from PDF.

    Args:
        pdf_text: Extracted text from PDF

    Returns:
        Prompt string with embedded PDF text
    """
    # Build prompt without f-string to avoid format errors
    prompt_template = """Extract all benchmark evaluation results from this research paper text.

The text below is extracted from a PDF research paper. Find all benchmark evaluation results.

{pdf_text}

---

Instructions:
Extract all benchmark names, scores, and evaluation contexts from the text above.

Focus on finding benchmark names, scores, and contexts from tables and text.

For each benchmark, extract:
{{
  "name": "benchmark name",
  "score": numeric_value,
  "metric": "accuracy|pass@1|exact_match|etc",
  "context": {{
    "shot_count": number or null,
    "subset": "specific variant if any",
    "special_conditions": "CoT|PoT|base|instruct|etc"
  }},
  "source_location": "Table X or Page Y"
}}

Return JSON:
{{
  "benchmarks": [ ...array of benchmarks... ],
  "metadata": {{
    "total_benchmarks": number,
    "extraction_confidence": "high|medium|low"
  }}
}}

Important:
- Extract ALL benchmarks found
- Include shot count if mentioned (5-shot, 0-shot, etc.)
- Include variant info (CoT, base, instruct, etc.)
- If unsure about score, omit it
- Return only JSON"""

    return prompt_template.format(pdf_text=pdf_text)


def _detect_environment() -> str:
    """Detect Claude environment (ambient or standard)."""
    import os

    ambient_indicators = [
        os.getenv("AMBIENT_SESSION_ID"),
        os.getenv("AMBIENT_WORKSPACE_ID"),
        os.getenv("CLAUDECODE") == "1",
    ]
    if any(ambient_indicators):
        return "ambient"
    return "standard"


def _empty_result(source_name: Optional[str], reason: str) -> Dict[str, Any]:
    """Create empty result with error reason."""
    return {
        "benchmarks": [],
        "metadata": {
            "document_source": source_name or "unknown",
            "extraction_date": datetime.utcnow().isoformat(),
            "total_benchmarks": 0,
            "source_type": "pdf_vision",
            "error": reason
        }
    }
