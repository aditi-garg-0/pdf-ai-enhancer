

from __future__ import annotations

import re
import io
import time
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from loguru import logger

# OCR engines (lazy imports with graceful fallback)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

# Text correction
try:
    import ftfy
    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False

try:
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False

try:
    import language_tool_python
    LANGUAGETOOL_AVAILABLE = True
except ImportError:
    LANGUAGETOOL_AVAILABLE = False

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from config import Config, OCRConfig


# ──────────────────────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    text:       str
    confidence: float       # 0.0 – 1.0
    engine:     str
    word_boxes: list[dict]  # [{text, conf, x, y, w, h}, ...]


@dataclass
class CorrectedText:
    raw_ocr:            str
    corrected:          str
    changes_made:       list[str]   # Log of corrections applied
    final_confidence:   float


# ──────────────────────────────────────────────────────────────
#  OCR Corrector
# ──────────────────────────────────────────────────────────────

class OCRCorrector:

    def __init__(self, config: Config):
        self.cfg: OCRConfig = config.ocr
        self.api_cfg        = config.api

        # Lazy-loaded
        self._paddle   = None
        self._easyocr  = None
        self._symspell = None
        self._lt_tool  = None
        self._anthropic = None

    # ── Public API ────────────────────────────────────────────

    def run_ocr(self, image: np.ndarray) -> OCRResult:
        """
        Run all configured OCR engines and ensemble the results.
        """
        results: list[OCRResult] = []

        for engine in self.cfg.engine_priority:
            try:
                result = self._run_engine(engine, image)
                if result and result.text.strip():
                    results.append(result)
            except Exception as e:
                logger.warning(f"  OCR engine '{engine}' failed: {e}")

        if not results:
            return OCRResult(text="", confidence=0.0, engine="none", word_boxes=[])

        return self._ensemble(results)

    def correct(self, ocr_result: OCRResult, context_hint: str = "") -> CorrectedText:
        """
        Full correction pipeline for raw OCR output.
        """
        text   = ocr_result.text
        changes: list[str] = []

        if not text.strip():
            return CorrectedText(
                raw_ocr="", corrected="", changes_made=[], final_confidence=0.0
            )

        # 1. Fix encoding artifacts
        if FTFY_AVAILABLE and self.cfg.enable_unicode_fix:
            fixed = ftfy.fix_text(text)
            if fixed != text:
                changes.append("unicode_fix")
                text = fixed

        # 2. Clean layout artifacts (double spaces, stray hyphens, etc.)
        text = self._clean_layout_artifacts(text)
        changes.append("layout_clean")

        # 3. SymSpell spelling correction
        if SYMSPELL_AVAILABLE and self.cfg.enable_spell_correction:
            corrected_spell = self._symspell_correct(text)
            if corrected_spell != text:
                changes.append("spell_correct")
                text = corrected_spell

        # 4. LanguageTool grammar
        if LANGUAGETOOL_AVAILABLE and self.cfg.enable_grammar_fix:
            corrected_grammar = self._grammar_correct(text)
            if corrected_grammar != text:
                changes.append("grammar_fix")
                text = corrected_grammar

        # 5. Claude AI correction (handles domain terms, equations, etc.)
        if self.cfg.enable_ai_correction and self.api_cfg.anthropic_api_key:
            ai_corrected = self._claude_correct(text, context_hint)
            if ai_corrected and ai_corrected != text:
                changes.append("ai_correction")
                text = ai_corrected

        return CorrectedText(
            raw_ocr          = ocr_result.text,
            corrected        = text,
            changes_made     = changes,
            final_confidence = ocr_result.confidence,
        )

    # ── OCR Engines ───────────────────────────────────────────

    def _run_engine(self, engine: str, image: np.ndarray) -> Optional[OCRResult]:
        if engine == "tesseract":
            return self._tesseract(image)
        if engine == "easyocr":
            return self._easyocr_run(image)
        if engine == "paddleocr":
            return self._paddleocr_run(image)
        if engine == "surya":
            return self._surya_run(image)
        return None

    def _tesseract(self, image: np.ndarray) -> OCRResult:
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract not installed")

        config_str = (
            f"--oem {self.cfg.tesseract_oem} "
            f"--psm {self.cfg.tesseract_psm} "
            f"--dpi {self.cfg.tesseract_dpi}"
        )

        pil_img = Image.fromarray(image)

        # Full data with confidence per word
        data = pytesseract.image_to_data(
            pil_img,
            lang         = self.cfg.tesseract_lang,
            config       = config_str,
            output_type  = pytesseract.Output.DICT,
        )

        words, boxes, confs = [], [], []
        for i, word in enumerate(data["text"]):
            conf = int(data["conf"][i])
            if conf < 0 or not word.strip():
                continue
            if conf / 100 >= self.cfg.min_confidence:
                words.append(word)
                confs.append(conf / 100)
                boxes.append({
                    "text": word, "conf": conf / 100,
                    "x": data["left"][i],  "y": data["top"][i],
                    "w": data["width"][i], "h": data["height"][i],
                })

        full_text = pytesseract.image_to_string(
            pil_img, lang=self.cfg.tesseract_lang, config=config_str
        )
        avg_conf = float(np.mean(confs)) if confs else 0.5

        return OCRResult(
            text       = full_text,
            confidence = avg_conf,
            engine     = "tesseract",
            word_boxes = boxes,
        )

    def _easyocr_run(self, image: np.ndarray) -> OCRResult:
        if not EASYOCR_AVAILABLE:
            raise ImportError("easyocr not installed")

        reader = self._get_easyocr()
        results = reader.readtext(image, detail=self.cfg.easyocr_detail)

        words, boxes, confs = [], [], []
        for item in results:
            bbox, text, conf = item
            if conf >= self.cfg.min_confidence:
                words.append(text)
                confs.append(conf)
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                boxes.append({
                    "text": text, "conf": conf,
                    "x": min(x_coords), "y": min(y_coords),
                    "w": max(x_coords) - min(x_coords),
                    "h": max(y_coords) - min(y_coords),
                })

        return OCRResult(
            text       = " ".join(words),
            confidence = float(np.mean(confs)) if confs else 0.5,
            engine     = "easyocr",
            word_boxes = boxes,
        )

    def _paddleocr_run(self, image: np.ndarray) -> OCRResult:
        if not PADDLE_AVAILABLE:
            raise ImportError("paddleocr not installed")

        ocr = self._get_paddle()
        result = ocr.ocr(image, cls=self.cfg.paddle_use_angle_classifier)

        words, boxes, confs = [], [], []
        if result and result[0]:
            for line in result[0]:
                bbox, (text, conf) = line
                if conf >= self.cfg.min_confidence:
                    words.append(text)
                    confs.append(conf)
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    boxes.append({
                        "text": text, "conf": conf,
                        "x": min(x_coords), "y": min(y_coords),
                        "w": max(x_coords) - min(x_coords),
                        "h": max(y_coords) - min(y_coords),
                    })

        return OCRResult(
            text       = "\n".join(words),
            confidence = float(np.mean(confs)) if confs else 0.5,
            engine     = "paddleocr",
            word_boxes = boxes,
        )

    def _surya_run(self, image: np.ndarray) -> Optional[OCRResult]:
        """Surya OCR — layout-aware line-level recognition."""
        try:
            from surya.ocr import run_ocr
            from surya.model.detection.model import load_model as load_det
            from surya.model.recognition.model import load_model as load_rec
            from surya.model.recognition.processor import load_processor

            det_model  = load_det()
            rec_model  = load_rec()
            processor  = load_processor()
            pil_img    = Image.fromarray(image)

            results = run_ocr(
                [pil_img], [self.cfg.easyocr_langs],
                det_model, rec_model, processor
            )

            lines = []
            for page_result in results:
                for line in page_result.text_lines:
                    lines.append(line.text)

            return OCRResult(
                text       = "\n".join(lines),
                confidence = 0.85,
                engine     = "surya",
                word_boxes = [],
            )
        except Exception as e:
            logger.debug(f"  Surya OCR failed: {e}")
            return None

    # ── Ensemble ──────────────────────────────────────────────

    def _ensemble(self, results: list[OCRResult]) -> OCRResult:
        strategy = self.cfg.ensemble_strategy

        if strategy == "confidence" or len(results) == 1:
            best = max(results, key=lambda r: r.confidence)
            return best

        if strategy == "longest":
            best = max(results, key=lambda r: len(r.text))
            return best

        if strategy == "voting":
            # Word-level majority voting
            all_words = [r.text.split() for r in results]
            max_len   = max(len(w) for w in all_words)
            merged    = []
            for i in range(max_len):
                candidates = [w[i] for w in all_words if i < len(w)]
                from collections import Counter
                winner = Counter(candidates).most_common(1)[0][0]
                merged.append(winner)
            best_conf = float(np.mean([r.confidence for r in results]))
            return OCRResult(
                text       = " ".join(merged),
                confidence = best_conf,
                engine     = "ensemble",
                word_boxes = results[0].word_boxes,
            )

        return results[0]

    # ── Text Correction ───────────────────────────────────────

    def _clean_layout_artifacts(self, text: str) -> str:
        """Remove common OCR / layout noise."""
        text = re.sub(r' +', ' ', text)                   # Multiple spaces → single
        text = re.sub(r'\n{3,}', '\n\n', text)            # Max 2 blank lines
        text = re.sub(r'(?<=[a-z])-\n(?=[a-z])', '', text)  # Rejoin hyphenated words
        text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]', '', text)  # Non-printable
        return text.strip()

    def _symspell_correct(self, text: str) -> str:
        ss = self._get_symspell()
        if ss is None:
            return text
        words = text.split()
        corrected = []
        for word in words:
            # Skip numbers, caps acronyms, short tokens
            if re.match(r'^[0-9.,\-]+$', word) or (word.isupper() and len(word) > 1):
                corrected.append(word)
                continue
            suggestions = ss.lookup(
                word.lower(), Verbosity.CLOSEST, max_edit_distance=2
            )
            if suggestions:
                # Preserve original casing
                best = suggestions[0].term
                if word[0].isupper():
                    best = best.capitalize()
                corrected.append(best)
            else:
                corrected.append(word)
        return " ".join(corrected)

    def _grammar_correct(self, text: str) -> str:
        tool = self._get_language_tool()
        if tool is None:
            return text
        try:
            matches = tool.check(text)
            return language_tool_python.utils.correct(text, matches)
        except Exception as e:
            logger.debug(f"  LanguageTool error: {e}")
            return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
    def _claude_correct(self, text: str, context_hint: str = "") -> str:
        """Use Claude to intelligently fix OCR errors in context."""
        client = self._get_anthropic()
        if client is None:
            return text

        context_note = f"\nDocument context hint: {context_hint}" if context_hint else ""
        prompt = f"""You are correcting OCR-extracted text from a PDF document.
Fix OCR errors, misrecognised characters, and broken word spacing.
Preserve the original structure, formatting, technical terms, equations, and proper nouns.
Return ONLY the corrected text — no explanation, no preamble.{context_note}

--- OCR TEXT ---
{text[:3000]}
--- END ---"""

        response = client.messages.create(
            model      = self.cfg.enable_ai_correction and self.api_cfg.claude_fast or "claude-sonnet-4-20250514",
            max_tokens = self.api_cfg.max_tokens_correction,
            messages   = [{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    # ── Lazy Loaders ──────────────────────────────────────────

    def _get_easyocr(self):
        if self._easyocr is None and EASYOCR_AVAILABLE:
            self._easyocr = easyocr.Reader(
                self.cfg.easyocr_langs,
                gpu    = self.cfg.easyocr_gpu,
                verbose = False,
            )
        return self._easyocr

    def _get_paddle(self):
        if self._paddle is None and PADDLE_AVAILABLE:
            self._paddle = PaddleOCR(
                use_angle_cls = self.cfg.paddle_use_angle_classifier,
                lang          = self.cfg.paddle_lang,
                use_gpu       = self.cfg.paddle_use_gpu,
                show_log      = False,
            )
        return self._paddle

    def _get_symspell(self):
        if self._symspell is None and SYMSPELL_AVAILABLE:
            ss = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            import pkg_resources
            dict_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt"
            )
            bigram_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
            )
            if Path(dict_path).exists():
                ss.load_dictionary(dict_path, term_index=0, count_index=1)
            if Path(bigram_path).exists():
                ss.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
            self._symspell = ss
        return self._symspell

    def _get_language_tool(self):
        if self._lt_tool is None and LANGUAGETOOL_AVAILABLE:
            try:
                self._lt_tool = language_tool_python.LanguageTool("en-US")
            except Exception as e:
                logger.warning(f"  LanguageTool init failed: {e}")
        return self._lt_tool

    def _get_anthropic(self):
        if self._anthropic is None:
            key = self.api_cfg.anthropic_api_key
            if key:
                self._anthropic = anthropic.Anthropic(api_key=key)
        return self._anthropic