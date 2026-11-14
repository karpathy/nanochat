"""
Test module for nanochat.glossary.

Author: Claude Haiku 4.5
Created: 2026-11-14
Scope: Unit tests for term catalog, STAGE_TERMS mapping, and learn line formatting.
"""

import os
import pytest

from nanochat import glossary


class TestGlossaryTerms:
    """Test the glossary term catalog."""

    def test_given_glossary_when_checked_then_all_nine_terms_exist(self):
        """
        Scenario: Verify all 9 required terms are defined.
        Slice: cli-learn-color-ux

        ### Given
        The GLOSSARY dict with term definitions.

        ### When
        We check for the 9 required term keys.

        ### Then
        All terms exist with non-empty definition and URL.
        """
        required_terms = {"BPE", "vocab", "compression ratio", "BPB", "CORE", "ChatCORE", "MFU", "Muon", "FLOPs"}
        for term in required_terms:
            assert term in glossary.GLOSSARY, f"Term '{term}' not in GLOSSARY"
            definition, url = glossary.GLOSSARY[term]
            assert definition, f"Term '{term}' has empty definition"
            assert url, f"Term '{term}' has empty URL"

    def test_given_glossary_terms_when_checked_then_each_has_one_sentence_definition(self):
        """
        Scenario: Verify each definition is a single short sentence.
        Slice: cli-learn-color-ux

        ### Given
        The GLOSSARY term definitions.

        ### When
        We check the sentence structure of each definition.

        ### Then
        Each definition is one short sentence (ends with period, no multiple sentences).
        """
        for term, (definition, _) in glossary.GLOSSARY.items():
            # Should end with period
            assert definition.endswith("."), f"Term '{term}' definition doesn't end with period: {definition}"
            # Should not have multiple sentences (i.e., period count == 1)
            period_count = definition.count(".")
            assert period_count == 1, f"Term '{term}' has {period_count} periods (should be 1): {definition}"


class TestStagTerms:
    """Test STAGE_TERMS mapping."""

    def test_given_stage_terms_when_checked_then_all_nine_glossary_terms_appear(self):
        """
        Scenario: Verify every term appears in at least one stage.
        Slice: cli-learn-color-ux

        ### Given
        The GLOSSARY and STAGE_TERMS.

        ### When
        We collect all terms referenced by STAGE_TERMS.

        ### Then
        All 9 terms appear in at least one stage entry.
        """
        required_terms = {"BPE", "vocab", "compression ratio", "BPB", "CORE", "ChatCORE", "MFU", "Muon", "FLOPs"}
        all_stage_terms = set()
        for stage_keys in glossary.STAGE_TERMS.values():
            all_stage_terms.update(stage_keys)

        for term in required_terms:
            assert term in all_stage_terms, f"Term '{term}' is not in any STAGE_TERMS entry"

    def test_given_stage_terms_when_checked_then_each_entry_has_at_most_four_terms(self):
        """
        Scenario: Verify no stage entry exceeds 4 terms.
        Slice: cli-learn-color-ux

        ### Given
        The STAGE_TERMS mapping.

        ### When
        We check the size of each tuple.

        ### Then
        Each stage entry has at most 4 terms.
        """
        for stage, keys in glossary.STAGE_TERMS.items():
            assert len(keys) <= 4, f"Stage '{stage}' has {len(keys)} terms (max 4): {keys}"

    def test_given_stage_terms_when_checked_then_all_reference_known_glossary_terms(self):
        """
        Scenario: Verify all terms in STAGE_TERMS are defined in GLOSSARY.
        Slice: cli-learn-color-ux

        ### Given
        GLOSSARY and STAGE_TERMS.

        ### When
        We check each term in STAGE_TERMS entries.

        ### Then
        All terms are defined in GLOSSARY.
        """
        for stage, keys in glossary.STAGE_TERMS.items():
            for key in keys:
                assert key in glossary.GLOSSARY, f"Stage '{stage}' references unknown term '{key}'"

    def test_given_stage_terms_when_checked_then_base_train_has_exactly_four_terms(self):
        """
        Scenario: Verify base_train has exactly 4 terms (the max).
        Slice: cli-learn-color-ux

        ### Given
        STAGE_TERMS.

        ### When
        We check base_train's entry.

        ### Then
        base_train has exactly 4 terms: BPB, Muon, MFU, FLOPs.
        """
        assert "base_train" in glossary.STAGE_TERMS
        base_train_terms = glossary.STAGE_TERMS["base_train"]
        assert len(base_train_terms) == 4
        assert set(base_train_terms) == {"BPB", "Muon", "MFU", "FLOPs"}

    def test_given_stage_terms_when_checked_then_chat_eval_includes_chatcore(self):
        """
        Scenario: Verify chat_eval includes ChatCORE.
        Slice: cli-learn-color-ux

        ### Given
        STAGE_TERMS.

        ### When
        We check chat_eval's entry.

        ### Then
        chat_eval includes ChatCORE.
        """
        assert "chat_eval" in glossary.STAGE_TERMS
        chat_eval_terms = glossary.STAGE_TERMS["chat_eval"]
        assert "ChatCORE" in chat_eval_terms


class TestFormatLearnLine:
    """Test format_learn_line function."""

    def test_given_base_train_stage_when_formatted_then_learn_line_contains_all_four_terms(self):
        """
        Scenario: Format learn line for base_train stage.
        Slice: cli-learn-color-ux

        ### Given
        The glossary with base_train terms.

        ### When
        We call format_learn_line("base_train").

        ### Then
        Returns one line starting with "learn:" and containing all 4 terms.
        """
        line = glossary.format_learn_line("base_train")
        assert line.startswith("learn:")
        assert "BPB" in line
        assert "Muon" in line
        assert "MFU" in line
        assert "FLOPs" in line
        # Should be one line (no newlines)
        assert "\n" not in line

    def test_given_tok_train_stage_when_formatted_then_learn_line_contains_three_terms(self):
        """
        Scenario: Format learn line for tok_train stage.
        Slice: cli-learn-color-ux

        ### Given
        The glossary with tok_train terms.

        ### When
        We call format_learn_line("tok_train").

        ### Then
        Returns one line with BPE, vocab, compression ratio.
        """
        line = glossary.format_learn_line("tok_train")
        assert line.startswith("learn:")
        assert "BPE" in line
        assert "vocab" in line
        assert "compression ratio" in line

    def test_given_chat_sft_stage_when_formatted_then_learn_line_returns_empty_string(self):
        """
        Scenario: Format learn line for stage with no terms.
        Slice: cli-learn-color-ux

        ### Given
        The glossary with chat_sft (empty terms).

        ### When
        We call format_learn_line("chat_sft").

        ### Then
        Returns empty string.
        """
        line = glossary.format_learn_line("chat_sft")
        assert line == ""

    def test_given_unknown_stage_when_formatted_then_learn_line_returns_empty_string(self):
        """
        Scenario: Format learn line for unknown stage.
        Slice: cli-learn-color-ux

        ### Given
        The glossary.

        ### When
        We call format_learn_line with an unknown stage name.

        ### Then
        Returns empty string.
        """
        line = glossary.format_learn_line("unknown_stage")
        assert line == ""

    def test_given_all_stages_when_formatted_then_all_non_empty_learn_lines_are_single_line(self):
        """
        Scenario: Verify all non-empty learn lines are single lines.
        Slice: cli-learn-color-ux

        ### Given
        The glossary with all stages.

        ### When
        We format learn lines for all stages.

        ### Then
        Each non-empty line has no newlines.
        """
        for stage in glossary.STAGE_TERMS.keys():
            line = glossary.format_learn_line(stage)
            if line:
                assert "\n" not in line, f"Stage '{stage}' returned multi-line: {repr(line)}"


class TestEmitDefine:
    """Test emit_define function."""

    def test_given_unbriefed_stage_when_emit_define_called_twice_then_prints_once(self, capsys, monkeypatch):
        """
        Scenario: emit_define prints once per term when NANOCHAT_STAGE_BRIEFED is not set.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_STAGE_BRIEFED is not set, and _defined_terms is empty.

        ### When
        We call emit_define("BPB") twice.

        ### Then
        It prints the define line once, the second call is a no-op.
        """
        # Clear the memoization set
        glossary._defined_terms.clear()

        # Ensure env var is not set
        monkeypatch.delenv("NANOCHAT_STAGE_BRIEFED", raising=False)

        glossary.emit_define("BPB")
        captured = capsys.readouterr()
        first_output = captured.out

        glossary.emit_define("BPB")
        captured = capsys.readouterr()
        second_output = captured.out

        # First call prints, second doesn't
        assert "define:" in first_output
        assert "BPB" in first_output
        assert second_output == ""

    def test_given_nanochat_stage_briefed_set_when_emit_define_called_then_prints_nothing(self, capsys, monkeypatch):
        """
        Scenario: emit_define is a no-op when NANOCHAT_STAGE_BRIEFED=1.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_STAGE_BRIEFED=1.

        ### When
        We call emit_define("BPB").

        ### Then
        It prints nothing.
        """
        glossary._defined_terms.clear()
        monkeypatch.setenv("NANOCHAT_STAGE_BRIEFED", "1")

        glossary.emit_define("BPB")
        captured = capsys.readouterr()

        assert captured.out == ""

    def test_given_unbriefed_stage_when_emit_define_called_for_different_terms_then_each_prints_once(self, capsys, monkeypatch):
        """
        Scenario: emit_define prints different terms on first call each.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_STAGE_BRIEFED is not set, and _defined_terms is empty.

        ### When
        We call emit_define for two different terms.

        ### Then
        Each term prints its define line once.
        """
        glossary._defined_terms.clear()
        monkeypatch.delenv("NANOCHAT_STAGE_BRIEFED", raising=False)

        glossary.emit_define("BPB")
        captured1 = capsys.readouterr()

        glossary.emit_define("MFU")
        captured2 = capsys.readouterr()

        assert "BPB" in captured1.out
        assert "MFU" in captured2.out
        assert "define:" in captured1.out
        assert "define:" in captured2.out

    def test_given_unbriefed_stage_when_emit_define_called_with_unknown_term_then_prints_nothing(self, capsys, monkeypatch):
        """
        Scenario: emit_define does nothing for unknown term.
        Slice: cli-learn-color-ux

        ### Given
        NANOCHAT_STAGE_BRIEFED is not set, and _defined_terms is empty.

        ### When
        We call emit_define with an unknown term.

        ### Then
        It prints nothing.
        """
        glossary._defined_terms.clear()
        monkeypatch.delenv("NANOCHAT_STAGE_BRIEFED", raising=False)

        glossary.emit_define("UnknownTerm")
        captured = capsys.readouterr()

        assert captured.out == ""
