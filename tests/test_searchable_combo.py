"""Tests for the searchable subtitle-language combo box."""

from __future__ import annotations

from PySide6.QtCore import Qt

LANGUAGES = [
    "Japanese",
    "English",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
    "Polish",
    "Korean + English",
]


def _combo(qtbot):
    from videocr_gui.widgets import SearchableComboBox

    combo = SearchableComboBox()
    qtbot.addWidget(combo)
    combo.addItems(LANGUAGES)
    combo.setCurrentIndex(LANGUAGES.index("English"))
    return combo


class TestSearchableComboBox:
    def test_is_editable_and_has_a_completer(self, qtbot):
        combo = _combo(qtbot)
        assert combo.isEditable()
        line = combo.lineEdit()
        assert line is not None
        completer = line.completer()
        assert completer is not None
        assert completer.filterMode() == Qt.MatchFlag.MatchContains
        assert completer.caseSensitivity() == Qt.CaseSensitivity.CaseInsensitive

    def test_typing_filters_the_popup(self, qtbot):
        """MatchContains means a mid-word fragment finds the entry."""
        combo = _combo(qtbot)
        completer = combo.lineEdit().completer()
        completer.setCompletionPrefix("jap")
        matches = [
            completer.completionModel().index(i, 0).data()
            for i in range(completer.completionCount())
        ]
        assert "Japanese" in matches

    def test_no_insert_policy_so_typing_cannot_add_entries(self, qtbot):
        from PySide6.QtWidgets import QComboBox

        combo = _combo(qtbot)
        assert combo.insertPolicy() == QComboBox.InsertPolicy.NoInsert
        assert combo.count() == len(LANGUAGES)

    def test_exact_name_is_accepted_case_insensitively(self, qtbot):
        combo = _combo(qtbot)
        combo.setEditText("polish")
        combo._snap_to_selection()
        assert combo.currentText() == "Polish"

    def test_unique_fragment_is_accepted(self, qtbot):
        combo = _combo(qtbot)
        combo.setEditText("jap")
        combo._snap_to_selection()
        assert combo.currentText() == "Japanese"

    def test_ambiguous_fragment_reverts_to_the_current_entry(self, qtbot):
        """ "chinese" matches two entries, so it must not pick one at random."""
        combo = _combo(qtbot)
        combo.setEditText("chinese")
        combo._snap_to_selection()
        assert combo.currentText() == "English"

    def test_unknown_text_reverts_to_the_current_entry(self, qtbot):
        combo = _combo(qtbot)
        combo.setEditText("klingon")
        combo._snap_to_selection()
        assert combo.currentText() == "English"

    def test_empty_text_reverts_to_the_current_entry(self, qtbot):
        combo = _combo(qtbot)
        combo.setEditText("")
        combo._snap_to_selection()
        assert combo.currentText() == "English"

    def test_model_is_never_rewritten(self, qtbot):
        """Filtering must not shrink the model callers rely on."""
        combo = _combo(qtbot)
        combo.setEditText("jap")
        combo._snap_to_selection()
        assert [combo.itemText(i) for i in range(combo.count())] == LANGUAGES

    def test_find_text_still_locates_saved_selections(self, qtbot):
        combo = _combo(qtbot)
        assert combo.findText("Polish") == LANGUAGES.index("Polish")


class TestLanguageComboWiring:
    def test_main_window_uses_the_searchable_combo(self, qtbot, monkeypatch):
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        from videocr_gui.app import MainWindow
        from videocr_gui.widgets import SearchableComboBox

        window = MainWindow()
        qtbot.addWidget(window)
        try:
            assert isinstance(window.lang_combo, SearchableComboBox)
        finally:
            window.close()

    def test_switching_engines_keeps_the_combo_searchable(self, qtbot, monkeypatch):
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        from videocr_gui.app import MainWindow

        window = MainWindow()
        qtbot.addWidget(window)
        try:
            for engine in (
                "PaddleOCR (Det. + Rec.)",
                "PaddleOCR (Det.) + Google Lens (Rec.)",
                "ONNX Runtime DirectML (AMD GPU Experimental)",
            ):
                window._update_lang_list(engine)
                assert window.lang_combo.isEditable()
                assert window.lang_combo.lineEdit().completer() is not None
        finally:
            window.close()
