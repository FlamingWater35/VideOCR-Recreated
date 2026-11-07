"""Tests for the About tab: hero header, update indicator, and links."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from videocr_gui import about_tab as about_module
from videocr_gui import i18n
from videocr_gui.about_tab import AboutTab, ClickableLabel
from videocr_gui.update_check import LATEST_RELEASE_URL


@pytest.fixture(autouse=True)
def _english():
    i18n.load_language("en")


@pytest.fixture
def tab(qtbot) -> AboutTab:
    widget = AboutTab("1.6.3")
    qtbot.addWidget(widget)
    widget.resize(720, 520)
    widget.show()
    qtbot.waitExposed(widget, timeout=3000)
    return widget


class TestLayout:
    def test_version_and_title_shown(self, tab):
        assert tab.title_lbl.text() == "VideOCR Recreated"
        assert tab.version_lbl.text() == "Version: 1.6.3"
        assert tab.tagline_lbl.text()

    def test_repo_links_are_clickable(self, tab):
        links = tab.findChildren(ClickableLabel)
        texts = {link.text() for link in links}
        assert set(about_module.REPO_LINKS) <= texts
        assert f"{about_module.RELEASES_REPO_URL}/issues" in texts

    def test_repo_links_open_in_browser(self, tab, qtbot, monkeypatch):
        opened: list[str] = []
        monkeypatch.setattr(
            about_module.webbrowser,
            "open",
            lambda url, *a, **k: opened.append(url) or True,
        )
        links = {link.text(): link for link in tab.findChildren(ClickableLabel)}
        for url in about_module.REPO_LINKS:
            qtbot.mouseClick(links[url], Qt.MouseButton.LeftButton)
        assert opened == list(about_module.REPO_LINKS)

    def test_fork_notice_mentions_original(self, tab):
        assert "timminator" in tab.fork_lbl.text()


class TestUpdateIndicator:
    def test_card_hidden_at_rest(self, tab):
        assert tab.update_state() == "idle"
        assert not tab.update_card.isVisible()

    def test_checking_state_shows_progress_badge(self, tab, qtbot):
        tab.set_update_state("checking")
        assert tab.update_card.isVisible()
        assert tab.update_badge.text() == "Checking for updates..."
        assert not tab.update_button.isVisible()

    def test_available_state_shows_button_and_versions(self, tab, qtbot):
        tab.set_update_state("available", "9.9.9")
        assert tab.update_card.isVisible()
        assert tab.has_update()
        assert tab.update_badge.text() == "Update Available"
        assert "9.9.9" in tab.update_message.text()
        # The current version lives in the hero pill only; repeating it in the
        # card was redundant.
        assert "1.6.3" not in tab.update_message.text()
        assert "1.6.3" in tab.version_lbl.text()
        assert tab.update_button.isVisible()
        assert tab.update_button.text() == "Open Latest Release"

    def test_available_state_marks_the_card(self, tab):
        tab.set_update_state("available", "9.9.9")
        assert tab.update_card.property("state") == "available"
        assert tab.update_badge.property("state") == "available"

    def test_button_opens_latest_release(self, tab, qtbot, monkeypatch):
        opened: list[str] = []
        monkeypatch.setattr(
            about_module.webbrowser,
            "open",
            lambda url, *a, **k: opened.append(url) or True,
        )
        tab.set_update_state("available", "9.9.9")
        qtbot.mouseClick(tab.update_button, Qt.MouseButton.LeftButton)
        assert opened == [LATEST_RELEASE_URL]

    def test_uptodate_state_hides_the_button(self, tab):
        tab.set_update_state("available", "9.9.9")
        tab.set_update_state("uptodate")
        assert tab.update_card.isVisible()
        assert not tab.has_update()
        assert tab.update_badge.text() == "Up to Date"
        assert not tab.update_button.isVisible()
        assert tab.update_message.text()

    def test_back_to_idle_hides_the_card(self, tab):
        tab.set_update_state("available", "9.9.9")
        tab.set_update_state("idle")
        assert not tab.update_card.isVisible()

    def test_unknown_state_is_treated_as_idle(self, tab):
        tab.set_update_state("nonsense")
        assert tab.update_state() == "idle"
        assert not tab.update_card.isVisible()

    def test_retranslate_preserves_the_state(self, tab):
        tab.set_update_state("available", "9.9.9")
        tab.retranslate("1.6.3")
        assert tab.has_update()
        assert "9.9.9" in tab.update_message.text()

    def test_retranslate_updates_the_version_pill(self, tab):
        tab.retranslate("2.0.0")
        assert tab.version_lbl.text() == "Version: 2.0.0"


class TestTranslations:
    @pytest.mark.parametrize("code", ["de", "ja", "ar", "ch", "ru"])
    def test_state_texts_are_translated(self, tab, code):
        i18n.load_language(code)
        tab.retranslate("1.6.3")
        english = {
            "Checking for updates...",
            "Update Available",
            "Up to Date",
            "Open Latest Release",
        }
        tab.set_update_state("available", "9.9.9")
        assert tab.update_button.text() not in english

    def test_missing_placeholder_does_not_crash(self, tab, monkeypatch):
        # A translation that lost its placeholders must render, not raise.
        monkeypatch.setattr(
            about_module.i18n, "tr", lambda key, default=None: "broken {} {"
        )
        tab.set_update_state("available", "9.9.9")
        assert tab.update_message.text()
