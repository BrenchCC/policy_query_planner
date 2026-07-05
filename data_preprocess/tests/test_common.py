from data_preprocess.common import normalized_key, normalize_text, stable_hash, strip_html


def test_normalize_text() -> None:
    """Normalize whitespace and smart punctuation."""
    assert normalize_text("  it’s\nvalid  ") == "it's valid"


def test_strip_html() -> None:
    """Extract visible text and preserve the source tag."""
    tag, text = strip_html("<li>children <b>under 18</b></li>")
    assert tag == "li"
    assert text == "children under 18"


def test_stable_hash() -> None:
    """Generate stable but input-sensitive identifiers."""
    assert stable_hash("a", "b") == stable_hash("a", "b")
    assert stable_hash("a", "b") != stable_hash("b", "a")


def test_normalized_key() -> None:
    """Ignore punctuation and case in comparison keys."""
    assert normalized_key("Gary Cherone?") == normalized_key("gary-cherone")

