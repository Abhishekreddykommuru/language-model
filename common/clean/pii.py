"""PII anonymisation: email addresses and phone numbers.

Replaced rather than deleted, because removing a token mid-sentence leaves
ungrammatical training text.
"""
import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Indian (+91) and Nepali (+977) phone shapes. Requires 10-12 digits total so
# that years, prices and short numeric strings are not clobbered.
PHONE_RE = re.compile(
    r"(?<![\d\w])(?:\+?9[17][\s\-.]?)?(?:\d[\s\-.]?){9,11}\d(?![\d\w])"
)


def anonymize(text: str) -> tuple:
    """Replace emails and phone numbers. Returns (text, counts)."""
    counts = {"email": 0, "phone": 0}

    def _email(_m):
        counts["email"] += 1
        return "<EMAIL>"

    def _phone(m):
        # Long digit runs are IDs, account numbers or table cells, not phone
        # numbers, and are better left alone.
        digits = sum(c.isdigit() for c in m.group())
        if digits > 12:
            return m.group()
        counts["phone"] += 1
        return "<PHONE>"

    text = EMAIL_RE.sub(_email, text)
    text = PHONE_RE.sub(_phone, text)
    return text, counts
