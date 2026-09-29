"""Build identifier, so a stale unzip on Colab is obvious rather than puzzling.

/content/code is deleted and re-unzipped every session. Uploading a new zip
without re-running that cell leaves the old code in place, and the symptom is a
bug you have already fixed reappearing with line numbers that do not match the
file you are looking at.
"""

BUILD = "2026-08-19 lid-numpy2-fix"
