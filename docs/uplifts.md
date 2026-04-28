# Platform Uplifts

This page records compatibility patches and platform-specific adaptations applied on top of the upstream `PoggioAI/PoggioAI_MSc` codebase.

---

## Linux → Windows (April 2026)

**Branch:** `MSc_Prod`  
**Motivation:** The upstream codebase targets Linux/macOS. Running on Windows 11 caused 6 test failures and one runtime crash in the `.env` loader.

### Changes

