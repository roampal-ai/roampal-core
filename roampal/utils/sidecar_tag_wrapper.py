"""v0.5.4: Sidecar-backed TagService LLM extractor factory.

Mirrors Desktop's `utils/sidecar_tag_wrapper.py` so Core and Desktop share
the same architectural pattern: a closure handed to TagService.set_llm_extract_fn
that reads sidecar configuration at CALL time, not at construction time.

Why this exists for Core: `roampal.sidecar_service.extract_tags` reads its
config from the module-level globals `CUSTOM_URL` / `CUSTOM_MODEL`, which are
populated at import time from `os.environ`. The FastAPI server is launched
independently of the OpenCode plugin, so those env vars often aren't set when
the module first loads. The lifespan hydration helper mutates the globals
after import — but if anything in the future bypasses hydration, the wrapper
still degrades cleanly to "no tags" instead of raising.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def make_llm_tag_extractor():
    """Return a sync `extract(text) -> Optional[List[str]]` closure.

    Reads `sidecar_service.CUSTOM_URL` / `CUSTOM_MODEL` at call time. If
    sidecar is unconfigured, returns None so TagService treats it as "no
    extraction" rather than raising.
    """

    def llm_tag_extractor(text: str) -> Optional[List[str]]:
        import roampal.sidecar_service as svc

        if not svc.CUSTOM_URL or not svc.CUSTOM_MODEL:
            return None
        try:
            return svc.extract_tags(text)
        except Exception as e:
            logger.warning(f"Sidecar tag extraction failed: {e}")
            return None

    return llm_tag_extractor
