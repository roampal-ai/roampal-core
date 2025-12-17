# roampal-core v0.1.5 Release Notes

**Release Date:** December 2024

## Summary

DEV/PROD isolation - run development and production instances simultaneously without collision.

## Changes

### DEV/PROD Separation (Main Feature)

**Ports:**
- PROD: 27182 (unchanged)
- DEV: 27183 (new)

**Data Directories:**
- PROD: `Roampal/data`
- DEV: `Roampal_DEV/data`

**MCP Server:**
- `run_mcp_server()` now accepts `dev` parameter
- Passes `ROAMPAL_DEV=1` environment variable when in dev mode
- Auto-starts FastAPI server on correct port based on mode

**CLI:**
- `--dev` flag now works on `status` and `stats` commands (not just `start`)
- Better mode/port display in output
- Shows which mode is running and which port to use

### Batch Working Memory Cleanup

- Auto-cleanup triggered every 50 outcome scores
- Removes working memories older than 24 hours
- Uses promotion_service.cleanup_old_working_memory()

### Minor Changes

- Server main.py accepts `--host` and `--port` args via argparse
- Arrow encoding fix (`→` to `->`) in log messages for Windows compatibility
- Ingest timeout increased from 60s to 300s for large files

## Files Changed

```
roampal/backend/modules/memory/outcome_service.py  | +20 lines
roampal/cli.py                                     | +111/-52 lines
roampal/mcp/server.py                              | +59 lines
roampal/server/main.py                             | +8 lines
```

Total: +146/-52 lines across 4 files

## Upgrade Notes

No breaking changes. Existing installations will continue using PROD port (27182) by default.

To use dev mode:
```bash
roampal start --dev    # Starts on port 27183 with Roampal_DEV data
roampal status --dev   # Check dev server status
roampal stats --dev    # Show dev memory statistics
```

## Previous Version

- v0.1.4: Add missing nltk dependency for BM25 hybrid search