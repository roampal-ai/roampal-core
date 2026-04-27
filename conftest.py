"""Root pytest config — prevents asyncio teardown hangs on Linux CI."""

import asyncio


def pytest_unconfigure(config):
    """Force-close any lingering event loops at session end to prevent teardown hangs."""
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            try:
                for task in asyncio.all_tasks(loop):
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
            except Exception:
                pass
            loop.close()
    except Exception:
        pass
