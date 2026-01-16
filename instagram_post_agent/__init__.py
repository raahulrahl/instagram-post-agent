# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""instagram-post-agent - A Bindu Agent."""

from instagram_post_agent.__version__ import __version__
from instagram_post_agent.main import (
    cleanup,
    handler,
    initialize_crew,
    main,
)

__all__ = [
    "__version__",
    "cleanup",
    "handler",
    "initialize_crew",
    "main",
]
