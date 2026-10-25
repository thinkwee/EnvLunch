"""slime data source that skips the image processor, so vision-language checkpoints train on text only."""

import slime.rollout.data_source as _ds
from slime.rollout.data_source import RolloutDataSourceWithBuffer


class TextOnlyRolloutDataSource(RolloutDataSourceWithBuffer):
    """Rollout data source with the multimodal processor disabled."""
    def __init__(self, args):
        """Build the parent data source while load_processor returns None."""
        original = _ds.load_processor
        _ds.load_processor = lambda *a, **k: None
        try:
            super().__init__(args)
        finally:
            _ds.load_processor = original
