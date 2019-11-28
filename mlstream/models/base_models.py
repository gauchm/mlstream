from pathlib import Path
import numpy as np

from ..datasets import (LumpedBasin, LumpedH5)


class LumpedModel:
    """Model that operates on lumped (daily, basin-averaged) inputs. """

    def load(self, model_file: Path) -> None:
        """Loads a trained and pickled model.

        Parameters
        ----------
        model_file : Path
            Path to the stored model.
        """
        pass

    def train(self, ds: LumpedH5) -> None:
        """Trains the model.

        Parameters
        ----------
        ds : LumpedH5
            Training dataset
        """
        pass

    def predict(self, ds: LumpedBasin) -> np.ndarray:
        """Generates predictions for a basin.

        Parameters
        ----------
        ds : LumpedBasin
            Dataset of the basin to predict.

        Returns
        -------
        np.ndarray
            Array of predictions.
        """
        pass
