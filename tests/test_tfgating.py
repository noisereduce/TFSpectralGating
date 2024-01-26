import unittest
import tensorflow as tf
from tfgating import TFGating as TG


class TestTFGating(unittest.TestCase):
    """
    Test cases for the TFGating class.
    """

    def test_nonstationary(self, sr: int = 8000):
        """
        Test Non-Stationary

        Args:
            sr (int): Signal sampling frequency.
        """
        # Create TFGating instance
        tg = TG(sr=sr, nonstationary=True)

        # Apply Spectral Gate to noisy speech signal
        noisy_speech = tf.random.normal(shape=(3, 32000), dtype=tf.float32)
        enhanced_speech = tg(noisy_speech)
        self.assertIsNotNone(enhanced_speech)

    def test_stationary(self, sr: int = 8000):
        """
        Test Stationary.

        Args:
            sr (int): Signal sampling frequency.
        """
        # Create TFGating instance
        tg = TG(sr=sr, nonstationary=False)

        # Apply Spectral Gate to noisy speech signal
        noisy_speech = tf.random.normal(shape=(3, 32000), dtype=tf.float32)
        enhanced_speech = tg(noisy_speech)
        self.assertIsNotNone(enhanced_speech)


if __name__ == '__main__':
    unittest.main()