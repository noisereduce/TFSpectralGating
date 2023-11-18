import tensorflow as tf
import tensorflow.signal as signal
from typing import Union, Optional

class TensorFlowGating(tf.keras.Model):
    def __init__(self, sr: int, nonstationary: bool = False, n_std_thresh_stationary: float = 1.5,
                 n_thresh_nonstationary: bool = 1.3, temp_coeff_nonstationary: float = 0.1,
                 n_movemean_nonstationary: int = 20, prop_decrease: float = 1.0, n_fft: int = 1024,
                 win_length: bool = None, hop_length: int = None, freq_mask_smooth_hz: float = 500,
                 time_mask_smooth_ms: float = 50):
        super().__init__()

        # General Params
        self.sr = sr
        self.nonstationary = nonstationary
        assert 0.0 <= prop_decrease <= 1.0
        self.prop_decrease = prop_decrease

        # STFT Params
        self.n_fft = n_fft
        self.win_length = self.n_fft if win_length is None else win_length
        self.hop_length = self.win_length // 4 if hop_length is None else hop_length

        # Stationary Params
        self.n_std_thresh_stationary = n_std_thresh_stationary

        # Non-Stationary Params
        self.temp_coeff_nonstationary = temp_coeff_nonstationary
        self.n_movemean_nonstationary = n_movemean_nonstationary
        self.n_thresh_nonstationary = n_thresh_nonstationary

        # Smooth Mask Params
        self.freq_mask_smooth_hz = freq_mask_smooth_hz
        self.time_mask_smooth_ms = time_mask_smooth_ms
        self.smoothing_filter = self._generate_mask_smoothing_filter()

    def _generate_mask_smoothing_filter(self) -> tf.Tensor:
        """
        A TensorFlow function that generates a smoothing filter for the mask.

        Returns:
            tf.Tensor: A 2D tensor representing the smoothing filter.
        """
        if self.freq_mask_smooth_hz is None and self.time_mask_smooth_ms is None:
            return None

        n_grad_freq = 1 if self.freq_mask_smooth_hz is None else int(self.freq_mask_smooth_hz / (self.sr / (self.n_fft / 2)))
        if n_grad_freq < 1:
            raise ValueError(f"freq_mask_smooth_hz needs to be at least {int(self.sr / (self.n_fft / 2))} Hz")

        n_grad_time = 1 if self.time_mask_smooth_ms is None else int(self.time_mask_smooth_ms / ((self.hop_length / self.sr) * 1000))
        if n_grad_time < 1:
            raise ValueError(f"time_mask_smooth_ms needs to be at least {int((self.hop_length / self.sr) * 1000)} ms")

        if n_grad_time == 1 and n_grad_freq == 1:
            return None

        v_f = tf.concat([linspace(0.0, 1.0, n_grad_freq + 1, endpoint=False), linspace(1.0, 0.0, n_grad_freq + 2)], axis=0)[1:-1]
        v_t = tf.concat([linspace(0.0, 1.0, n_grad_time + 1, endpoint=False), linspace(1.0, 0.0, n_grad_time + 2)], axis=0)[1:-1]
        smoothing_filter = tf.einsum('i,j->ij', v_f, v_t)[tf.newaxis, tf.newaxis]

        return smoothing_filter / tf.reduce_sum(smoothing_filter)

    def _stationary_mask(self, X_db, xn=None):
        if xn is not None:
            XN = signal.stft(xn, frame_length=self.n_fft, frame_step=self.hop_length, pad_end=True, fft_length=self.n_fft, window_fn=tf.signal.hann_window)
            XN_db = amp_to_db(XN)
        else:
            XN_db = X_db

        std_freq_noise, mean_freq_noise = tf.math.reduce_std(XN_db, axis=1, keepdims=True), tf.math.reduce_mean(XN_db, axis=1, keepdims=True)

        noise_thresh = mean_freq_noise + std_freq_noise * self.n_std_thresh_stationary
        sig_mask = tf.math.greater(X_db, noise_thresh)

        return sig_mask

    def _nonstationary_mask(self, X_abs):
        X_smoothed = signal.conv1d(X_abs, tf.ones(self.n_movemean_nonstationary, dtype=X_abs.dtype, shape=[1, 1, self.n_movemean_nonstationary]), padding="SAME")
        slowness_ratio = (X_abs - X_smoothed) / X_smoothed
        sig_mask = temperature_sigmoid(slowness_ratio, self.n_thresh_nonstationary, self.temp_coeff_nonstationary)

        return sig_mask

    def call(self, x, xn=None):
        assert x.shape.ndims == 2
        if x.shape[-1] < self.win_length * 2:
            raise Exception(f'x must be bigger than {self.win_length * 2}')

        assert xn is None or xn.shape.ndims == 1 or xn.shape.ndims == 2
        if xn is not None and xn.shape[-1] < self.win_length * 2:
            raise Exception(f'xn must be bigger than {self.win_length * 2}')

        X = signal.stft(x, frame_length=self.n_fft, frame_step=self.hop_length, pad_end=True, fft_length=self.n_fft, window_fn=tf.signal.hann_window)

        if self.nonstationary:
            sig_mask = self._nonstationary_mask(tf.abs(X))
        else:
            sig_mask = self._stationary_mask(amp_to_db(X), xn)

        sig_mask = self.prop_decrease * (tf.where(sig_mask, 1.0, 0.0) - 1.0) + 1.0

        self.smoothing_filter = tf.reshape(self.smoothing_filter, [self.smoothing_filter.shape[2], self.smoothing_filter.shape[3], 1, 1])

        if self.smoothing_filter is not None:
            sig_mask = tf.nn.conv2d(
                sig_mask[..., tf.newaxis],  # Add an extra channel dimension
                self.smoothing_filter,
                strides=[1, 1, 1, 1],
                padding='SAME'
                )


        sig_mask = tf.complex(sig_mask, 0.0)[..., 0]
        Y = X * sig_mask

        y = signal.inverse_stft(Y, frame_length=self.n_fft, frame_step=self.hop_length, fft_length=self.n_fft, window_fn=tf.signal.hann_window)

        return y
    