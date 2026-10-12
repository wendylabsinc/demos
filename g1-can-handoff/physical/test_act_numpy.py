"""Regression coverage for caching the frozen ACT decoder sublayer."""
from pathlib import Path
import unittest

import numpy as np

from act_numpy import NumpyACT, verify_parity


DIRECTORY = Path(__file__).parent / 'act_policy'


def uncached_chunk(model, observation):
    """Original full decoder, including the observation-independent sublayer."""
    obs = np.asarray(observation, dtype=np.float32)
    memory = np.stack((model.linear(obs, 'obs_embed'), model.weights['latent_embed.bias']))
    value = model.weights['queries'][0].copy()
    for index in range(2):
        prefix = f'decoder.layers.{index}.'
        value = model.norm(value + model.attention(value, value, prefix + 'self_attn'), prefix + 'norm1')
        value = model.norm(value + model.attention(value, memory, prefix + 'multihead_attn'), prefix + 'norm2')
        feedforward = model.linear(np.maximum(model.linear(value, prefix + 'linear1'), 0), prefix + 'linear2')
        value = model.norm(value + feedforward, prefix + 'norm3')
    return np.tanh(model.linear(value, 'head'))


class NumpyACTCacheTests(unittest.TestCase):
    def test_cached_decoder_matches_uncached_and_1000_pytorch_fixtures(self):
        model = NumpyACT(DIRECTORY)
        with np.load(DIRECTORY / 'parity-fixtures.npz', allow_pickle=False) as fixture:
            self.assertEqual(len(fixture['observations']), 1000)
            for obs in fixture['observations']:
                np.testing.assert_array_equal(model.predict_chunk(obs), uncached_chunk(model, obs))
        self.assertTrue(verify_parity(DIRECTORY)['passed'])

    def test_temporal_weight_cache_matches_original_all_history_lengths(self):
        model = NumpyACT(DIRECTORY)
        with np.load(DIRECTORY / 'parity-fixtures.npz', allow_pickle=False) as fixture:
            history = []
            for obs in fixture['observations'][:model.chunk_size * 3]:
                history.append(uncached_chunk(model, obs))
                history = history[-model.chunk_size:]
                predictions = np.stack([chunk[age] for age, chunk in enumerate(reversed(history))])
                weights = np.exp(-np.float32(.05) * np.arange(len(history), dtype=np.float32))
                expected = np.clip((predictions * weights[:, None]).sum(0) / weights.sum(), -1, 1)
                np.testing.assert_array_equal(model.predict(obs), expected)

    def test_predictions_and_reset_do_not_mutate_frozen_cache(self):
        model = NumpyACT(DIRECTORY)
        initial = model._initial_value.copy()
        observation = np.zeros(94, dtype=np.float32)
        expected = model.predict(observation)
        model.predict(np.ones(94, dtype=np.float32))
        model.reset()
        np.testing.assert_array_equal(model._initial_value, initial)
        np.testing.assert_array_equal(model.predict(observation), expected)
        for invalid in (np.zeros(93), np.full(94, np.nan), np.full(94, np.inf)):
            with self.assertRaises(ValueError):
                model.predict_chunk(invalid)


if __name__ == '__main__':
    unittest.main()
