"""Frozen ACT inference using NumPy; no Torch or motor interfaces are required.

The CVAE latent is zero at inference, matching ACTPolicy.forward(obs). The
training-only encoder/posterior is intentionally absent. See parity fixtures
exported from the selected PyTorch checkpoint for numerical verification.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np


class NumpyACT:
    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / 'policy.json').read_text())
        weights = directory / 'weights.npz'
        if hashlib.sha256(weights.read_bytes()).hexdigest() != self.metadata['weights_sha256']:
            raise ValueError('ACT weights checksum mismatch')
        expected = dict(schema=1, obs_dim=94, hidden=64, heads=4, layers=2,
                        normalization_epsilon=1e-5, temporal_decay=.05, policy_hz=10)
        if any(self.metadata.get(k) != v for k, v in expected.items()):
            raise ValueError('Unsupported ACT observation/architecture contract')
        self.chunk_size = int(self.metadata['chunk_size'])
        if not 4 <= self.chunk_size <= 32:
            raise ValueError('Invalid ACT chunk length')
        with np.load(weights, allow_pickle=False) as data:
            self.weights = {k: data[k].copy() for k in data.files}
        if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in self.weights.values()):
            raise ValueError('ACT weights must be finite float32 arrays')
        if self.weights['queries'].shape != (1, self.chunk_size, 64):
            raise ValueError('ACT query shape mismatch')
        self.history = deque(maxlen=self.chunk_size)
        # The first self-attention sees only the frozen learned queries. With
        # dropout absent at inference, this entire sublayer is independent of
        # the observation and can be evaluated once without approximating ACT.
        prefix = 'decoder.layers.0.'
        queries = self.weights['queries'][0]
        self._initial_value = self.norm(
            queries + self.attention(queries, queries, prefix + 'self_attn'),
            prefix + 'norm1')
        # Temporal ensemble coefficients depend only on the history length.
        self._temporal_weights = np.exp(
            -np.float32(.05) * np.arange(self.chunk_size, dtype=np.float32))
        self._temporal_sums = np.array([
            self._temporal_weights[:length].sum()
            for length in range(1, self.chunk_size + 1)])

    def reset(self):
        self.history.clear()

    def linear(self, value, name):
        return value @ self.weights[name + '.weight'].T + self.weights[name + '.bias']

    def norm(self, value, name):
        mean = value.mean(axis=-1, keepdims=True)
        variance = np.square(value - mean).mean(axis=-1, keepdims=True)
        normalized = (value - mean) / np.sqrt(variance + np.float32(1e-5))
        return normalized * self.weights[name + '.weight'] + self.weights[name + '.bias']

    def attention(self, query, memory, name):
        weight = self.weights[name + '.in_proj_weight']
        bias = self.weights[name + '.in_proj_bias']
        q = query @ weight[:64].T + bias[:64]
        k = memory @ weight[64:128].T + bias[64:128]
        v = memory @ weight[128:].T + bias[128:]
        q = q.reshape(-1, 4, 16).transpose(1, 0, 2)
        k = k.reshape(-1, 4, 16).transpose(1, 0, 2)
        v = v.reshape(-1, 4, 16).transpose(1, 0, 2)
        scores = (q @ k.transpose(0, 2, 1)) * np.float32(.25)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        attended = (weights @ v).transpose(1, 0, 2).reshape(len(query), 64)
        return self.linear(attended, name + '.out_proj')

    def predict_chunk(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (94,) or not np.isfinite(obs).all():
            raise ValueError('ACT requires 94 finite observation values')
        # ACT uses a zero latent at inference; its embedding is the trained bias.
        memory = np.stack((self.linear(obs, 'obs_embed'), self.weights['latent_embed.bias']))
        value = self._initial_value
        for index in range(2):
            prefix = f'decoder.layers.{index}.'
            if index:
                value = self.norm(value + self.attention(value, value, prefix + 'self_attn'), prefix + 'norm1')
            value = self.norm(value + self.attention(value, memory, prefix + 'multihead_attn'), prefix + 'norm2')
            feedforward = self.linear(np.maximum(self.linear(value, prefix + 'linear1'), 0), prefix + 'linear2')
            value = self.norm(value + feedforward, prefix + 'norm3')
        chunk = np.tanh(self.linear(value, 'head'))
        if chunk.shape != (self.chunk_size, 5) or not np.isfinite(chunk).all():
            raise RuntimeError('ACT returned an invalid action chunk')
        return chunk

    def predict(self, observation):
        self.history.append(self.predict_chunk(observation))
        predictions = np.stack([chunk[age] for age, chunk in enumerate(reversed(self.history))])
        weights = self._temporal_weights[:len(predictions)]
        action = (predictions * weights[:, None]).sum(axis=0) / self._temporal_sums[len(predictions) - 1]
        if not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
            raise RuntimeError('ACT residual exceeds its trained action bounds')
        return np.clip(action, -1, 1)


def verify_parity(directory):
    model = NumpyACT(directory)
    fixture = np.load(Path(directory) / 'parity-fixtures.npz', allow_pickle=False)
    predicted = np.stack([model.predict_chunk(obs) for obs in fixture['observations']])
    error = np.abs(predicted - fixture['chunks'])
    if not np.allclose(predicted, fixture['chunks'], atol=2e-5, rtol=2e-5):
        raise RuntimeError(f'ACT NumPy/PyTorch parity failed: {error.max():.8g}')
    return dict(passed=True, samples=len(predicted), maximum_action_error=float(error.max()),
                rms_action_error=float(np.sqrt(np.mean(error**2))),
                checkpoint_sha256=model.metadata['checkpoint_sha256'], hardware_commands=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    print(json.dumps(verify_parity(args.directory), indent=2))
