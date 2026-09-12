import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import TensorDataset

from models.SemConsFL import (ClientSemanticBank, FrozenMeasurementHead,
    LocalUpdateSemConsFL, SemConsConfig, SemConsServer, compute_local_delta,
    covered_median)


class TinyNet(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.conv = nn.Conv2d(3, width, 1)
        self.classifier = nn.Linear(width, 2)

    def forward(self, x):
        rep = self.conv(x).mean((2, 3))
        return {"representation": rep, "output": self.classifier(rep)}


class SemConsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.args = SimpleNamespace(device=torch.device('cpu'), seed=1,
            num_classes=2, num_users=4, model='resnet', local_bs=4, local_ep=2,
            optimizer='sgd', lr=.1, lr_decay=.998, momentum=.5, weight_decay=1e-4)
        self.cfg = SemConsConfig(emb_dim=2)
        self.server = SemConsServer(self.args, [TinyNet(2), TinyNet(4)], self.cfg)

    def test_covered_median_excludes_missing_coordinates(self):
        ref, mask = covered_median([torch.tensor([2.]), torch.tensor([4., 8.])], (3,))
        torch.testing.assert_close(ref, torch.tensor([3., 8., 0.]))
        self.assertEqual(mask.tolist(), [True, True, False])

    def test_absolute_sample_weights_no_coverage_renormalization(self):
        result = self.server._apply_updates({'w': torch.zeros(2)},
            [{'w': torch.tensor([1.])}, {'w': torch.tensor([-1., 2.])}], [.5, .5], [.9, .1])
        torch.testing.assert_close(result['w'], torch.tensor([.4, .1]))

    def test_covered_median_matches_coordinate_reference(self):
        deltas = [torch.randn(*shape) for shape in [(2, 3), (4, 2), (3, 4)]]
        ref, mask = covered_median(deltas, (5, 5))
        for i in range(5):
            for j in range(5):
                values = [d[i, j] for d in deltas if i < d.shape[0] and j < d.shape[1]]
                self.assertEqual(bool(mask[i, j]), bool(values))
                if values:
                    torch.testing.assert_close(ref[i, j], torch.quantile(torch.stack(values), .5))

    def test_protective_step_and_zero_step(self):
        state = {'w': torch.ones(2), 'counter': torch.tensor(3)}
        result = self.server._apply_updates(state, [{'w': torch.ones(2)}], [1.], [.01])
        torch.testing.assert_close(result['w'], torch.full((2,), 1.05))
        self.assertTrue(self.server.last_stats['protective_update'])
        result = self.server._apply_updates(state, [{'w': torch.ones(2)}], [1.], [0.])
        for name in state:
            torch.testing.assert_close(result[name], state[name])

    def test_memory_uses_previous_anchor_and_deduplicates(self):
        p = {0: torch.tensor([1., 0.])}
        self.server._memory_pass(0, [0, 0], [p, p], [{0: 5}, {0: 5}])
        self.assertEqual(self.server.evidence_history[0], [])
        opposite = {0: -p[0]}
        beta, _ = self.server._memory_pass(1, [0, 0], [opposite, opposite], [{0: 5}, {0: 5}])
        self.assertEqual(self.server.evidence_history[0], [(1, -1.)])
        self.assertEqual(beta[0], .05)
        self.assertTrue(self.server.last_stats['trusted_fallback'])

    def test_window_is_rounds_not_participations(self):
        self.assertEqual(self.server._append_and_compute_reliability(0, 0, 1.), 0.)
        self.assertEqual(self.server._append_and_compute_reliability(0, 4, 1.), 1.)
        self.server._append_and_compute_reliability(0, 4, 1.)
        self.assertEqual(len(self.server.evidence_history[0]), 2)
        self.assertEqual(self.server._append_and_compute_reliability(0, 9, 1.), 0.)

    def test_direction_sign_and_unique_reference_requirement(self):
        name = 'conv.weight'
        state = {name: torch.zeros(4, 3, 1, 1)}
        deltas = [{name: torch.ones(4, 3, 1, 1)} for _ in range(3)]
        deltas.append({name: -torch.ones(4, 3, 1, 1)})
        g = self.server._directional_gates([0, 1, 2, 3], deltas,
            dict.fromkeys(range(4), 1.), {0, 1, 2}, state)
        self.assertAlmostEqual(g[0], 1.)
        self.assertEqual(g[3], 0.)
        g = self.server._directional_gates([0, 0, 0], deltas[:3], {0: .2}, {0}, state)
        self.assertEqual(g, [.2, .2, .2])

    def test_delta_is_relative_to_exact_dispatch(self):
        d = compute_local_delta({'w': torch.tensor([4.])}, {'w': torch.tensor([3.])})
        torch.testing.assert_close(d['w'], torch.ones(1))

    def test_private_bank_head_and_rng(self):
        state = torch.get_rng_state().clone()
        head = FrozenMeasurementHead(4, 2)
        bank = ClientSemanticBank()
        adapter, classifier = bank.load(0, 4, 4, 2, 2, 'cpu')
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        x = torch.ones(1, 4, requires_grad=True)
        head(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        with torch.no_grad():
            classifier.bias.fill_(2.)
        bank.save(0, 4, adapter, classifier)
        _, restored = bank.load(0, 2, 4, 2, 2, 'cpu')
        torch.testing.assert_close(restored.bias, classifier.bias)
        _, other = bank.load(1, 4, 4, 2, 2, 'cpu')
        torch.testing.assert_close(other.bias, torch.zeros(2))

    def test_local_training_prototypes_and_restoration(self):
        net = TinyNet()
        before = {k: v.clone() for k, v in net.state_dict().items()}
        dataset = TensorDataset(torch.randn(12, 3, 4, 4), torch.tensor([0]*6 + [1]*6))
        local = LocalUpdateSemConsFL(self.args, dataset, range(12), 0)
        bank = ClientSemanticBank()
        head_before = self.server.head.proj.weight.clone()
        state, protos, counts = local.train(0, net, bank, self.server.head, 4, self.cfg)
        self.assertEqual(state.keys(), before.keys())
        self.assertFalse(torch.equal(before['conv.weight'], state['conv.weight']))
        self.assertFalse(torch.equal(bank.adapter_states[(0, 4)]['weight'], torch.eye(4)))
        torch.testing.assert_close(head_before, self.server.head.proj.weight)
        self.assertEqual(counts, {0: 6, 1: 6})
        for p in protos.values():
            self.assertAlmostEqual(p.norm().item(), 1., places=5)
        adapter, _ = bank.load(0, 4, 4, 2, 2, 'cpu')
        rng = torch.get_rng_state().clone()
        local._prototypes(0, net, adapter, self.server.head, self.cfg)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_sub_batch_client_preserves_zero_optimizer_steps(self):
        self.args.local_bs = 50
        net = TinyNet()
        before = {k: v.clone() for k, v in net.state_dict().items()}
        dataset = TensorDataset(torch.randn(6, 3, 4, 4), torch.zeros(6, dtype=torch.long))
        local = LocalUpdateSemConsFL(self.args, dataset, range(6), 0)
        state, protos, counts = local.train(0, net, ClientSemanticBank(), self.server.head, 4, self.cfg)
        self.assertEqual(local.stats['steps'], 0)
        for name in before:
            torch.testing.assert_close(state[name], before[name])
        self.assertEqual(counts, {0: 6})
        self.assertEqual(set(protos), {0})


if __name__ == '__main__':
    unittest.main()
