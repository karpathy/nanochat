
import unittest
from nanochat.scheduler import get_lr_multiplier

class TestAdamWWarmup(unittest.TestCase):
    def test_warmup(self):
        num_iterations = 100
        warmup_ratio = 0.1
        warmdown_ratio = 0.0
        final_lr_frac = 0.0

        # Step 0
        lrm = get_lr_multiplier(0, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
        self.assertAlmostEqual(lrm, 0.1)

        # Step 4
        lrm = get_lr_multiplier(4, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
        self.assertAlmostEqual(lrm, 0.5)

        # Step 9
        lrm = get_lr_multiplier(9, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
        self.assertAlmostEqual(lrm, 1.0)

        # Step 10
        lrm = get_lr_multiplier(10, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
        self.assertEqual(lrm, 1.0)

    def test_adam_warmup_logic(self):
        # Simulate the logic in base_train.py
        step = 5
        adam_warmup_ratio = 0.1
        muon_warmup_ratio = 0.2
        num_iterations = 100
        warmdown_ratio = 0.0
        final_lr_frac = 0.0

        adam_lrm = get_lr_multiplier(step, adam_warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
        muon_lrm = get_lr_multiplier(step, muon_warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)

        # Adam warmup is 10 steps. Step 5 is 60% of warmup (since (5+1)/10 = 0.6)
        self.assertAlmostEqual(adam_lrm, 0.6)

        # Muon warmup is 20 steps. Step 5 is 30% of warmup (since (5+1)/20 = 0.3)
        self.assertAlmostEqual(muon_lrm, 0.3)

        print(f"Adam LRM: {adam_lrm}, Muon LRM: {muon_lrm}")

if __name__ == '__main__':
    unittest.main()
