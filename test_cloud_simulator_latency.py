"""
Unit tests for CloudEdgeSimulator dynamic inference latency calculation.
Tests that latency is calculated based on AI mode and CPU load.
"""

import unittest
from main import CloudEdgeSimulator


class TestCloudSimulatorDynamicLatency(unittest.TestCase):
    """Test CloudEdgeSimulator dynamic inference latency calculation."""
    
    def setUp(self):
        """Set up test simulator."""
        self.simulator = CloudEdgeSimulator()
    
    def test_latency_increases_with_cpu_load(self):
        """Test that latency increases as CPU load increases."""
        mode = "BALANCED_FP16"
        
        # Low CPU load
        low_cpu_latency = self.simulator.generate_latency(mode, cpu_percent=10.0)
        
        # High CPU load
        high_cpu_latency = self.simulator.generate_latency(mode, cpu_percent=90.0)
        
        # High CPU should result in higher latency
        self.assertGreater(high_cpu_latency, low_cpu_latency)
    
    def test_latency_within_mode_ranges(self):
        """Test that latency stays within expected ranges for each mode."""
        test_cases = [
            ("ECO_INT8", 8.0, 30.0),  # Base: 8-15ms, max with load: ~30ms
            ("BALANCED_FP16", 15.0, 50.0),  # Base: 15-25ms, max with load: ~50ms
            ("PERFORMANCE_FP32", 25.0, 90.0),  # Base: 25-45ms, max with load: ~90ms
            ("STANDBY", 50.0, 200.0),  # Base: 50-100ms, max with load: ~200ms
        ]
        
        for mode, min_expected, max_expected in test_cases:
            with self.subTest(mode=mode):
                # Test at various CPU loads
                for cpu_percent in [0, 25, 50, 75, 100]:
                    latency = self.simulator.generate_latency(mode, cpu_percent)
                    
                    # Verify latency is within reasonable bounds
                    self.assertGreaterEqual(latency, min_expected * 0.9)  # Allow 10% margin
                    self.assertLessEqual(latency, max_expected * 1.1)  # Allow 10% margin
    
    def test_eco_mode_faster_than_performance(self):
        """Test that ECO_INT8 mode has lower latency than PERFORMANCE_FP32."""
        cpu_percent = 50.0
        
        eco_latency = self.simulator.generate_latency("ECO_INT8", cpu_percent)
        performance_latency = self.simulator.generate_latency("PERFORMANCE_FP32", cpu_percent)
        
        # ECO mode should be faster (lower latency)
        self.assertLess(eco_latency, performance_latency)
    
    def test_latency_correlation_with_load(self):
        """Test that latency shows clear correlation with CPU load."""
        mode = "BALANCED_FP16"
        
        # Collect latencies at different CPU loads
        latencies = []
        cpu_loads = [10, 30, 50, 70, 90]
        
        for cpu in cpu_loads:
            # Average over multiple samples to reduce randomness
            samples = [self.simulator.generate_latency(mode, cpu) for _ in range(10)]
            avg_latency = sum(samples) / len(samples)
            latencies.append(avg_latency)
        
        # Verify general upward trend (allowing for some variance)
        # At least the last latency should be higher than the first
        self.assertGreater(latencies[-1], latencies[0])
        
        # Verify middle values show progression
        self.assertGreater(latencies[2], latencies[0])  # 50% > 10%
        self.assertGreater(latencies[4], latencies[2])  # 90% > 50%
    
    def test_generate_telemetry_includes_latency(self):
        """Test that generate_telemetry includes latency in output."""
        telemetry = self.simulator.generate_telemetry()
        
        # Verify telemetry structure includes latency
        self.assertIn('inference_latency', telemetry)
        self.assertIsNotNone(telemetry['inference_latency'])
        self.assertGreater(telemetry['inference_latency'], 0)
    
    def test_latency_correlates_with_cpu_in_telemetry(self):
        """Test that latency in telemetry correlates with CPU metrics."""
        # Generate multiple telemetry samples
        samples = [self.simulator.generate_telemetry() for _ in range(20)]
        
        # Find samples with low and high CPU
        low_cpu_samples = [s for s in samples if s['hardware_cpu'] < 30]
        high_cpu_samples = [s for s in samples if s['hardware_cpu'] > 70]
        
        if low_cpu_samples and high_cpu_samples:
            # Calculate average latencies
            avg_low_latency = sum(s['inference_latency'] for s in low_cpu_samples) / len(low_cpu_samples)
            avg_high_latency = sum(s['inference_latency'] for s in high_cpu_samples) / len(high_cpu_samples)
            
            # High CPU should generally result in higher latency
            self.assertGreater(avg_high_latency, avg_low_latency)


if __name__ == '__main__':
    unittest.main()
