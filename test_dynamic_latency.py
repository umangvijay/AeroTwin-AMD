"""
Unit tests for dynamic inference latency calculation.
Tests that latency is calculated based on AI mode and system load.
"""

import unittest
from edge_simulator import EdgeVisionPipeline


class TestDynamicLatencyCalculation(unittest.TestCase):
    """Test dynamic inference latency calculation."""
    
    def setUp(self):
        """Set up test pipeline."""
        self.pipeline = EdgeVisionPipeline()
    
    def test_latency_increases_with_cpu_load(self):
        """Test that latency increases as CPU load increases."""
        mode = "BALANCED_FP16"
        
        # Low CPU load
        low_cpu_latency = self.pipeline._calculate_dynamic_latency(mode, cpu_percent=10.0)
        
        # High CPU load
        high_cpu_latency = self.pipeline._calculate_dynamic_latency(mode, cpu_percent=90.0)
        
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
                    latency = self.pipeline._calculate_dynamic_latency(mode, cpu_percent)
                    
                    # Verify latency is within reasonable bounds
                    self.assertGreaterEqual(latency, min_expected * 0.9)  # Allow 10% margin
                    self.assertLessEqual(latency, max_expected * 1.1)  # Allow 10% margin
    
    def test_eco_mode_faster_than_performance(self):
        """Test that ECO_INT8 mode has lower latency than PERFORMANCE_FP32."""
        cpu_percent = 50.0
        
        eco_latency = self.pipeline._calculate_dynamic_latency("ECO_INT8", cpu_percent)
        performance_latency = self.pipeline._calculate_dynamic_latency("PERFORMANCE_FP32", cpu_percent)
        
        # ECO mode should be faster (lower latency)
        self.assertLess(eco_latency, performance_latency)
    
    def test_run_inference_accepts_cpu_percent(self):
        """Test that run_inference accepts cpu_percent parameter."""
        # Should not raise an exception
        result = self.pipeline.run_inference("BALANCED_FP16", cpu_percent=45.0)
        
        # Verify result structure
        self.assertIsNotNone(result.latency_ms)
        self.assertGreater(result.latency_ms, 0)
        self.assertEqual(result.mode, "BALANCED_FP16")
        self.assertEqual(result.quantization, "FP16")
    
    def test_run_inference_default_cpu_percent(self):
        """Test that run_inference works with default cpu_percent."""
        # Should use default value of 50.0
        result = self.pipeline.run_inference("ECO_INT8")
        
        # Verify result structure
        self.assertIsNotNone(result.latency_ms)
        self.assertGreater(result.latency_ms, 0)
        self.assertEqual(result.mode, "ECO_INT8")
        self.assertEqual(result.quantization, "INT8")
    
    def test_latency_correlation_with_load(self):
        """Test that latency shows clear correlation with CPU load."""
        mode = "BALANCED_FP16"
        
        # Collect latencies at different CPU loads
        latencies = []
        cpu_loads = [10, 30, 50, 70, 90]
        
        for cpu in cpu_loads:
            # Average over multiple samples to reduce randomness
            samples = [self.pipeline._calculate_dynamic_latency(mode, cpu) for _ in range(10)]
            avg_latency = sum(samples) / len(samples)
            latencies.append(avg_latency)
        
        # Verify general upward trend (allowing for some variance)
        # At least the last latency should be higher than the first
        self.assertGreater(latencies[-1], latencies[0])
        
        # Verify middle values show progression
        self.assertGreater(latencies[2], latencies[0])  # 50% > 10%
        self.assertGreater(latencies[4], latencies[2])  # 90% > 50%


if __name__ == '__main__':
    unittest.main()
