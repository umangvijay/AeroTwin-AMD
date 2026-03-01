"""
Integration test for CloudEdgeSimulator to verify end-to-end functionality.
"""

import unittest
from main import CloudEdgeSimulator


class TestCloudSimulatorIntegration(unittest.TestCase):
    """Integration tests for CloudEdgeSimulator."""
    
    def test_generate_telemetry_complete_workflow(self):
        """Test that generate_telemetry produces valid data with all fields."""
        simulator = CloudEdgeSimulator(zone_id="TestZone", demo_spike_interval=100)
        
        # Generate multiple telemetry samples
        for _ in range(10):
            telemetry = simulator.generate_telemetry()
            
            # Verify all required fields are present
            self.assertIn('zone', telemetry)
            self.assertIn('occupancy', telemetry)
            self.assertIn('ai_mode', telemetry)
            self.assertIn('hardware_cpu', telemetry)
            self.assertIn('inference_latency', telemetry)
            self.assertIn('system_log', telemetry)
            self.assertIn('cycle', telemetry)
            
            # Verify data types and ranges
            self.assertEqual(telemetry['zone'], "TestZone")
            self.assertIsInstance(telemetry['occupancy'], int)
            self.assertGreaterEqual(telemetry['occupancy'], 0)
            self.assertLessEqual(telemetry['occupancy'], 50)
            
            self.assertIn(telemetry['ai_mode'], ['eco', 'auto', 'performance', 'standby'])
            
            self.assertIsInstance(telemetry['hardware_cpu'], float)
            self.assertGreaterEqual(telemetry['hardware_cpu'], 5)
            self.assertLessEqual(telemetry['hardware_cpu'], 95)
            
            self.assertIsInstance(telemetry['inference_latency'], float)
            self.assertGreater(telemetry['inference_latency'], 0)
            
            self.assertIsInstance(telemetry['system_log'], str)
            self.assertGreater(len(telemetry['system_log']), 0)
            
            self.assertIsInstance(telemetry['cycle'], int)
            self.assertGreater(telemetry['cycle'], 0)
    
    def test_latency_cpu_correlation_in_telemetry(self):
        """Test that latency and CPU show realistic correlation in generated telemetry."""
        simulator = CloudEdgeSimulator(demo_spike_interval=1000)
        
        # Collect samples
        samples = [simulator.generate_telemetry() for _ in range(50)]
        
        # Group by AI mode
        eco_samples = [s for s in samples if s['ai_mode'] == 'eco']
        performance_samples = [s for s in samples if s['ai_mode'] == 'performance']
        
        if eco_samples and performance_samples:
            # ECO mode should have lower average latency than PERFORMANCE
            avg_eco_latency = sum(s['inference_latency'] for s in eco_samples) / len(eco_samples)
            avg_perf_latency = sum(s['inference_latency'] for s in performance_samples) / len(performance_samples)
            
            self.assertLess(avg_eco_latency, avg_perf_latency)
            
            # ECO mode should have lower average CPU than PERFORMANCE
            avg_eco_cpu = sum(s['hardware_cpu'] for s in eco_samples) / len(eco_samples)
            avg_perf_cpu = sum(s['hardware_cpu'] for s in performance_samples) / len(performance_samples)
            
            self.assertLess(avg_eco_cpu, avg_perf_cpu)


if __name__ == '__main__':
    unittest.main()
