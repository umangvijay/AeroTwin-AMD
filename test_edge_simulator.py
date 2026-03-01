"""
Unit tests for edge_simulator.py hardware metrics collection.
Tests both real psutil metrics and graceful fallback when psutil is unavailable.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock


class TestHardwareMonitor(unittest.TestCase):
    """Test hardware metrics collection with and without psutil."""
    
    def test_hardware_monitor_with_psutil(self):
        """Test that HardwareMonitor collects real metrics when psutil is available."""
        # Import after ensuring psutil is available
        import edge_simulator
        
        if not edge_simulator.PSUTIL_AVAILABLE:
            self.skipTest("psutil not available")
        
        monitor = edge_simulator.HardwareMonitor()
        metrics = monitor.get_metrics()
        
        # Verify metrics structure
        self.assertIsNotNone(metrics.cpu_percent)
        self.assertIsNotNone(metrics.memory_percent)
        self.assertIsNotNone(metrics.memory_used_gb)
        self.assertIsNotNone(metrics.memory_total_gb)
        self.assertIsNotNone(metrics.cpu_cores)
        
        # Verify reasonable ranges
        self.assertGreaterEqual(metrics.cpu_percent, 0.0)
        self.assertLessEqual(metrics.cpu_percent, 100.0)
        self.assertGreaterEqual(metrics.memory_percent, 0.0)
        self.assertLessEqual(metrics.memory_percent, 100.0)
        self.assertGreater(metrics.cpu_cores, 0)
    
    def test_hardware_monitor_fallback_without_psutil(self):
        """Test that HardwareMonitor provides simulated metrics when psutil is unavailable."""
        # Mock psutil as unavailable
        with patch.dict('sys.modules', {'psutil': None}):
            # Force reimport to trigger the ImportError path
            import importlib
            import edge_simulator
            
            # Temporarily set PSUTIL_AVAILABLE to False
            original_value = edge_simulator.PSUTIL_AVAILABLE
            edge_simulator.PSUTIL_AVAILABLE = False
            
            try:
                monitor = edge_simulator.HardwareMonitor()
                metrics = monitor.get_metrics()
                
                # Verify metrics structure exists (simulated)
                self.assertIsNotNone(metrics.cpu_percent)
                self.assertIsNotNone(metrics.memory_percent)
                self.assertIsNotNone(metrics.memory_used_gb)
                self.assertIsNotNone(metrics.memory_total_gb)
                self.assertIsNotNone(metrics.cpu_cores)
                
                # Verify simulated values are in reasonable ranges
                self.assertGreaterEqual(metrics.cpu_percent, 0.0)
                self.assertLessEqual(metrics.cpu_percent, 100.0)
                self.assertGreaterEqual(metrics.memory_percent, 0.0)
                self.assertLessEqual(metrics.memory_percent, 100.0)
                self.assertEqual(metrics.cpu_cores, 8)  # Simulated default
                
            finally:
                # Restore original value
                edge_simulator.PSUTIL_AVAILABLE = original_value
    
    def test_cpu_frequency_graceful_fallback(self):
        """Test that CPU frequency collection handles exceptions gracefully."""
        import edge_simulator
        
        if not edge_simulator.PSUTIL_AVAILABLE:
            self.skipTest("psutil not available")
        
        monitor = edge_simulator.HardwareMonitor()
        metrics = monitor.get_metrics()
        
        # cpu_freq_mhz can be None if not supported on the platform
        # This should not raise an exception
        self.assertTrue(
            metrics.cpu_freq_mhz is None or isinstance(metrics.cpu_freq_mhz, (int, float))
        )


class TestSystemMetrics(unittest.TestCase):
    """Test SystemMetrics dataclass."""
    
    def test_system_metrics_creation(self):
        """Test that SystemMetrics can be created with all required fields."""
        import edge_simulator
        
        metrics = edge_simulator.SystemMetrics(
            cpu_percent=45.5,
            memory_percent=60.2,
            memory_used_gb=8.5,
            memory_total_gb=16.0,
            cpu_freq_mhz=3200.0,
            cpu_cores=8
        )
        
        self.assertEqual(metrics.cpu_percent, 45.5)
        self.assertEqual(metrics.memory_percent, 60.2)
        self.assertEqual(metrics.memory_used_gb, 8.5)
        self.assertEqual(metrics.memory_total_gb, 16.0)
        self.assertEqual(metrics.cpu_freq_mhz, 3200.0)
        self.assertEqual(metrics.cpu_cores, 8)
    
    def test_system_metrics_with_none_cpu_freq(self):
        """Test that SystemMetrics handles None cpu_freq_mhz."""
        import edge_simulator
        
        metrics = edge_simulator.SystemMetrics(
            cpu_percent=45.5,
            memory_percent=60.2,
            memory_used_gb=8.5,
            memory_total_gb=16.0,
            cpu_freq_mhz=None,
            cpu_cores=8
        )
        
        self.assertIsNone(metrics.cpu_freq_mhz)


if __name__ == '__main__':
    unittest.main()
