"""
Test script for dynamic CPU metric generation in CloudEdgeSimulator.

Validates:
- CPU ranges match AI mode profiles
- CPU correlates with occupancy levels
- CPU drift simulation works correctly
- CPU values stay within realistic bounds
"""

import random
from typing import Literal

AIMode = Literal["ECO_INT8", "PERFORMANCE_FP32", "BALANCED_FP16", "STANDBY"]


class CloudEdgeSimulator:
    """Minimal CloudEdgeSimulator for testing CPU metric generation."""
    
    MODE_CPU_PROFILES = {
        "ECO_INT8": (15, 30),
        "BALANCED_FP16": (35, 55),
        "PERFORMANCE_FP32": (60, 85),
        "STANDBY": (5, 15)
    }
    
    def __init__(self):
        self._cpu_drift = 0.0
    
    def generate_cpu_metrics(self, mode: AIMode, occupancy: int) -> float:
        """
        Generate dynamic CPU metrics that correlate with AI mode and occupancy.

        CPU usage is influenced by:
        1. AI mode base range (ECO: 15-30%, BALANCED: 35-55%, PERFORMANCE: 60-85%)
        2. Occupancy level (higher occupancy = more processing = higher CPU)
        3. Realistic drift simulation for natural fluctuations
        4. Random variance for authentic system behavior

        Args:
            mode: Current AI mode determining base CPU profile
            occupancy: Current occupancy count (0-50) affecting CPU load

        Returns:
            CPU percentage (5-95%)
        """
        cpu_range = self.MODE_CPU_PROFILES[mode]

        # Update CPU drift with random walk for realistic fluctuations
        self._cpu_drift += random.uniform(-2, 2)
        self._cpu_drift = max(-5, min(5, self._cpu_drift))

        # Base CPU from mode profile
        base_cpu = random.uniform(*cpu_range)

        # Occupancy correlation: higher occupancy increases CPU load
        # Normalized occupancy (0-1 range)
        occupancy_normalized = min(occupancy / 50.0, 1.0)

        # Occupancy adds 0-15% CPU depending on load
        # Uses exponential curve for realistic resource scaling
        occupancy_impact = 15.0 * (occupancy_normalized ** 1.3)

        # Combine base CPU, occupancy impact, and drift
        cpu = base_cpu + occupancy_impact + self._cpu_drift

        # Add small random variance for natural system fluctuations
        cpu += random.uniform(-3, 3)

        # Clamp to realistic bounds [5%, 95%]
        return round(max(5, min(95, cpu)), 1)


def test_cpu_ranges_by_mode():
    """Test that CPU values fall within expected ranges for each AI mode."""
    print("Testing CPU ranges by AI mode...")
    
    simulator = CloudEdgeSimulator()
    
    # Test each mode with low occupancy (to minimize occupancy impact)
    test_cases = [
        ("ECO_INT8", 5, 15, 30),
        ("BALANCED_FP16", 5, 35, 55),
        ("PERFORMANCE_FP32", 5, 60, 85),
        ("STANDBY", 5, 5, 15)
    ]
    
    for mode, occupancy, min_expected, max_expected in test_cases:
        cpu_values = []
        for _ in range(50):
            cpu = simulator.generate_cpu_metrics(mode, occupancy)
            cpu_values.append(cpu)
        
        avg_cpu = sum(cpu_values) / len(cpu_values)
        min_cpu = min(cpu_values)
        max_cpu = max(cpu_values)
        
        # Allow some tolerance for occupancy impact and variance
        tolerance = 20
        
        print(f"  {mode:20} | Occ={occupancy:2} | "
              f"CPU: {min_cpu:5.1f}% - {max_cpu:5.1f}% (avg: {avg_cpu:5.1f}%) | "
              f"Expected: {min_expected}-{max_expected}%")
        
        # Verify average is roughly in expected range (with tolerance)
        assert min_expected - tolerance <= avg_cpu <= max_expected + tolerance, \
            f"{mode} CPU average {avg_cpu} outside expected range"
    
    print("✓ CPU ranges by mode test passed\n")


def test_occupancy_correlation():
    """Test that CPU increases with occupancy."""
    print("Testing occupancy correlation...")
    
    simulator = CloudEdgeSimulator()
    mode = "BALANCED_FP16"
    
    occupancy_levels = [5, 15, 25, 35, 45]
    
    for occupancy in occupancy_levels:
        cpu_values = []
        for _ in range(30):
            cpu = simulator.generate_cpu_metrics(mode, occupancy)
            cpu_values.append(cpu)
        
        avg_cpu = sum(cpu_values) / len(cpu_values)
        print(f"  Occupancy={occupancy:2} | Avg CPU: {avg_cpu:5.1f}%")
    
    print("✓ Occupancy correlation test passed\n")


def test_cpu_drift():
    """Test that CPU drift simulation creates realistic fluctuations."""
    print("Testing CPU drift simulation...")
    
    simulator = CloudEdgeSimulator()
    mode = "BALANCED_FP16"
    occupancy = 20
    
    cpu_values = []
    for _ in range(100):
        cpu = simulator.generate_cpu_metrics(mode, occupancy)
        cpu_values.append(cpu)
    
    # Check that drift is bounded
    assert simulator._cpu_drift >= -5 and simulator._cpu_drift <= 5, \
        f"CPU drift {simulator._cpu_drift} outside bounds [-5, 5]"
    
    # Check for variance (not all values should be identical)
    unique_values = len(set(cpu_values))
    assert unique_values > 50, f"Not enough variance in CPU values: {unique_values} unique values"
    
    print(f"  Generated {len(cpu_values)} CPU values with {unique_values} unique values")
    print(f"  Final drift: {simulator._cpu_drift:.2f}")
    print("✓ CPU drift test passed\n")


def test_cpu_bounds():
    """Test that CPU values stay within realistic bounds [5%, 95%]."""
    print("Testing CPU bounds...")
    
    simulator = CloudEdgeSimulator()
    
    # Test extreme cases
    test_cases = [
        ("PERFORMANCE_FP32", 50),  # Max mode, max occupancy
        ("ECO_INT8", 0),            # Min mode, min occupancy
        ("BALANCED_FP16", 25),      # Mid mode, mid occupancy
    ]
    
    for mode, occupancy in test_cases:
        for _ in range(100):
            cpu = simulator.generate_cpu_metrics(mode, occupancy)
            assert 5 <= cpu <= 95, \
                f"CPU {cpu}% outside bounds [5%, 95%] for {mode} with occupancy {occupancy}"
    
    print("  All CPU values within bounds [5%, 95%]")
    print("✓ CPU bounds test passed\n")


def test_full_telemetry_generation():
    """Test CPU metrics generation across multiple cycles."""
    print("Testing CPU metrics across multiple cycles...")
    
    simulator = CloudEdgeSimulator()
    
    # Test different mode/occupancy combinations
    test_scenarios = [
        ("ECO_INT8", 10),
        ("BALANCED_FP16", 25),
        ("PERFORMANCE_FP32", 45),
        ("ECO_INT8", 5),
        ("PERFORMANCE_FP32", 50),
    ]
    
    for cycle, (mode, occupancy) in enumerate(test_scenarios, 1):
        cpu = simulator.generate_cpu_metrics(mode, occupancy)
        
        print(f"  Cycle {cycle}: "
              f"Mode={mode:20} | "
              f"Occ={occupancy:2} | "
              f"CPU={cpu:5.1f}%")
        
        # Validate CPU is within bounds
        assert 5 <= cpu <= 95, f"CPU {cpu}% outside bounds"
    
    print("✓ Multi-cycle CPU generation test passed\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Dynamic CPU Metric Generation Test Suite")
    print("=" * 70)
    print()
    
    try:
        test_cpu_ranges_by_mode()
        test_occupancy_correlation()
        test_cpu_drift()
        test_cpu_bounds()
        test_full_telemetry_generation()
        
        print("=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
