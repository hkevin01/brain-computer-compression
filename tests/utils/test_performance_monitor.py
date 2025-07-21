from src.utils import performance_monitor
import time

def test_measure_latency():
    def dummy():
        time.sleep(0.01)
    latency = performance_monitor.measure_latency(dummy)
    assert latency >= 10

def test_get_throughput():
    throughput = performance_monitor.get_throughput(1000, 100)
    assert throughput == 10000

def test_get_memory_usage():
    mem = performance_monitor.get_memory_usage()
    assert mem > 0
