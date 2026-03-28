"""
ACE-Step OpenRouter API stress test script

Supports concurrent testing to measure the maximum QPS and performance of the service.

Usage:
    # Basic test - 10 concurrent workers, 100 requests
    python -m openrouter.stress_test

    # Custom parameter test
    python -m openrouter.stress_test --concurrency 50 --requests 500

    # Ramp-up test
    python -m openrouter.stress_test --mode ramp --max-concurrency 100 --step 10

    # Duration-based stress test
    python -m openrouter.stress_test --mode duration --duration 60 --concurrency 20
"""

import argparse
import json
import os
import sys
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import defaultdict
from datetime import datetime
import queue

import requests


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BASE_URL = "https://api.acemusic.ai"
DEFAULT_CONCURRENCY = 4
DEFAULT_TOTAL_REQUESTS = 100


@dataclass
class RequestResult:
    """Result of a single request"""
    success: bool
    status_code: int
    latency: float  # seconds
    error_message: str = ""
    timestamp: float = 0.0


@dataclass
class StressTestStats:
    """Stress test statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies: List[float] = field(default_factory=list)
    errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    status_codes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    start_time: float = 0.0
    end_time: float = 0.0

    def add_result(self, result: RequestResult):
        """Add a request result"""
        self.total_requests += 1
        self.status_codes[result.status_code] += 1

        if result.success:
            self.successful_requests += 1
            self.latencies.append(result.latency)
        else:
            self.failed_requests += 1
            self.errors[result.error_message] += 1

    @property
    def success_rate(self) -> float:
        """Success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def duration(self) -> float:
        """Test duration"""
        return self.end_time - self.start_time

    @property
    def qps(self) -> float:
        """Requests per second (QPS)"""
        if self.duration == 0:
            return 0.0
        return self.total_requests / self.duration

    @property
    def successful_qps(self) -> float:
        """QPS for successful requests"""
        if self.duration == 0:
            return 0.0
        return self.successful_requests / self.duration

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.latencies:
            return {
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
                "median": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)

        return {
            "min": min(sorted_latencies),
            "max": max(sorted_latencies),
            "avg": statistics.mean(sorted_latencies),
            "median": statistics.median(sorted_latencies),
            "p90": sorted_latencies[int(n * 0.90)] if n > 0 else 0.0,
            "p95": sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
        }


class StressTester:
    """Stress tester"""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 300,
        test_type: str = "health",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.test_type = test_type
        self.session = requests.Session()
        self.lock = threading.Lock()
        self.request_counter = 0
        self.live_stats = StressTestStats()

    def get_headers(self) -> dict:
        """Build request headers"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def make_request(self) -> RequestResult:
        """Execute a single request"""
        start_time = time.time()
        timestamp = start_time

        try:
            if self.test_type == "health":
                resp = requests.get(
                    f"{self.base_url}/health",
                    timeout=self.timeout
                )
            elif self.test_type == "models":
                resp = requests.get(
                    f"{self.base_url}/api/v1/models",
                    headers=self.get_headers(),
                    timeout=self.timeout
                )
            elif self.test_type == "generate":
                payload = self._get_generate_payload()
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
            elif self.test_type == "instrumental":
                payload = self._get_instrumental_payload()
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
            else:
                # Default to health
                resp = requests.get(
                    f"{self.base_url}/health",
                    timeout=self.timeout
                )

            latency = time.time() - start_time

            if resp.status_code == 200:
                return RequestResult(
                    success=True,
                    status_code=resp.status_code,
                    latency=latency,
                    timestamp=timestamp
                )
            else:
                return RequestResult(
                    success=False,
                    status_code=resp.status_code,
                    latency=latency,
                    error_message=f"HTTP {resp.status_code}",
                    timestamp=timestamp
                )

        except requests.exceptions.Timeout:
            return RequestResult(
                success=False,
                status_code=0,
                latency=time.time() - start_time,
                error_message="Timeout",
                timestamp=timestamp
            )
        except requests.exceptions.ConnectionError as e:
            return RequestResult(
                success=False,
                status_code=0,
                latency=time.time() - start_time,
                error_message=f"ConnectionError: {str(e)[:50]}",
                timestamp=timestamp
            )
        except Exception as e:
            return RequestResult(
                success=False,
                status_code=0,
                latency=time.time() - start_time,
                error_message=f"{type(e).__name__}: {str(e)[:50]}",
                timestamp=timestamp
            )

    def _get_generate_payload(self) -> dict:
        """Get the payload for a generate request"""
        return {
            "messages": [
                {"role": "user", "content": "Generate an upbeat pop song about summer"}
            ],
            "sample_mode": True,
            "audio_config": {
                "vocal_language": "en",
                "duration": 30,
            },
        }

    def _get_instrumental_payload(self) -> dict:
        """Get the payload for an instrumental request"""
        return {
            "messages": [
                {"role": "user", "content": "<prompt>Epic orchestral cinematic score</prompt>"}
            ],
            "audio_config": {
                "instrumental": True,
                "duration": 30,
            },
        }

    def run_fixed_requests(
        self,
        concurrency: int,
        total_requests: int,
        show_progress: bool = True
    ) -> StressTestStats:
        """Fixed request count mode"""
        stats = StressTestStats()
        stats.start_time = time.time()

        completed = 0
        completed_lock = threading.Lock()

        def worker():
            nonlocal completed
            result = self.make_request()

            with completed_lock:
                completed += 1
                stats.add_result(result)

                if show_progress and completed % 10 == 0:
                    elapsed = time.time() - stats.start_time
                    current_qps = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"\rProgress: {completed}/{total_requests} "
                        f"({completed/total_requests*100:.1f}%) | "
                        f"Success: {stats.successful_requests} | "
                        f"Failed: {stats.failed_requests} | "
                        f"Current QPS: {current_qps:.2f}",
                        end="", flush=True
                    )

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker) for _ in range(total_requests)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"\nWorker thread exception: {e}")

        stats.end_time = time.time()

        if show_progress:
            print()  # newline

        return stats

    def run_duration_based(
        self,
        concurrency: int,
        duration: int,
        show_progress: bool = True
    ) -> StressTestStats:
        """Duration-based mode"""
        stats = StressTestStats()
        stats.start_time = time.time()
        stop_event = threading.Event()

        def worker():
            while not stop_event.is_set():
                result = self.make_request()
                with self.lock:
                    stats.add_result(result)

        # Start worker threads
        threads = []
        for _ in range(concurrency):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)

        # Show progress
        try:
            end_time = time.time() + duration
            while time.time() < end_time:
                elapsed = time.time() - stats.start_time
                remaining = duration - elapsed
                current_qps = stats.total_requests / elapsed if elapsed > 0 else 0

                if show_progress:
                    print(
                        f"\rTime remaining: {remaining:.1f}s | "
                        f"Requests: {stats.total_requests} | "
                        f"Success: {stats.successful_requests} | "
                        f"Failed: {stats.failed_requests} | "
                        f"QPS: {current_qps:.2f}",
                        end="", flush=True
                    )
                time.sleep(0.5)
        finally:
            stop_event.set()

        # Wait for all threads to finish
        for t in threads:
            t.join(timeout=5)

        stats.end_time = time.time()

        if show_progress:
            print()  # newline

        return stats

    def run_ramp_up(
        self,
        max_concurrency: int,
        step: int,
        requests_per_step: int,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Ramp-up mode"""
        results = []

        for concurrency in range(step, max_concurrency + 1, step):
            print(f"\n{'='*60}")
            print(f"Testing concurrency: {concurrency}")
            print("=" * 60)

            stats = self.run_fixed_requests(
                concurrency=concurrency,
                total_requests=requests_per_step,
                show_progress=show_progress
            )

            latency_stats = stats.get_latency_stats()

            result = {
                "concurrency": concurrency,
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "failed_requests": stats.failed_requests,
                "success_rate": stats.success_rate,
                "qps": stats.qps,
                "successful_qps": stats.successful_qps,
                "avg_latency": latency_stats["avg"],
                "p95_latency": latency_stats["p95"],
                "p99_latency": latency_stats["p99"],
            }
            results.append(result)

            self._print_step_summary(result)

            # Brief pause to let the service recover
            time.sleep(2)

        return results


    def _print_step_summary(self, result: Dict[str, Any]):
        """Print summary for a single step"""
        print(f"\nConcurrency {result['concurrency']} test results:")
        print(f"  Total requests: {result['total_requests']}")
        print(f"  Success/Failed: {result['successful_requests']}/{result['failed_requests']}")
        print(f"  Success rate: {result['success_rate']:.2f}%")
        print(f"  QPS: {result['qps']:.2f}")
        print(f"  Successful QPS: {result['successful_qps']:.2f}")
        print(f"  Avg latency: {result['avg_latency']*1000:.2f}ms")
        print(f"  P95 latency: {result['p95_latency']*1000:.2f}ms")
        print(f"  P99 latency: {result['p99_latency']*1000:.2f}ms")


def print_stats(stats: StressTestStats, title: str = "Stress Test Results"):
    """Print statistics"""
    latency_stats = stats.get_latency_stats()

    print("\n")
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)

    print("\n📊 Basic Statistics")
    print("-" * 40)
    print(f"  Total requests:       {stats.total_requests}")
    print(f"  Successful requests:  {stats.successful_requests}")
    print(f"  Failed requests:      {stats.failed_requests}")
    print(f"  Success rate:         {stats.success_rate:.2f}%")

    print("\n⏱️ Time Statistics")
    print("-" * 40)
    print(f"  Test duration:        {stats.duration:.2f} seconds")
    print(f"  Total QPS:            {stats.qps:.2f}")
    print(f"  Successful QPS:       {stats.successful_qps:.2f}")

    print("\n📈 Latency Statistics (milliseconds)")
    print("-" * 40)
    print(f"  Min latency:          {latency_stats['min']*1000:.2f}ms")
    print(f"  Max latency:          {latency_stats['max']*1000:.2f}ms")
    print(f"  Avg latency:          {latency_stats['avg']*1000:.2f}ms")
    print(f"  Median latency:       {latency_stats['median']*1000:.2f}ms")
    print(f"  P90 latency:          {latency_stats['p90']*1000:.2f}ms")
    print(f"  P95 latency:          {latency_stats['p95']*1000:.2f}ms")
    print(f"  P99 latency:          {latency_stats['p99']*1000:.2f}ms")

    if stats.status_codes:
        print("\n📋 Status Code Distribution")
        print("-" * 40)
        for code, count in sorted(stats.status_codes.items()):
            percentage = (count / stats.total_requests) * 100
            print(f"  {code}:  {count:>8} ({percentage:.1f}%)")

    if stats.errors:
        print("\n❌ Error Distribution (Top 10)")
        print("-" * 40)
        sorted_errors = sorted(stats.errors.items(), key=lambda x: x[1], reverse=True)[:10]
        for error, count in sorted_errors:
            percentage = (count / stats.total_requests) * 100
            print(f"  {error[:50]}: {count} ({percentage:.1f}%)")

    print("\n" + "=" * 70)


def print_ramp_summary(results: List[Dict[str, Any]]):
    """Print summary for a ramp-up test"""
    print("\n")
    print("=" * 90)
    print(" Ramp-Up Test Summary")
    print("=" * 90)

    # Header
    print(f"\n{'Concurrency':>8} | {'Requests':>8} | {'Success%':>8} | {'QPS':>10} | {'Succ.QPS':>10} | {'Avg Lat':>10} | {'P99 Lat':>10}")
    print("-" * 90)

    # Data rows
    for r in results:
        print(
            f"{r['concurrency']:>8} | "
            f"{r['total_requests']:>8} | "
            f"{r['success_rate']:>7.1f}% | "
            f"{r['qps']:>10.2f} | "
            f"{r['successful_qps']:>10.2f} | "
            f"{r['avg_latency']*1000:>9.1f}ms | "
            f"{r['p99_latency']*1000:>9.1f}ms"
        )

    print("-" * 90)

    # Find best QPS
    best_qps = max(results, key=lambda x: x['successful_qps'])
    print(f"\n🏆 Best successful QPS: {best_qps['successful_qps']:.2f} (concurrency: {best_qps['concurrency']})")

    # Find latency bottleneck (point where P99 latency starts rising sharply)
    for i in range(1, len(results)):
        if results[i]['p99_latency'] > results[i-1]['p99_latency'] * 2:
            print(f"⚠️  Latency bottleneck: concurrency {results[i]['concurrency']} (P99 latency starts rising sharply)")
            break

    # Find point where error rate starts increasing
    for i, r in enumerate(results):
        if r['success_rate'] < 99:
            print(f"⚠️  Stability degradation point: concurrency {r['concurrency']} (success rate: {r['success_rate']:.1f}%)")
            break

    print()


def main():
    parser = argparse.ArgumentParser(
        description="ACE-Step OpenRouter API stress test tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stress test the health check endpoint
  python -m openrouter.stress_test --test health --concurrency 100 --requests 1000

  # Stress test the model list endpoint
  python -m openrouter.stress_test --test models --concurrency 50 --requests 500

  # Stress test the music generation endpoint (note: generation requests are slow)
  python -m openrouter.stress_test --test generate --concurrency 5 --requests 20

  # Ramp-up test
  python -m openrouter.stress_test --mode ramp --max-concurrency 100 --step 10

  # Duration-based stress test (60 seconds)
  python -m openrouter.stress_test --mode duration --duration 60 --concurrency 50
        """
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        help=f"API base URL (default: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="API key (optional)"
    )
    parser.add_argument(
        "--test",
        choices=["health", "models", "generate", "instrumental"],
        default="health",
        help="Endpoint type to test (default: health)"
    )
    parser.add_argument(
        "--mode",
        choices=["fixed", "duration", "ramp"],
        default="fixed",
        help="Test mode: fixed=fixed request count, duration=time-based, ramp=ramp-up (default: fixed)"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrency level (default: {DEFAULT_CONCURRENCY})"
    )
    parser.add_argument(
        "--requests", "-n",
        type=int,
        default=DEFAULT_TOTAL_REQUESTS,
        help=f"Total number of requests (fixed mode) (default: {DEFAULT_TOTAL_REQUESTS})"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="Test duration in seconds (duration mode) (default: 60)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=100,
        help="Maximum concurrency level (ramp mode) (default: 100)"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Concurrency step size (ramp mode) (default: 10)"
    )
    parser.add_argument(
        "--requests-per-step",
        type=int,
        default=100,
        help="Number of requests per step (ramp mode) (default: 100)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to a JSON file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Print configuration info
    print("=" * 70)
    print(" ACE-Step OpenRouter API Stress Test")
    print("=" * 70)
    print(f"  Time:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Base URL:         {args.base_url}")
    print(f"  API Key:          {'set' if args.api_key else 'not set'}")
    print(f"  Test endpoint:    {args.test}")
    print(f"  Test mode:        {args.mode}")

    if args.mode == "fixed":
        print(f"  Concurrency:      {args.concurrency}")
        print(f"  Total requests:   {args.requests}")
    elif args.mode == "duration":
        print(f"  Concurrency:      {args.concurrency}")
        print(f"  Duration:         {args.duration} seconds")
    elif args.mode == "ramp":
        print(f"  Max concurrency:  {args.max_concurrency}")
        print(f"  Step size:        {args.step}")
        print(f"  Requests/step:    {args.requests_per_step}")

    print(f"  Request timeout:  {args.timeout} seconds")
    print("=" * 70)

    # Create tester
    tester = StressTester(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        test_type=args.test
    )

    # Run tests
    try:
        if args.mode == "fixed":
            print(f"\nStarting fixed request count test (concurrency: {args.concurrency}, requests: {args.requests})...\n")
            stats = tester.run_fixed_requests(
                concurrency=args.concurrency,
                total_requests=args.requests,
                show_progress=not args.quiet
            )
            print_stats(stats, f"Stress Test Results - {args.test.upper()} endpoint")

            # Save results
            if args.output:
                latency_stats = stats.get_latency_stats()
                output_data = {
                    "test_type": args.test,
                    "mode": args.mode,
                    "concurrency": args.concurrency,
                    "total_requests": stats.total_requests,
                    "successful_requests": stats.successful_requests,
                    "failed_requests": stats.failed_requests,
                    "success_rate": stats.success_rate,
                    "duration": stats.duration,
                    "qps": stats.qps,
                    "successful_qps": stats.successful_qps,
                    "latency": latency_stats,
                    "status_codes": dict(stats.status_codes),
                    "errors": dict(stats.errors),
                    "timestamp": datetime.now().isoformat()
                }
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"\nResults saved to: {args.output}")

        elif args.mode == "duration":
            print(f"\nStarting duration-based test (concurrency: {args.concurrency}, duration: {args.duration}s)...\n")
            stats = tester.run_duration_based(
                concurrency=args.concurrency,
                duration=args.duration,
                show_progress=not args.quiet
            )
            print_stats(stats, f"Stress Test Results - {args.test.upper()} endpoint ({args.duration}s)")

        elif args.mode == "ramp":
            print(f"\nStarting ramp-up test (max concurrency: {args.max_concurrency}, step: {args.step})...\n")
            results = tester.run_ramp_up(
                max_concurrency=args.max_concurrency,
                step=args.step,
                requests_per_step=args.requests_per_step,
                show_progress=not args.quiet
            )
            print_ramp_summary(results)

            # Save results
            if args.output:
                output_data = {
                    "test_type": args.test,
                    "mode": args.mode,
                    "max_concurrency": args.max_concurrency,
                    "step": args.step,
                    "requests_per_step": args.requests_per_step,
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                }
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"Results saved to: {args.output}")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
