import time
from datetime import datetime, timedelta
from collections import deque
import threading

class RateLimiter:
    """Thread-safe rate limiter with better error handling."""
    
    def __init__(self, max_requests_per_minute=450, max_tokens_per_minute=180000):
        self.max_rpm = max_requests_per_minute
        self.max_tpm = max_tokens_per_minute
        
        self.request_times = deque()
        self.token_counts = deque()
        
        self.lock = threading.Lock()
        
        # Track consecutive failures
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10  # Stop after 10 failures in a row
        
        print(f"Rate limiter initialized:")
        print(f"  Max requests/min: {self.max_rpm}")
        print(f"  Max tokens/min: {self.max_tpm}")
    
    def wait_if_needed(self, estimated_tokens=100):
        """Wait if necessary to stay within rate limits."""
        with self.lock:
            # Check for too many consecutive failures
            if self.consecutive_failures >= self.max_consecutive_failures:
                raise Exception(f"Too many consecutive failures ({self.consecutive_failures}). Stopping to prevent runaway requests.")
            
            now = datetime.now()
            one_minute_ago = now - timedelta(minutes=1)
            
            # Remove requests older than 1 minute
            while self.request_times and self.request_times[0] < one_minute_ago:
                self.request_times.popleft()
                self.token_counts.popleft()
            
            # Check if we need to wait for request limit
            if len(self.request_times) >= self.max_rpm:
                sleep_time = (self.request_times[0] - one_minute_ago).total_seconds() + 1
                if sleep_time > 0:
                    print(f"  ⏸ Rate limit: waiting {sleep_time:.1f}s for RPM...")
                    time.sleep(sleep_time)
                    return self.wait_if_needed(estimated_tokens)
            
            # Check if we need to wait for token limit
            total_tokens = sum(self.token_counts)
            if total_tokens + estimated_tokens > self.max_tpm:
                sleep_time = (self.request_times[0] - one_minute_ago).total_seconds() + 1
                if sleep_time > 0:
                    print(f"  ⏸ Rate limit: waiting {sleep_time:.1f}s for TPM...")
                    time.sleep(sleep_time)
                    return self.wait_if_needed(estimated_tokens)
            
            # Record this request
            self.request_times.append(now)
            self.token_counts.append(estimated_tokens)
    
    def record_success(self):
        """Record a successful request."""
        self.consecutive_failures = 0
    
    def record_failure(self):
        """Record a failed request."""
        self.consecutive_failures += 1
    
    def get_current_usage(self):
        """Get current usage stats."""
        with self.lock:
            now = datetime.now()
            one_minute_ago = now - timedelta(minutes=1)
            
            # Remove old entries
            while self.request_times and self.request_times[0] < one_minute_ago:
                self.request_times.popleft()
                self.token_counts.popleft()
            
            return {
                "requests_last_minute": len(self.request_times),
                "tokens_last_minute": sum(self.token_counts),
                "rpm_percentage": (len(self.request_times) / self.max_rpm) * 100,
                "tpm_percentage": (sum(self.token_counts) / self.max_tpm) * 100,
                "consecutive_failures": self.consecutive_failures
            }