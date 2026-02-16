// PicoClaw - Ultra-lightweight personal AI agent
// HTTP client utilities with custom DNS support

package httpclient

import (
	"context"
	"net"
	"net/http"
	"os"
	"strings"
	"time"
)

// isTermux returns true if running in Termux environment
func IsTermux() bool {
	// Check for Termux environment variable
	return strings.Contains(os.Getenv("PREFIX"), "com.termux")
}

// createDNSResolver creates a resolver that uses Google DNS (8.8.8.8)
func createDNSResolver() *net.Resolver {
	return &net.Resolver{
		PreferGo: true,
		Dial: func(ctx context.Context, network, address string) (net.Conn, error) {
			d := &net.Dialer{
				Timeout: 5 * time.Second,
			}
			// Use TCP4 to avoid IPv6 issues on Termux
			return d.DialContext(ctx, "tcp4", "8.8.8.8:53")
		},
	}
}

// resolveHost resolves a hostname to IPv4 using Google DNS
func ResolveHost(host string) ([]net.IP, error) {
	resolver := createDNSResolver()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return resolver.LookupIP(ctx, "ip", host)
}

// NewClient returns an HTTP client configured with IPv4-only connections
// and custom DNS resolution for Termux compatibility
func NewClient(timeout time.Duration) *http.Client {
	dialer := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}

	transport := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			// Split host and port
			host, port, err := net.SplitHostPort(addr)
			if err != nil {
				return dialer.DialContext(ctx, "tcp4", addr)
			}

			// Use custom DNS resolver to avoid localhost DNS issues
			if IsTermux() {
				resolver := createDNSResolver()
				ctxDNS, cancelDNS := context.WithTimeout(ctx, 5*time.Second)
				defer cancelDNS()

				ips, err := resolver.LookupIP(ctxDNS, "ip", host)
				if err != nil || len(ips) == 0 {
					// Fallback: try direct connection
					return dialer.DialContext(ctx, "tcp4", addr)
				}

				// Use first IPv4 address
				target := net.JoinHostPort(ips[0].String(), port)
				return dialer.DialContext(ctx, "tcp4", target)
			}

			// Standard: just use IPv4
			return dialer.DialContext(ctx, "tcp4", addr)
		},
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
	}

	return &http.Client{
		Transport: transport,
		Timeout:   timeout,
	}
}

// DefaultClient returns a client with 30-second timeout
func DefaultClient() *http.Client {
	return NewClient(30 * time.Second)
}

// ShortClient returns a client with 10-second timeout (for web lookups)
func ShortClient() *http.Client {
	return NewClient(10 * time.Second)
}