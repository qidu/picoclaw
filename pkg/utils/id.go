package utils

import (
	"crypto/rand"
	"encoding/hex"
	"time"
)

// GenerateRunID generates a unique run ID (OpenClaw pattern).
func GenerateRunID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// GenerateSessionID generates a unique session ID.
func GenerateSessionID() string {
	timestamp := time.Now().UnixNano()
	b := make([]byte, 4)
	rand.Read(b)
	return timestampString(timestamp) + "-" + hex.EncodeToString(b)
}

func timestampString(ts int64) string {
	// Use base36 for shorter representation
	const chars = "0123456789abcdefghijklmnopqrstuvwxyz"
	result := ""
	for ts > 0 {
		result = string(chars[ts%36]) + result
		ts /= 36
	}
	if result == "" {
		result = "0"
	}
	return result
}