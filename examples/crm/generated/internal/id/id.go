// Package id provides short unique identifiers and timestamp helpers used across
// LadyCRM. Timestamps are RFC3339 strings, suitable as TIMESTAMP query params.
package id

import (
	"crypto/rand"
	"encoding/hex"
	"time"
)

// New returns a short, random, unique identifier (16 hex chars).
func New() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// Now returns the current UTC time as an RFC3339 string.
func Now() string {
	return time.Now().UTC().Format(time.RFC3339)
}
