// Package rlm provides a thin wrapper around the rrlm-solve CLI for
// recursive language model inference.
package rlm

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
)

// SolveResult is the JSON shape returned by rrlm-solve --json.
type SolveResult struct {
	Answer string `json:"answer"`
}

// Solve runs rrlm-solve with the given instruction and data, returning the
// answer string. Data is written to a temporary file and passed as @<tmpfile>.
// If the command fails, the stderr is returned as the error.
func Solve(instruction, data string) (string, error) {
	tmp, err := os.CreateTemp("", "rlm-data-*.txt")
	if err != nil {
		return "", fmt.Errorf("create tmp file: %w", err)
	}
	defer os.Remove(tmp.Name())

	if _, err := tmp.WriteString(data); err != nil {
		tmp.Close()
		return "", fmt.Errorf("write tmp file: %w", err)
	}
	tmp.Close()

	cmd := exec.Command("rrlm-solve",
		"-i", instruction,
		"-d", "@"+tmp.Name(),
		"--json",
	)
	out, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.Stderr != nil {
			return "", fmt.Errorf("rrlm-solve: %s", string(exitErr.Stderr))
		}
		return "", fmt.Errorf("rrlm-solve: %v", err)
	}

	var result SolveResult
	if err := json.Unmarshal(out, &result); err != nil {
		return "", fmt.Errorf("parse rrlm-solve output: %w (raw=%s)", err, string(out))
	}
	return result.Answer, nil
}
