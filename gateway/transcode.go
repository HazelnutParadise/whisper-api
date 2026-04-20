package gateway

import (
	"bytes"
	"fmt"
	"io"
	"os/exec"
	"strings"
)

func contentTypeForFormat(format string) string {
	switch format {
	case "mp3":
		return "audio/mpeg"
	case "opus":
		return "audio/ogg"
	case "aac":
		return "audio/aac"
	case "flac":
		return "audio/flac"
	case "wav":
		return "audio/wav"
	default:
		return "audio/pcm"
	}
}

func ffmpegOutputFormat(format string) string {
	switch format {
	case "mp3", "opus", "aac", "flac", "wav":
		return format
	default:
		return "s16le"
	}
}

func buildAtempoFilter(speed float64) string {
	if speed == 1 {
		return ""
	}

	factors := make([]string, 0, 4)
	remaining := speed

	for remaining > 2.0 {
		factors = append(factors, "atempo=2.0")
		remaining /= 2.0
	}
	for remaining < 0.5 {
		factors = append(factors, "atempo=0.5")
		remaining /= 0.5
	}
	factors = append(factors, fmt.Sprintf("atempo=%g", remaining))
	return strings.Join(factors, ",")
}

func transcodePCM(input []byte, responseFormat string, speed float64) ([]byte, error) {
	cmdArgs := []string{
		"-f", "s16le",
		"-ar", fmt.Sprintf("%d", DefaultSamplingRate),
		"-ac", "1",
		"-i", "pipe:0",
	}

	if filter := buildAtempoFilter(speed); filter != "" {
		cmdArgs = append(cmdArgs, "-filter:a", filter)
	}

	cmdArgs = append(cmdArgs, "-f", ffmpegOutputFormat(responseFormat), "pipe:1")
	cmd := exec.Command("ffmpeg", cmdArgs...)
	cmd.Stdin = bytes.NewReader(input)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg transcode failed: %w: %s", err, stderr.String())
	}
	return stdout.Bytes(), nil
}

func streamTranscodePCM(input io.Reader, output io.Writer, responseFormat string, speed float64) error {
	cmdArgs := []string{
		"-f", "s16le",
		"-ar", fmt.Sprintf("%d", DefaultSamplingRate),
		"-ac", "1",
		"-i", "pipe:0",
	}
	if filter := buildAtempoFilter(speed); filter != "" {
		cmdArgs = append(cmdArgs, "-filter:a", filter)
	}
	cmdArgs = append(cmdArgs, "-f", ffmpegOutputFormat(responseFormat), "pipe:1")

	cmd := exec.Command("ffmpeg", cmdArgs...)
	cmd.Stdin = input
	cmd.Stdout = output
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg streaming transcode failed: %w: %s", err, stderr.String())
	}
	return nil
}
