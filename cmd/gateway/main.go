package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"whisperapi/gateway"
)

func main() {
	cfg := gateway.NewConfigFromEnv()
	addr := os.Getenv("GATEWAY_ADDR")
	if addr == "" {
		addr = ":5000"
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := gateway.Serve(ctx, cfg, addr); err != nil {
		log.Fatal(err)
	}
}
