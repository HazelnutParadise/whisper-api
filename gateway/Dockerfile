FROM golang:1.24.1-bookworm AS builder

WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
COPY gateway ./gateway

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /out/gateway ./cmd/gateway

FROM debian:bookworm-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/gateway /usr/local/bin/gateway

ENV GATEWAY_ADDR=:5000

CMD ["gateway"]
