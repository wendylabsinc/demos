package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"time"

	pb "github.com/wendylabsinc/demos/go2-teleop/internal/agentpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const forwardLimit = .15
const yawLimit = .3
const inputBudget = 350 * time.Millisecond

type command struct {
	Challenge string  `json:"challenge"`
	Forward   float64 `json:"forward"`
	Yaw       float64 `json:"yaw"`
	Stop      bool    `json:"stop"`
}
type ready struct {
	Challenge string `json:"challenge"`
	TimeoutMS uint32 `json:"timeout_ms"`
	Mode      string `json:"mode"`
}
type controller interface {
	Initial() ready
	Exchange(context.Context, command) (ready, error)
	Close()
}
type backend interface {
	Open(context.Context) (controller, error)
	Mode() string
}

func bounded(c command) bool {
	return !math.IsNaN(c.Forward) && !math.IsInf(c.Forward, 0) && math.Abs(c.Forward) <= forwardLimit && !math.IsNaN(c.Yaw) && !math.IsInf(c.Yaw, 0) && math.Abs(c.Yaw) <= yawLimit
}
func nonce() string {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		panic("secure random unavailable")
	}
	return hex.EncodeToString(b)
}

type fakeBackend struct{}

func (fakeBackend) Mode() string { return "fake — no robot connected" }
func (fakeBackend) Open(ctx context.Context) (controller, error) {
	return &fakeController{ready: ready{Challenge: nonce(), TimeoutMS: 350, Mode: fakeBackend{}.Mode()}}, nil
}

type fakeController struct {
	ready  ready
	closed bool
}

func (c *fakeController) Initial() ready { return c.ready }
func (c *fakeController) Exchange(ctx context.Context, f command) (ready, error) {
	if c.closed || ctx.Err() != nil || f.Challenge != c.ready.Challenge || !bounded(f) {
		return ready{}, errors.New("invalid or expired control input")
	}
	if f.Stop {
		c.closed = true
		return ready{}, io.EOF
	}
	c.ready.Challenge = nonce()
	return c.ready, nil
}
func (c *fakeController) Close() { c.closed = true }

type agentBackend struct {
	client pb.WendyUnitreeTeleopServiceClient
	serial string
}

func (b agentBackend) Mode() string {
	return "hardware — paired serial " + b.serial + "; verify the physical robot before Enable"
}

// This plaintext credential setting is ONLY for the host-private Unix socket,
// never TCP. No robot keys or Wendy certificates are read/copied by this demo.
func newAgentBackend(socket, serial string) (agentBackend, *grpc.ClientConn, error) {
	if socket == "" || socket[0] != '/' || serial == "" {
		return agentBackend{}, nil, errors.New("hardware needs exact serial and absolute private agent socket path")
	}
	conn, err := grpc.NewClient("unix://"+socket, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return agentBackend{}, nil, err
	}
	return agentBackend{pb.NewWendyUnitreeTeleopServiceClient(conn), serial}, conn, nil
}
func (b agentBackend) Open(ctx context.Context) (controller, error) {
	ctx, cancel := context.WithCancel(ctx)
	stream, err := b.client.Control(ctx)
	if err != nil {
		cancel()
		return nil, errors.New("agent control unavailable")
	}
	c := &agentController{stream: stream, serial: b.serial, mode: b.Mode(), cancel: cancel}
	openCtx, openCancel := context.WithTimeout(ctx, 2*time.Second)
	defer openCancel()
	r, err := c.roundtrip(openCtx, &pb.UnitreeTeleopFrame{Serial: b.serial})
	if err != nil {
		cancel()
		return nil, errors.New("agent refused control; check pairing and operator enable setting")
	}
	c.ready = r
	return c, nil
}

type agentController struct {
	stream grpc.BidiStreamingClient[pb.UnitreeTeleopFrame, pb.UnitreeTeleopReady]
	serial string
	mode   string
	ready  ready
	cancel context.CancelFunc
}

func (c *agentController) Initial() ready { return c.ready }
func (c *agentController) Exchange(ctx context.Context, f command) (ready, error) {
	if f.Challenge != c.ready.Challenge || !bounded(f) {
		return ready{}, errors.New("invalid control input")
	}
	r, err := c.roundtrip(ctx, &pb.UnitreeTeleopFrame{Serial: c.serial, Challenge: f.Challenge, Forward: f.Forward, Yaw: f.Yaw, Stop: f.Stop})
	c.ready = r
	return r, err
}
func (c *agentController) roundtrip(ctx context.Context, f *pb.UnitreeTeleopFrame) (ready, error) {
	type result struct {
		r   *pb.UnitreeTeleopReady
		err error
	}
	out := make(chan result, 1)
	go func() {
		if err := c.stream.Send(f); err != nil {
			out <- result{err: err}
			return
		}
		r, err := c.stream.Recv()
		out <- result{r, err}
	}()
	select {
	case <-ctx.Done():
		c.cancel()
		return ready{}, ctx.Err()
	case v := <-out:
		if v.err != nil {
			return ready{}, v.err
		}
		if v.r == nil || len(v.r.Challenge) != 32 || v.r.TimeoutMs == 0 || v.r.TimeoutMs > 350 {
			c.cancel()
			return ready{}, errors.New("invalid agent control challenge")
		}
		return ready{Challenge: v.r.Challenge, TimeoutMS: v.r.TimeoutMs, Mode: c.mode}, nil
	}
}
func (c *agentController) Close() { c.cancel() }
